import math
from typing import Optional

import einops
import torch
from torch import nn

from vae.models.predictor.joint_fm_dit_blocks import (
    AdaptivePreNorm,
    Attention,
    FeedForward,
    FinalLayer,
    RotaryPositionalEmbeddings,
    TimestepEmbedder,
)


class JointTransformerEncoderSeparableAttention(nn.Module):
    """
    Separable attention backbone reused from the original predictor, generalized
    for joint-token modeling (action / optional proprio / registers / visual).
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale_shift = nn.Parameter(torch.randn(9, dim) / dim**0.5)

        head_dim = dim // heads
        self.rope_temporal = RotaryPositionalEmbeddings(head_dim)
        self.h_dim = head_dim // 2
        self.w_dim = head_dim - self.h_dim
        self.rope_h = RotaryPositionalEmbeddings(self.h_dim)
        self.rope_w = RotaryPositionalEmbeddings(self.w_dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        AdaptivePreNorm(
                            dim,
                            Attention(
                                dim,
                                heads,
                                dropout=dropout,
                                use_qk_norm=True,
                            ),
                        ),
                        AdaptivePreNorm(
                            dim,
                            Attention(
                                dim,
                                heads,
                                dropout=dropout,
                                use_qk_norm=True,
                            ),
                        ),
                        AdaptivePreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def _build_spatial_freqs(
        self,
        h: int,
        w: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        freqs_h = self.rope_h(h, device)
        freqs_h = einops.repeat(freqs_h, "h d -> h w d", w=w)

        freqs_w = self.rope_w(w, device)
        freqs_w = einops.repeat(freqs_w, "w d -> h w d", h=h)

        visual_freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        visual_freqs = einops.rearrange(visual_freqs, "h w d -> (h w) d")

        visual_tokens = h * w
        if tokens_per_frame < visual_tokens:
            raise ValueError(
                f"tokens_per_frame ({tokens_per_frame}) < visual_tokens ({visual_tokens})"
            )
        return visual_freqs.unsqueeze(0)

    def forward(
        self,
        x,
        mod,
        full_shape,
        spatial_attn_mask: Optional[torch.Tensor] = None,
    ):
        b, t, h, w, _ = full_shape
        tokens_per_frame = x.shape[1] // t

        mod = mod.reshape(b, t, 9, -1)
        (
            temp_shift_msa,
            temp_scale_msa,
            temp_gate_msa,
            spatial_shift_msa,
            spatial_scale_msa,
            spatial_gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (self.scale_shift[None, None, :, :] + mod).unbind(dim=2)

        temp_freqs = self.rope_temporal(t, x.device).unsqueeze(0)
        spatial_freqs = self._build_spatial_freqs(h, w, tokens_per_frame, x.device)
        visual_tokens = h * w
        visual_start = tokens_per_frame - visual_tokens
        if visual_start < 0:
            raise ValueError(
                f"Invalid token layout: tokens_per_frame={tokens_per_frame}, visual_tokens={visual_tokens}"
            )

        for attn_temporal, attn_spatial, ff in self.layers:
            x_temp = einops.rearrange(
                x, "b (t n) c -> (b n) t c", t=t, n=tokens_per_frame
            )
            ts, tsc, tg = [
                einops.repeat(p, "b t d -> (b n) t d", n=tokens_per_frame)
                for p in (temp_shift_msa, temp_scale_msa, temp_gate_msa)
            ]
            x_temp = attn_temporal(x_temp, ts, tsc, tg, rotary_freqs=temp_freqs)
            x = x + einops.rearrange(x_temp, "(b n) t c -> b (t n) c", b=b, t=t)

            x_spatial = einops.rearrange(
                x, "b (t n) c -> (b t) n c", t=t, n=tokens_per_frame
            )
            ss, ssc, sg = [
                einops.rearrange(p, "b t d -> (b t) 1 d")
                for p in (spatial_shift_msa, spatial_scale_msa, spatial_gate_msa)
            ]
            x_spatial_out = attn_spatial(
                x_spatial,
                ss,
                ssc,
                sg,
                rotary_freqs=spatial_freqs,
                attn_mask=spatial_attn_mask,
                rotary_start_index=visual_start,
            )
            x = x + einops.rearrange(x_spatial_out, "(b t) n c -> b (t n) c", b=b, t=t)

            shift_mlp_expand, scale_mlp_expand, gate_mlp_expand = [
                einops.repeat(p, "b t d -> b (t n) d", n=tokens_per_frame)
                for p in (shift_mlp, scale_mlp, gate_mlp)
            ]
            x = x + ff(x, shift_mlp_expand, scale_mlp_expand, gate_mlp_expand)

        return x, None


class JointMaskTransformerFM(nn.Module):
    """
    Joint FM backbone in latent space:
      - token stream is UWM-style prediction window only:
        [action] [optional proprio] [registers] [future visual patches],
      - context (state / optional proprio) is encoded as global AdaLN condition,
      - decoupled action/state timestep conditioning is preserved for FM mode.
    """

    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        action_dim=10,
        proprio_dim=4,
        use_proprio: bool = True,
        context_frames=1,
        prediction_frames=1,
        num_registers=8,
        fm_variant: str = "fm",
        use_timestep: Optional[bool] = None,
    ):
        super().__init__()
        self.init_std = 0.02
        self.hidden_dim = hidden_dim
        self.prediction_frames = int(prediction_frames)
        self.num_registers = int(num_registers)

        variant = str(fm_variant).lower()
        if use_timestep is None:
            if variant in {"eqm", "eqm_no_t", "eqm_no_time", "equilibrium"}:
                self.use_timestep = False
            elif variant in {"fm", "cfm", "flow"}:
                self.use_timestep = True
            else:
                raise ValueError(f"Unsupported fm_variant: {fm_variant}")
        else:
            self.use_timestep = bool(use_timestep)
        self.fm_variant = variant

        self.action_dim = int(action_dim) if action_dim is not None else 0
        self.has_action = self.action_dim > 0
        if not self.has_action:
            raise ValueError("Joint FM requires action_dim > 0 for action branch.")

        self.proprio_dim = int(proprio_dim) if proprio_dim is not None else 0
        self.has_proprio = bool(use_proprio)
        if self.has_proprio and self.proprio_dim <= 0:
            raise ValueError("use_proprio=True requires proprio_dim > 0.")

        # Keep the same action/proprio encoding style as the original predictor code.
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.has_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(self.proprio_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proprio_context_proj = nn.Sequential(
                nn.Linear(self.proprio_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.proprio_context_proj = None

        self.state_context_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self.use_timestep:
            self.action_time_embd = TimestepEmbedder(hidden_size=hidden_dim)
            self.state_time_embd = TimestepEmbedder(hidden_size=hidden_dim)
            self.time_fuse = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.action_time_embd = None
            self.state_time_embd = None
            self.time_fuse = None

        # Fuse global context and (optional) dual timestep embeddings.
        self.cond_fuse = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim, bias=True),
        )

        self.transformer = JointTransformerEncoderSeparableAttention(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.fc_in = nn.Linear(embedding_dim, hidden_dim)
        self.register_tokens = nn.Parameter(
            torch.zeros(1, 1, self.num_registers, hidden_dim)
        )

        # Readout heads
        self.action_out = FinalLayer(hidden_dim, self.action_dim, cond_dim=hidden_dim)
        self.fc_out = FinalLayer(hidden_dim, embedding_dim, cond_dim=hidden_dim)
        if self.has_proprio:
            self.proprio_out = FinalLayer(hidden_dim, self.proprio_dim, cond_dim=hidden_dim)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.register_tokens, std=self.init_std)

        # Zero-init AdaLN/style output heads as in DiT-style training.
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

        nn.init.constant_(self.action_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.action_out.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.action_out.linear.weight, 0)
        nn.init.constant_(self.action_out.linear.bias, 0)

        nn.init.constant_(self.fc_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.fc_out.linear.weight, 0)
        nn.init.constant_(self.fc_out.linear.bias, 0)

        if self.has_proprio:
            nn.init.constant_(self.proprio_out.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.proprio_out.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.proprio_out.linear.weight, 0)
            nn.init.constant_(self.proprio_out.linear.bias, 0)

        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.transformer.layers):
            rescale(layer[0].fn.project.weight.data, layer_id + 1)
            rescale(layer[1].fn.project.weight.data, layer_id + 1)
            rescale(layer[2].fn.net[3].weight.data, layer_id + 1)

    def _build_condition(
        self,
        action_time: Optional[torch.Tensor],
        next_state_time: Optional[torch.Tensor],
        context_state: Optional[torch.Tensor],
        context_proprio: Optional[torch.Tensor],
        b: int,
        t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Context state -> global condition
        if context_state is None:
            state_ctx = torch.zeros(b, self.hidden_dim, device=device, dtype=dtype)
        else:
            if context_state.ndim != 5:
                raise ValueError(
                    f"context_state must have shape [B, ctx, h, w, c], got {tuple(context_state.shape)}"
                )
            pooled_state = context_state.mean(dim=(1, 2, 3))
            state_ctx = self.state_context_proj(pooled_state.to(dtype=dtype))

        # Context proprio -> global condition
        if self.has_proprio and context_proprio is not None:
            if context_proprio.ndim != 3 or context_proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"context_proprio must have shape [B, ctx, {self.proprio_dim}], got {tuple(context_proprio.shape)}"
                )
            pooled_prop = context_proprio.mean(dim=1)
            proprio_ctx = self.proprio_context_proj(pooled_prop.to(dtype=dtype))
        else:
            proprio_ctx = torch.zeros(b, self.hidden_dim, device=device, dtype=dtype)

        # Dual-time FM condition (or zeros for EqM mode)
        if not self.use_timestep:
            time_ctx = torch.zeros(b, self.hidden_dim, device=device, dtype=dtype)
        else:
            if action_time is None or next_state_time is None:
                raise ValueError("action_time and next_state_time are required in FM mode")

            action_time = action_time.reshape(-1)
            next_state_time = next_state_time.reshape(-1)
            if action_time.shape[0] != b or next_state_time.shape[0] != b:
                raise ValueError(
                    "action_time and next_state_time must have shape [B] or be broadcastable to [B]."
                )

            a_emb = self.action_time_embd(action_time)
            s_emb = self.state_time_embd(next_state_time)
            time_ctx = self.time_fuse(torch.cat([a_emb, s_emb], dim=-1))

        cond_global = self.cond_fuse(torch.cat([state_ctx, proprio_ctx, time_ctx], dim=-1))
        cond = einops.repeat(cond_global, "b d -> b t d", t=t)
        return cond.to(dtype=dtype)

    def _encode_token_sequence(
        self,
        value: Optional[torch.Tensor],
        encoder: nn.Module,
        feature_dim: int,
        expected_len: int,
    ) -> torch.Tensor:
        if value is None:
            raise ValueError(
                f"Missing token input for feature_dim={feature_dim}; expected shape [B, {expected_len}, {feature_dim}]"
            )

        if value.ndim != 3 or value.shape[-1] != feature_dim:
            raise ValueError(
                f"Expected token input shape [B, T, {feature_dim}], got {tuple(value.shape)}"
            )

        if value.size(1) != expected_len:
            raise ValueError(
                f"Token input length {value.size(1)} is incompatible with prediction_frames={expected_len}."
            )
        return encoder(value)

    def _build_policy_spatial_mask(
        self,
        tokens_per_frame: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build additive attention mask for per-frame spatial attention:
          - block action query token from attending to visual tokens.
          - keep all other query/key pairs unchanged.
        """
        visual_tokens = h * w
        visual_start = tokens_per_frame - visual_tokens

        mask = torch.zeros(
            1,
            1,
            tokens_per_frame,
            tokens_per_frame,
            device=device,
            dtype=torch.float32,
        )
        # action token is always the first token in each frame.
        mask[:, :, 0, visual_start:] = -1e9
        return mask

    def forward(
        self,
        x: torch.Tensor,
        noisy_action: Optional[torch.Tensor],
        action_time: Optional[torch.Tensor],
        next_state_time: Optional[torch.Tensor],
        noisy_proprio: Optional[torch.Tensor] = None,
        context_state: Optional[torch.Tensor] = None,
        context_proprio: Optional[torch.Tensor] = None,
        mask_action_from_visual: bool = False,
    ):
        """
        Args:
            x: [B, pred, h, w, c], noised future visual latents only.
            noisy_action: [B, pred, action_dim].
            action_time: [B]
            next_state_time: [B]
            noisy_proprio: [B, pred, proprio_dim].
            context_state: [B, context, h, w, c], used as global AdaLN condition.
            context_proprio: [B, context, proprio_dim], optional global condition.
        Returns:
            action_pred: [B, pred, action_dim]
            state_pred: [B, pred, h, w, c]
            proprio_pred(optional): [B, pred, proprio_dim]
        """
        b, t, h, w, c = x.size()
        if t != self.prediction_frames:
            raise ValueError(
                f"Expected prediction_frames={self.prediction_frames}, got {t} in transformer input"
            )

        cond = self._build_condition(
            action_time,
            next_state_time,
            context_state,
            context_proprio,
            b,
            t,
            x.device,
            x.dtype,
        )
        mod = self.modulation(cond)

        vis_tokens = self.fc_in(x)
        vis_tokens = einops.rearrange(vis_tokens, "b t h w c -> b t (h w) c")

        action_tok = self._encode_token_sequence(
            noisy_action,
            self.action_encoder,
            self.action_dim,
            self.prediction_frames,
        )
        action_tok = action_tok.unsqueeze(2)

        token_list = [action_tok]
        if self.has_proprio:
            proprio_tok = self._encode_token_sequence(
                noisy_proprio,
                self.proprio_encoder,
                self.proprio_dim,
                self.prediction_frames,
            )
            proprio_tok = proprio_tok.unsqueeze(2)
            token_list.append(proprio_tok)

        registers = self.register_tokens.expand(b, t, -1, -1)
        token_list.append(registers)
        token_list.append(vis_tokens)

        tokens = torch.cat(token_list, dim=2)
        tokens_per_frame = tokens.shape[2]

        spatial_attn_mask = None
        if mask_action_from_visual:
            spatial_attn_mask = self._build_policy_spatial_mask(
                tokens_per_frame=tokens_per_frame,
                h=h,
                w=w,
                device=x.device,
            )

        x_flat = einops.rearrange(tokens, "b t n c -> b (t n) c")
        x_flat, _ = self.transformer(
            x_flat,
            mod,
            full_shape=[b, t, h, w, c],
            spatial_attn_mask=spatial_attn_mask,
        )
        x = einops.rearrange(x_flat, "b (t n) c -> b t n c", t=t, n=tokens_per_frame)
        x_future = x
        cond_future = cond

        idx = 0

        a_out = x_future[:, :, idx, :]
        idx += 1
        a_out_flat = einops.rearrange(a_out, "b t c -> (b t) c")
        cond_flat = einops.rearrange(cond_future, "b t d -> (b t) d")
        action_pred = self.action_out(a_out_flat, cond_flat)
        action_pred = einops.rearrange(
            action_pred,
            "(b t) c -> b t c",
            b=b,
            t=self.prediction_frames,
        )

        proprio_pred = None
        if self.has_proprio:
            p_out = x_future[:, :, idx, :]
            idx += 1
            p_out_flat = einops.rearrange(p_out, "b t c -> (b t) c")
            proprio_pred = self.proprio_out(p_out_flat, cond_flat)
            proprio_pred = einops.rearrange(
                proprio_pred,
                "(b t) c -> b t c",
                b=b,
                t=self.prediction_frames,
            )

        idx += self.num_registers
        vis_out = x_future[:, :, idx : idx + h * w, :]

        vis_out_flat = einops.rearrange(vis_out, "b t n c -> (b t n) c")
        cond_expanded_vis = einops.repeat(cond_future, "b t d -> (b t n) d", n=h * w)
        state_pred = self.fc_out(vis_out_flat, cond_expanded_vis)
        state_pred = einops.rearrange(
            state_pred,
            "(b t h w) c -> b t h w c",
            b=b,
            t=self.prediction_frames,
            h=h,
            w=w,
        )

        return action_pred, state_pred, proprio_pred
