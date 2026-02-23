import math
import einops
import torch
import torch.nn.functional as F
from torch import nn

# ==========================================
# 1. RoPE, Utils
# ==========================================

class RotaryPositionalEmbeddings(nn.Module):
    """
    standard 1D RoPE calculatror.
    to generate frequency of Time, Height, Width and combined in Attention.
    """
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        # dim has to be even for RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        # generate [seq_len, dim/2]
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # concat to [seq_len, dim] -> for required shape of cos/sin
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb 

def apply_rotary_pos_emb(t, freqs):
    """
    t: [B, Seq, Heads, Dim]
    freqs: [Seq, Dim] or [1, Seq, Dim] or broadcastable
    """
    # make sure freqs dimensions match
    if freqs.dim() == 2:
        freqs = freqs.unsqueeze(0).unsqueeze(2) # [1, Seq, 1, Dim]
    elif freqs.dim() == 3:
        freqs = freqs.unsqueeze(2) # [1, Seq, 1, Dim]

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., norm=nn.RMSNorm, use_qk_norm=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv_attn = nn.Linear(embed_dim, embed_dim * 3)
        self.project = nn.Linear(embed_dim, embed_dim)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = norm(self.head_dim)
            self.k_norm = norm(self.head_dim)

    def forward(self, x: torch.Tensor, rotary_freqs=None):
        B, N, C = x.size()
        qkv = self.qkv_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE applied
        if rotary_freqs is not None:
            q = apply_rotary_pos_emb(q, rotary_freqs)
            k = apply_rotary_pos_emb(k, rotary_freqs)

        q = q.transpose(1, 2) # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        y = y.transpose(1, 2).contiguous().view(B, N, C)
        y = self.project(y)
        return y, None

class AdaptivePreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fn = fn

    def forward(self, x, shift, scale, gate, **kwargs):
        # 1. Modulation
        out = self.fn(
            modulate(self.norm(x), shift, scale),
            **kwargs
        )
        if isinstance(self.fn, Attention):
            f, _ = out
        else:
            f = out
        # 2. Gating
        return gate * f

# ==========================================
# 2. Transformer Encoder (3D RoPE + AdaLN)
# ==========================================

class TransformerEncoderSeparableAttention(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        # AdaLN parameters (Time + Space + MLP) * 3 params = 9
        # （3modules * 3params)
        self.scale_shift = nn.Parameter(torch.randn(9, dim) / dim**0.5)

        head_dim = dim // heads
        
        # === 3D RoPE settings ===
        # 1. Temporal RoPE
        self.rope_temporal = RotaryPositionalEmbeddings(head_dim)
        
        # 2. Axial RoPE (Height + Width)
        self.h_dim = head_dim // 2
        self.w_dim = head_dim - self.h_dim
        self.rope_h = RotaryPositionalEmbeddings(self.h_dim)
        self.rope_w = RotaryPositionalEmbeddings(self.w_dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AdaptivePreNorm(dim, Attention(dim, heads, dropout=dropout, use_qk_norm=True)), # Temporal
                AdaptivePreNorm(dim, Attention(dim, heads, dropout=dropout, use_qk_norm=True)), # Spatial
                AdaptivePreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, time, full_shape, has_proprio=False):
        b, t, h, w, d = full_shape
        device = x.device
        
        # [修改] AdaLN (Per-frame) Modulation
        # input 'time' (mod) shape: [B, T, 9*D]
        # Reshape to [B, T, 9, D]
        time = time.reshape(b, t, 9, -1)
        
        # Add parameter bias. scale_shift is [9, D] -> Broadcast to [1, 1, 9, D]
        params = self.scale_shift[None, None, :, :] + time
        
        # Unbind into 9 tensors, each is [B, T, D]
        (
            temp_shift_msa, temp_scale_msa, temp_gate_msa,
            spatial_shift_msa, spatial_scale_msa, spatial_gate_msa,
            shift_mlp, scale_mlp, gate_mlp
        ) = params.unbind(dim=2)

        # ==========================================
        # 3D RoPE Frequencies Calculation
        # ==========================================
        
        # 1. Temporal Freqs [1, T, D]
        temp_freqs = self.rope_temporal(t, device).unsqueeze(0) 

        # 2. Spatial Freqs (Axial: Height + Width)
        freqs_h = self.rope_h(h, device) 
        freqs_h = einops.repeat(freqs_h, 'h d -> h w d', w=w)
        
        freqs_w = self.rope_w(w, device)
        freqs_w = einops.repeat(freqs_w, 'w d -> h w d', h=h)
        
        spatial_freqs_vis = torch.cat([freqs_h, freqs_w], dim=-1)
        spatial_freqs_vis = einops.rearrange(spatial_freqs_vis, 'h w d -> (h w) d')
        
        if has_proprio:
            # Proprio Token do not take rotation (freq=0)
            proprio_freqs = torch.zeros(1, spatial_freqs_vis.shape[-1], device=device)
            spatial_freqs = torch.cat([proprio_freqs, spatial_freqs_vis], dim=0) 
        else:
            spatial_freqs = spatial_freqs_vis
            
        spatial_freqs = spatial_freqs.unsqueeze(0) # [1, N, D]

        # ==========================================
        # Transformer Blocks
        # ==========================================
        
        tokens_per_frame = h * w + (1 if has_proprio else 0)

        for attn_temporal, attn_spatial, ff in self.layers:
            # --- 1. Temporal Attention ---
            # x: (b n) t c
            x_temp = einops.rearrange(x, 'b (t n) c -> (b n) t c', t=t, n=tokens_per_frame)
            
            # Broadcast AdaLN for Temporal: [B, T, D] -> [(B*N), T, D]
            # 每一个空间 token 在时间轴上都有对应的那一帧的 action modulation
            # every spatial token has corresponding frame's action modulation along temporal axis
            ts, tsc, tg = [einops.repeat(p, 'b t d -> (b n) t d', n=tokens_per_frame) 
                           for p in (temp_shift_msa, temp_scale_msa, temp_gate_msa)]
                           
            x_temp = attn_temporal(x_temp, ts, tsc, tg, rotary_freqs=temp_freqs)
            x = x + einops.rearrange(x_temp, '(b n) t c -> b (t n) c', b=b, t=t)

            # --- 2. Spatial Attention ---
            # x: (b t) n c
            x_spatial = einops.rearrange(x, 'b (t n) c -> (b t) n c', t=t, n=tokens_per_frame)
            
            # Broadcast AdaLN for Spatial: [B, T, D] -> [(B*T), 1, D]
            # all spatial tokens of the same frame share the same frame's action modulation
            ss, ssc, sg = [einops.rearrange(p, 'b t d -> (b t) 1 d') 
                           for p in (spatial_shift_msa, spatial_scale_msa, spatial_gate_msa)]
                           
            x_spatial_out = attn_spatial(x_spatial, ss, ssc, sg, rotary_freqs=spatial_freqs)
            
            x = x + einops.rearrange(x_spatial_out, '(b t) n c -> b (t n) c', b=b, t=t)

            # --- 3. Feed Forward ---
            # x: b (t n) c
            # Broadcast AdaLN for MLP: [B, T, D] -> [B, (T*N), D]
            # need to align with flattened sequence
            shift_mlp_expand, scale_mlp_expand, gate_mlp_expand = [
                einops.repeat(p, 'b t d -> b (t n) d', n=tokens_per_frame)
                for p in (shift_mlp, scale_mlp, gate_mlp)
            ]
            x = x + ff(x, shift_mlp_expand, scale_mlp_expand, gate_mlp_expand)

        return x, None

# ==========================================
# 3. Embedder & FinalLayer
# ==========================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# ==========================================
# 4. Transformer
# ==========================================

class MaskTransformer(nn.Module):
    def __init__(
        self,
        shape=None, # keep port, do not need at the moment, determined by input
        embedding_dim=768, # DINO Latent Dimension
        hidden_dim=768,    # Model Internal Dimension
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        action_dim=10,    # [16, ctx+pred, 2*frameskip]
        proprio_dim=4,   # [16, ctx+pred, 4]
        context_frames=3, 
        prediction_frames=1 #
    ):
        super().__init__()
        self.init_std = 0.02
        self.hidden_dim = hidden_dim
        
        self.context_frames = context_frames
        self.prediction_frames = prediction_frames
        self.total_frames = context_frames + prediction_frames
        
        # 1. Embedders
        self.time_embd = TimestepEmbedder(hidden_size=hidden_dim)
        
        self.action_dim = action_dim
        if action_dim is not None:
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim), 
                nn.SiLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )
            
        self.proprio_dim = proprio_dim
        if proprio_dim is not None:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                nn.SiLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )

        # 2. AdaLN Condition
        # Condition = Time + Action(Per Frame)
        mod_input_dim = hidden_dim + (hidden_dim if action_dim else 0)
        # 9 * hidden_dim is the number of params of AdaLN
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_input_dim, 9 * hidden_dim, bias=True)
        )

        # 3. Backbone
        self.transformer = TransformerEncoderSeparableAttention(
            dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout
        )

        # 4. Projections
        self.fc_in = nn.Linear(embedding_dim, hidden_dim)
        
        # Output Heads
        self.fc_out = FinalLayer(hidden_dim, embedding_dim, cond_dim=mod_input_dim)
        if proprio_dim:
            self.proprio_out = FinalLayer(hidden_dim, proprio_dim, cond_dim=mod_input_dim)

        # 5. Initialization
        self.apply(self._init_weights)
        
        # Zero-Init for AdaLN (Standard DiT Practice)
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)
        
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.fc_out.linear.weight, 0)
        nn.init.constant_(self.fc_out.linear.bias, 0)
        
        if proprio_dim:
            nn.init.constant_(self.proprio_out.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.proprio_out.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.proprio_out.linear.weight, 0)
            nn.init.constant_(self.proprio_out.linear.bias, 0)
        
        # Rescale Residual Blocks
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

    def forward(self, x, time, action=None, proprio=None):
        """
        Input Shapes:
            x       : [B, Total_Frames, H, W, C] (concat: Context Frames + Predict Frames)
            time    : [B]             (Diffusion Timesteps)
            action  : [B, Total_Frames, 2]       (Action Trajectory)
            proprio : [B, Total_Frames, 4]       (Proprio Trajectory)
            
        Output Shapes:
            vis_pred: [B, Prediction_Frames, H, W, C] (Predicted Future Frames)
            pro_pred: [B, Prediction_Frames, 4]       (Predicted Future Proprio)
        """
        b, t, h, w, c = x.size() # t should be self.total_frames
        
        # === 1. Building Condition ===
        # Timestep Embedding: [B, D] -> [B, t, D]
        # every step has Time info
        t_emb = self.time_embd(time) 
        t_emb_seq = einops.repeat(t_emb, 'b d -> b t d', t=t)
        
        if self.action_dim and action is not None:
            # a_emb: [B, t, D]
            a_emb = self.action_encoder(action)
            # cond: [B, t, 2*D]
            cond = torch.cat([t_emb_seq, a_emb], dim=-1)
        else:
            cond = t_emb_seq
            
        mod = self.modulation(cond) # [B, t, 9*D]

        # === 2. Embedding Input ===
        x = self.fc_in(x) # [B, t, H, W, D]
        x = einops.rearrange(x, 'b t h w c -> b t (h w) c') # Flatten spatial

        has_proprio = False
        if self.proprio_dim and proprio is not None:
            # Proprio: [B, t, 4] -> [B, t, D]
            p_emb = self.proprio_encoder(proprio).unsqueeze(2) # [B, t, 1, D]
            x = torch.cat([p_emb, x], dim=2) # [B, t, 1+HW, D]
            has_proprio = True

        # === 3. Transformer Backbone (Full Sequence) ===
        # Flatten time for transformer input
        x = einops.rearrange(x, 'b t n c -> b (t n) c')
        
        # Transformer can see all the frames
        x, _ = self.transformer(x, mod, full_shape=[b, t, h, w, c], has_proprio=has_proprio)

        # === 4. Readout Future (Prediction) ===
        # recover time dimension
        num_tokens = (h * w) + (1 if has_proprio else 0)
        x = einops.rearrange(x, 'b (t n) c -> b t n c', t=t, n=num_tokens)

        # Get Prediction Frames
        future_indices = slice(self.context_frames, self.total_frames) 
        future_steps = self.prediction_frames 
        
        x_future = x[:, future_indices, :, :] # [B, prediction_frames, N, D]
        
        # extract future frame condition for FinalLayer
        # cond: [B, Total, D] -> [B, Pred, D]
        cond_future = cond[:, future_indices, :]

        # Proprio Output
        proprio_pred = None
        if has_proprio:
            p_out = x_future[:, :, 0, :] # [B, future_steps, D]
            vis_out = x_future[:, :, 1:, :] # [B, future_steps, HW, D]
            
            p_out_flat = einops.rearrange(p_out, 'b t c -> (b t) c')
            # flatten condition to match (b t)
            cond_flat = einops.rearrange(cond_future, 'b t d -> (b t) d')
            
            proprio_pred = self.proprio_out(p_out_flat, cond_flat)
            proprio_pred = einops.rearrange(proprio_pred, '(b t) c -> b t c', b=b, t=future_steps)
        else:
            vis_out = x_future

        # Visual Output
        vis_out_flat = einops.rearrange(vis_out, 'b t n c -> (b t n) c')
        
        # Expand condition for visual tokens
        # cond_future: [B, Pred, D] -> Broadcast to [B, Pred, HW, D] -> Flatten to [(B*Pred*HW), D]
        cond_expanded_vis = einops.repeat(cond_future, 'b t d -> (b t n) d', n=h*w)
        
        vis_pred = self.fc_out(vis_out_flat, cond_expanded_vis)
        
        # Reshape back to image grid
        vis_pred = einops.rearrange(vis_pred, '(b t h w) c -> b t h w c', b=b, t=future_steps, h=h, w=w)

        return vis_pred, proprio_pred


class MaskTransformerRegression(nn.Module):
    """
    Regression predictor that uses context frames only.
    Future latent tokens are learned query tokens (not noise), and AdaLN is
    conditioned by frame-level action embeddings only.
    """

    def __init__(
        self,
        shape=None,
        embedding_dim=768,
        hidden_dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        action_dim=10,
        proprio_dim=4,
        context_frames=3,
        prediction_frames=1
    ):
        super().__init__()
        self.init_std = 0.02
        self.hidden_dim = hidden_dim

        self.context_frames = context_frames
        self.prediction_frames = prediction_frames
        self.total_frames = context_frames + prediction_frames

        self.action_dim = action_dim
        if action_dim is not None:
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        self.proprio_dim = proprio_dim
        if proprio_dim is not None:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        self.mod_input_dim = hidden_dim
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.mod_input_dim, 9 * hidden_dim, bias=True)
        )

        self.transformer = TransformerEncoderSeparableAttention(
            dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout
        )

        self.fc_in = nn.Linear(embedding_dim, hidden_dim)

        # Learned queries for future frames in place of diffusion noise tokens.
        self.future_vis_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        if proprio_dim:
            self.future_proprio_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

        self.fc_out = FinalLayer(hidden_dim, embedding_dim, cond_dim=self.mod_input_dim)
        if proprio_dim:
            self.proprio_out = FinalLayer(hidden_dim, proprio_dim, cond_dim=self.mod_input_dim)

        self.apply(self._init_weights)

        nn.init.trunc_normal_(self.future_vis_token, std=self.init_std)
        if proprio_dim:
            nn.init.trunc_normal_(self.future_proprio_token, std=self.init_std)

        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

        nn.init.constant_(self.fc_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.fc_out.linear.weight, 0)
        nn.init.constant_(self.fc_out.linear.bias, 0)

        if proprio_dim:
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

    def _build_frame_condition(self, action, b, device, dtype):
        if self.action_dim and action is not None:
            if action.size(1) < self.total_frames:
                raise ValueError(
                    f"action length {action.size(1)} is shorter than total_frames {self.total_frames}"
                )
            return self.action_encoder(action[:, :self.total_frames, :])
        return torch.zeros(
            b, self.total_frames, self.mod_input_dim, device=device, dtype=dtype
        )

    def forward(self, x_ctx, action=None, proprio=None):
        """
        Input Shapes:
            x_ctx   : [B, Context_Frames, H, W, C]
            action  : [B, Total_Frames, action_dim]
            proprio : [B, Total_Frames or Context_Frames, proprio_dim]

        Output Shapes:
            vis_pred: [B, Prediction_Frames, H, W, C]
            pro_pred: [B, Prediction_Frames, proprio_dim]
        """
        b, t_ctx, h, w, c = x_ctx.size()

        if t_ctx < self.context_frames:
            raise ValueError(
                f"context length {t_ctx} is shorter than required {self.context_frames}"
            )
        if t_ctx > self.context_frames:
            x_ctx = x_ctx[:, :self.context_frames, ...]

        cond = self._build_frame_condition(action, b, x_ctx.device, x_ctx.dtype)
        mod = self.modulation(cond)

        x_ctx = self.fc_in(x_ctx)
        x_ctx = einops.rearrange(x_ctx, 'b t h w c -> b t (h w) c')

        future_vis = self.future_vis_token.expand(b, self.prediction_frames, h * w, self.hidden_dim)

        has_proprio = False
        if self.proprio_dim and proprio is not None:
            if proprio.size(1) < self.context_frames:
                raise ValueError(
                    f"proprio length {proprio.size(1)} is shorter than context_frames {self.context_frames}"
                )
            p_ctx = self.proprio_encoder(proprio[:, :self.context_frames, :]).unsqueeze(2)
            p_future = self.future_proprio_token.expand(b, self.prediction_frames, 1, self.hidden_dim)
            x_ctx = torch.cat([p_ctx, x_ctx], dim=2)
            future_vis = torch.cat([p_future, future_vis], dim=2)
            has_proprio = True

        x = torch.cat([x_ctx, future_vis], dim=1)
        x = einops.rearrange(x, 'b t n c -> b (t n) c')

        x, _ = self.transformer(
            x,
            mod,
            full_shape=[b, self.total_frames, h, w, c],
            has_proprio=has_proprio
        )

        num_tokens = (h * w) + (1 if has_proprio else 0)
        x = einops.rearrange(x, 'b (t n) c -> b t n c', t=self.total_frames, n=num_tokens)

        future_indices = slice(self.context_frames, self.total_frames)
        x_future = x[:, future_indices, :, :]
        cond_future = cond[:, future_indices, :]

        proprio_pred = None
        if has_proprio:
            p_out = x_future[:, :, 0, :]
            vis_out = x_future[:, :, 1:, :]

            p_out_flat = einops.rearrange(p_out, 'b t c -> (b t) c')
            cond_flat = einops.rearrange(cond_future, 'b t d -> (b t) d')
            proprio_pred = self.proprio_out(p_out_flat, cond_flat)
            proprio_pred = einops.rearrange(
                proprio_pred, '(b t) c -> b t c', b=b, t=self.prediction_frames
            )
        else:
            vis_out = x_future

        vis_out_flat = einops.rearrange(vis_out, 'b t n c -> (b t n) c')
        cond_expanded_vis = einops.repeat(cond_future, 'b t d -> (b t n) d', n=h * w)
        vis_pred = self.fc_out(vis_out_flat, cond_expanded_vis)
        vis_pred = einops.rearrange(
            vis_pred,
            '(b t h w) c -> b t h w c',
            b=b,
            t=self.prediction_frames,
            h=h,
            w=w
        )

        return vis_pred, proprio_pred
