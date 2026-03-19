from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf

from vae.models.predictor.autoregressive_dino_foresight_latent import (
    AutoregressiveDINOForesightLatent,
)
from vae.models.predictor.joint_fm_transformer import JointMaskTransformerFM


class JointFMLatentPredictor(AutoregressiveDINOForesightLatent):
    """
    Joint FM predictor in the original DINO+VAE latent space.

    Reuses:
      - DINO feature preprocessing,
      - VAE encode/decode,
      - latent normalization stats,
    and swaps only the predictor backbone to a joint FM transformer.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        separable_attention: bool,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        vae: OmegaConf,
        action_dim: int,
        proprio_dim: int,
        use_proprio: bool = True,
        context_frames: int = 1,
        prediction_frames: int = 1,
        fm_variant: str = "fm",
        num_registers: int = 8,
        use_timestep: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            separable_attention=separable_attention,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            vae=vae,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            context_frames=context_frames,
            prediction_frames=prediction_frames,
            **kwargs,
        )

        self.is_regression_model = False
        self.transformer = JointMaskTransformerFM(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            depth=layers,
            heads=heads,
            mlp_dim=4 * hidden_dim,
            dropout=dropout,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            use_proprio=use_proprio,
            context_frames=self.context_frames,
            prediction_frames=self.prediction_frames,
            num_registers=num_registers,
            fm_variant=fm_variant,
            use_timestep=use_timestep,
        )

    @property
    def uses_timestep(self) -> bool:
        return bool(self.transformer.use_timestep)

    def forward(
        self,
        z_ctx: torch.Tensor,
        z_next_noisy: torch.Tensor,
        action_noisy: torch.Tensor,
        action_time: Optional[torch.Tensor] = None,
        next_state_time: Optional[torch.Tensor] = None,
        proprio_noisy: Optional[torch.Tensor] = None,
        proprio_ctx: Optional[torch.Tensor] = None,
        mask_action_from_visual: bool = False,
    ):
        """
        Args:
            z_ctx: [B, context_frames, h, w, c]
            z_next_noisy: [B, prediction_frames, h, w, c]
            action_noisy: [B, prediction_frames, action_dim]
            action_time: [B]
            next_state_time: [B]
            proprio_noisy: [B, prediction_frames, proprio_dim] (optional)
        Returns:
            action_pred, state_pred, proprio_pred
        """
        if self.uses_timestep:
            if action_time is None or next_state_time is None:
                raise ValueError(
                    "action_time and next_state_time are required in FM mode"
                )
        else:
            action_time = None
            next_state_time = None

        action_pred, state_pred, proprio_pred = self.transformer(
            z_next_noisy,
            noisy_action=action_noisy,
            action_time=action_time,
            next_state_time=next_state_time,
            noisy_proprio=proprio_noisy,
            context_state=z_ctx,
            context_proprio=proprio_ctx,
            mask_action_from_visual=mask_action_from_visual,
        )
        return action_pred, state_pred, proprio_pred

