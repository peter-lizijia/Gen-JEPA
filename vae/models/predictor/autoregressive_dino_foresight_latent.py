from typing import Tuple, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from vae.models.predictor.dino_foresight_dit import (
    MaskTransformer,
    MaskTransformerRegression,
)
from vae.models.vae.dino_raw_vae import DINORawVAE


class AutoregressiveDINOForesightLatent(nn.Module):
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
        context_frames: int = 3,    
        prediction_frames: int = 6, 
        **kwargs 
    ):
        super(AutoregressiveDINOForesightLatent, self).__init__()
        
        self.patch_size = patch_size
        self.is_regression_model = False
        self.context_frames = context_frames
        self.prediction_frames = prediction_frames
        self.total_frames = self.context_frames + self.prediction_frames
        
        assert self.prediction_frames > 0, f"Prediction frames must be > 0. Got {prediction_frames}."
        
        # calculate Latent grid size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        
        # Transformer shape (keep the portal)
        shape = (self.total_frames, self.grid_size[0], self.grid_size[1])

        # ==========================
        # 1. Load VAE and initialize
        # ==========================
        # initialize DINORawVAE Wrapper
        self.vae = DINORawVAE(**vae.model)  # type: ignore
        
        # loading weights
        print(f"Loading VAE from {vae.ckpt_path}...")
        ckpt = torch.load(vae.ckpt_path, map_location="cpu") # weights_only=True depends on Pytorch version
        
        # loading weights with flexible key handling
        # -----------------------------------------------------------
        # Fix: Robust loading logic to handle Key mismatch (vae. prefix)
        # -----------------------------------------------------------
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Detect prefix mismatch
        model_keys = list(self.vae.state_dict().keys())
        ckpt_keys = list(state_dict.keys())
        
        # Check if model expects 'vae.' prefix but ckpt doesn't have it, or vice versa
        model_has_prefix = any(k.startswith("vae.") for k in model_keys)
        ckpt_has_prefix = any(k.startswith("vae.") for k in ckpt_keys)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if model_has_prefix and not ckpt_has_prefix:
                new_key = f"vae.{k}" # Add prefix
            elif not model_has_prefix and ckpt_has_prefix:
                new_key = k.replace("vae.", "") # Remove prefix
            new_state_dict[new_key] = v
            
        # Load with strict=False to ignore missing DINO weights
        missing, unexpected = self.vae.load_state_dict(new_state_dict, strict=False)
        
        # Critical Check: Verify if Encoder/Decoder weights are actually loaded
        encoder_missing = [k for k in missing if "encoder" in k and "dino" not in k]
        decoder_missing = [k for k in missing if "decoder" in k]
        
        if len(encoder_missing) > 0 or len(decoder_missing) > 0:
            # Raise error to stop training immediately if VAE is not loaded correctly
            raise RuntimeError(
                f"❌ CRITICAL ERROR: VAE weights failed to load!\n"
                f"   Missing Encoder keys: {len(encoder_missing)}\n"
                f"   Missing Decoder keys: {len(decoder_missing)}\n"
                f"   Please check the checkpoint key names."
            )
        else:
            print("✅ VAE weights loaded successfully (Encoder & Decoder matched).")
            # Optional: Print info about ignored keys (likely DINO)
            # print(f"   Note: {len(missing)} keys missing (expected for DINO/Heads).")
        # -----------------------------------------------------------
            
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # ==========================
        # 2. load Latent stats for normalization
        # ==========================
        stats = np.load(vae.latent_dist_stats_path)
        self.register_buffer("latent_dist_mean", torch.from_numpy(stats["mean"]).float())
        self.register_buffer("latent_dist_std", torch.from_numpy(stats["std"]).float())
        self.embedding_dim = self.latent_dist_mean.shape[0]  # type: ignore

        # ==========================
        # 3. Transformer (MaskTransformer / DiT)
        # ==========================
        self.transformer = MaskTransformer(
            shape=shape,
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            depth=layers,
            heads=heads,
            mlp_dim=4 * hidden_dim,
            dropout=dropout,
            action_dim=action_dim,   
            proprio_dim=proprio_dim,
            context_frames=self.context_frames,       
            prediction_frames=self.prediction_frames  
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """
        Use the VAE to encode pixels into normalized latents.
        
        Input: 
            x: [B, T, C, H, W] (Pixels)
        Output: 
            z: [B, T, h, w, c] (Normalized Latents, Channel Last)
        """
        b, t = x.shape[:2]
        
        output = self.vae(x) 
        z = output.latents  # Shape [(B*T), h, w, c]

        z = einops.rearrange(z, "(b t) h w c -> b t h w c", b=b, t=t)
        
        # 3. Normalize
        # mean/std shape: [1, 1, 1, 1, C]
        mean = self.latent_dist_mean.view(1, 1, 1, 1, -1)
        std = self.latent_dist_std.view(1, 1, 1, 1, -1)
        return (z - mean) / std

    @torch.no_grad()
    def decode(self, z: torch.Tensor):
        """
        Decode latents to DINO features
        
        Input: 
            z: [B, T, h, w, c] (Normalized Latents)
        Output: 
            f: [B, T, c, h, w] (Reconstructed Features) or [B, T, C, H, W] (Pixels)
        """
        b, t = z.shape[:2]
        
        # 1. Denormalize
        mean = self.latent_dist_mean.view(1, 1, 1, 1, -1)
        std = self.latent_dist_std.view(1, 1, 1, 1, -1)
        z = z * std + mean
        
        # 2. VAE Decoder
        z_flat = einops.rearrange(z, "b t h w c -> (b t) h w c")
        
        # Added safety check for nested structure
        if hasattr(self.vae, "vae"):
             f = self.vae.vae.decode(z_flat)
        else:
             f = self.vae.decode(z_flat)
        
        # 3. Reshape back
        if f.ndim == 4:
            f = einops.rearrange(f, "(b t) h w c -> b t h w c", b=b, t=t)
                
        return f

    def forward(
        self, 
        z_ctx: torch.Tensor, 
        z_t: torch.Tensor, 
        t: torch.Tensor, 
        action: Optional[torch.Tensor] = None, 
        proprio: Optional[torch.Tensor] = None
    ):
        """
        
        Args:
            z_ctx: [B, context_frames, h, w, c]
            z_t:   [B, prediction_frames, h, w, c]
            t:     [B] (Time steps)
            action: [B, total_frames, action_dim]
            proprio:[B, total_frames, proprio_dim]
        """
        
        # 1. concat Context 和 Noisy Target
        # z_input: [B, total_frames, h, w, c]
        z_input = torch.cat([z_ctx, z_t], dim=1)  
        
        # 2.  MaskTransformer
        # MaskTransformer needs action to be [B, T, D]
        v_vis, v_prop = self.transformer(
            z_input, 
            time=t, 
            action=action, 
            proprio=proprio
        )
        
        return v_vis, v_prop


class AutoregressiveDINOForesightLatentRegressor(AutoregressiveDINOForesightLatent):
    """
    L2 regression predictor:
    - input: context latents only
    - output: future latents in one forward pass
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
        context_frames: int = 3,
        prediction_frames: int = 6,
        **kwargs
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
            **kwargs
        )
        self.is_regression_model = True

        shape = (self.total_frames, self.grid_size[0], self.grid_size[1])
        self.transformer = MaskTransformerRegression(
            shape=shape,
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            depth=layers,
            heads=heads,
            mlp_dim=4 * hidden_dim,
            dropout=dropout,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            context_frames=self.context_frames,
            prediction_frames=self.prediction_frames
        )

    def forward(
        self,
        z_ctx: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None
    ):
        """
        Args:
            z_ctx: [B, context_frames, h, w, c]
            action: [B, total_frames, action_dim]
            proprio:[B, total_frames, proprio_dim] or [B, context_frames, proprio_dim]
        """
        pred_vis, pred_prop = self.transformer(
            z_ctx,
            action=action,
            proprio=proprio
        )
        return pred_vis, pred_prop
