from typing import List, Optional

import einops
import numpy as np
import torch
import torch.nn as nn

from .convnext_isotropic_vae import ConvNeXtIsotropicVAE
from .dino_foresight_vae import DINOForesightVAE
from .vae_output import VAEOutput

# [Author's Hack] Prevent torch hub errors
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

class DINORawVAE(nn.Module):
    def __init__(
        self,
        name: str,
        dinov2_variant: str,
        patch_size: int,
        feature_stats: Optional[str] = None,
        **kwargs
    ):
        super(DINORawVAE, self).__init__()
        
        # [Modified] Load DINOv2 following author's style
        # We construct the name based on your config 'vits14' -> 'dinov2_vits14'
        full_dino_name = 'dinov2_' + dinov2_variant
        self.dino_v2 = torch.hub.load(
            'facebookresearch/dinov2',
            full_dino_name
        )
        
        # [Modified] Set feature key to match author's config
        # Author uses "x_norm_patchtokens" which corresponds to the last layer, normalized, no CLS
        self.feature_key = "x_norm_patchtokens" 

        self.dino_v2.eval()
        for param in self.dino_v2.parameters():
            param.requires_grad = False
            

        if feature_stats is not None:
            stats = np.load(feature_stats)
            self.feature_per_channel_mean = torch.from_numpy(stats['mean'])
            self.feature_per_channel_std = torch.from_numpy(stats['std'])
            self.epsilon = 1e-8
        else:
            self.feature_per_channel_mean = None
            self.feature_per_channel_std = None

        self.patch_size = patch_size

        if "convnext" in name:
            self.vae = ConvNeXtIsotropicVAE(**kwargs)

    def extract_features(self, x, reshape=False):
        # [Modified] Use forward_features instead of get_intermediate_layers
        # This matches the author's DinoV2Encoder.forward method
        with torch.no_grad():
            # The author's code: emb = self.base_model.forward_features(x)[self.feature_key]
            features = self.dino_v2.forward_features(x)
            x = features[self.feature_key]
            
        return x

    def feature_transform(self, x):
        mean = (
            self.feature_per_channel_mean
            .view((1, 1, -1))
            .to(x.device)
        )
        std = (
            self.feature_per_channel_std
            .view((1, 1, -1))
            .to(x.device)
        ) + self.epsilon
        return (x - mean) / std

    def feature_inverse_transform(self, x):
        mean = (
            self.feature_per_channel_mean
            .view((1, 1, -1))
            .to(x.device)
        )
        std = (
            self.feature_per_channel_std
            .view((1, 1, -1))
            .to(x.device)
        ) + self.epsilon
        x = x * std + mean
        return x

    def preprocess_features(self, f):
        # f: [b, t, h, w, c]
        B, T, H, W, *_ = f.shape
        f = einops.rearrange(
            f,
            "b t h w c -> (b t) (h w) c"
        )
        if self.feature_per_channel_mean is not None:
            f = self.feature_transform(f)
        f = einops.rearrange(
            f,
            '(b t) (h w) c -> b t h w c',
            b=B,
            t=T,
            h=H,
            w=W
        )
        return f

    def encode_features(self, f: torch.Tensor, deterministic=False):
        # f: [b, t, h, w, c]
        b, *_ = f.shape
        f = self.preprocess_features(f)
        f = einops.rearrange(f, "b t h w c -> (b t) h w c")
        z = self.vae(f, deterministic=deterministic).latents
        return einops.rearrange(z, '(b t) h w c -> b t h w c', b=b)

    def preprocess(self, x):
        B, T, C, H, W = x.shape
        # DINOv2 accepts 4 dimensions [B,C,H,W].
        # We use flatten at batch and time dim of x.
        x = x.flatten(end_dim=1)  # x.shape [B*T,C,H,W]
        
        # This calls our modified extract_features
        x = self.extract_features(x)  # [B*T,H*W,C]
        
        if self.feature_per_channel_mean is not None:
            x = self.feature_transform(x)
        
        # Rearrange using patch size logic
        x = einops.rearrange(
            x,
            'b (h w) c -> b h w c',
            h=H // self.patch_size,
            w=W // self.patch_size
        )
        x = x.unflatten(dim=0, sizes=(B, T))  # [B,T,H,W,C]
        return x

    def postprocess(self, x):
        if self.feature_per_channel_mean is not None:
            x = self.feature_inverse_transform(x)
        return x

    def forward(self, x: torch.Tensor, deterministic=False) -> VAEOutput:
        f = self.preprocess(x)
        f = einops.rearrange(f, "b t h w c -> (b t) h w c")
        return self.vae(f, deterministic=deterministic)