import os
from typing import List, Optional, Union

import einops
import numpy as np
import torch
import torch.nn as nn

from .convnext_isotropic_vae import ConvNeXtIsotropicVAE
from .vae_output import VAEOutput

class DINORawVAE(nn.Module):
    def __init__(
        self,
        name: str,
        dinov2_variant: str,
        intermediate_layers: Optional[List[int]] = None,
        patch_size: int = 14,
        feature_stats: Optional[str] = None,   # 数据集统计量 (Dataset Stats)
        rae_stats_path: Optional[str] = None,  # RAE/ImageNet统计量 (RAE Stats)
        use_rae_extraction: bool = False,
        **kwargs
    ):
        super(DINORawVAE, self).__init__()

        # === 1. 模型加载 ===
        model_name = dinov2_variant
        if "reg" in dinov2_variant or "registers" in dinov2_variant:
            if "base" in dinov2_variant or "vitb14" in dinov2_variant:
                model_name = "dinov2_vitb14_reg"
            elif "large" in dinov2_variant or "vitl14" in dinov2_variant:
                model_name = "dinov2_vitl14_reg"

        print(f"[DINORawVAE] Loading DINOv2 model: {model_name}")
        self.dino_v2 = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.dino_v2.eval()
        for param in self.dino_v2.parameters():
            param.requires_grad = False

        self.d_layers = intermediate_layers
        self.d_num_layers = len(self.d_layers) if self.d_layers else 1
        self.use_rae_extraction = use_rae_extraction
        self.patch_size = patch_size
        self.epsilon = 1e-6

        # ==========================================
        # 2. 加载 Dataset Stats
        # ==========================================
        self.data_mean = None
        self.data_std = None
        
        if feature_stats is not None and os.path.exists(feature_stats):
            print(f"[DINORawVAE] Loading DATASET stats from: {feature_stats}")
            if feature_stats.endswith('.npz'):
                stats = np.load(feature_stats)
                self.data_mean = torch.from_numpy(stats['mean']).float()
                self.data_std = torch.from_numpy(stats['std']).float()
            elif feature_stats.endswith('.pt'):
                stats = torch.load(feature_stats, map_location="cpu")
                self.data_mean = stats.get('mean', None)
                self.data_std = stats.get('std', None) or torch.sqrt(stats.get('var'))
            
            if self.data_mean is not None:
                self.data_mean = self.data_mean.detach()
                self.data_std = self.data_std.detach()

        # ==========================================
        # 3. 加载 RAE Stats (保持原始加载逻辑，不压缩)
        # ==========================================
        self.rae_mean = None
        self.rae_std = None

        if rae_stats_path is not None and os.path.exists(rae_stats_path):
            print(f"[DINORawVAE] Loading RAE (ImageNet) stats from: {rae_stats_path}")
            stats = torch.load(rae_stats_path, map_location="cpu")
            
            if 'mean' in stats and stats['mean'] is not None:
                self.rae_mean = stats['mean'].float().detach()
            else:
                print("[DINORawVAE] RAE mean is None, using zeros.")
                self.rae_mean = torch.zeros(768) 

            if 'var' in stats:
                self.rae_std = torch.sqrt(stats['var']).float().detach()
            elif 'std' in stats:
                self.rae_std = stats['std'].float().detach()

        if "convnext" in name:
            self.vae = ConvNeXtIsotropicVAE(**kwargs)

    def extract_features(self, x, reshape=False):
        with torch.no_grad():
            if self.use_rae_extraction:
                ret = self.dino_v2.forward_features(x)
                x = ret["x_norm_patchtokens"]
                if reshape:
                    B, N, C = x.shape
                    H = W = int(N ** 0.5)
                    x = x.view(B, H, W, C).permute(0, 3, 1, 2)
            else:
                x = self.dino_v2.get_intermediate_layers(x, n=self.d_layers, reshape=reshape)
                if self.d_num_layers > 1:
                    x = torch.cat(x, dim=-1)
                else:
                    x = x[0]
        return x

    def feature_transform(self, x):
        if self.data_mean is None: return x
        view_shape = [1] * (x.ndim - 1) + [-1]
        mean = self.data_mean.to(x.device).view(*view_shape)
        std = self.data_std.to(x.device).view(*view_shape) + self.epsilon
        return (x - mean) / std

    def feature_inverse_transform(self, x):
        if self.data_mean is None: return x
        view_shape = [1] * (x.ndim - 1) + [-1]
        mean = self.data_mean.to(x.device).view(*view_shape)
        std = self.data_std.to(x.device).view(*view_shape) + self.epsilon
        return x * std + mean

    def convert_to_rae_space(self, vae_output, target_dim=768):
        # 1. 还原到 Raw 空间
        x_raw = self.feature_inverse_transform(vae_output)
        # x_raw = vae_output
        x_raw_sub = x_raw[..., -target_dim:] 
        
        if self.rae_mean is None or self.rae_std is None:
            return x_raw_sub
            
        device = vae_output.device
        
        # 2. 直接写死转换逻辑
        if self.rae_mean.numel() == 196608:
            mean = self.rae_mean.to(device).view(1, 16, 16, 768)
        else:
            mean = self.rae_mean.to(device).view(1, 1, 1, target_dim)
            
        if self.rae_std.numel() == 196608:
            std = self.rae_std.to(device).view(1, 16, 16, 768)
        else:
            std = self.rae_std.to(device).view(1, 1, 1, target_dim)
            
        # 3. 运算
        x_rae = (x_raw_sub - mean) / (std + self.epsilon)
        x_rae = x_raw_sub
        return x_rae

    def preprocess(self, x):
        B, T, C, H, W = x.shape
        x = x.flatten(end_dim=1)
        x = self.extract_features(x, reshape=False)
        if self.data_mean is not None:
            x = self.feature_transform(x)
        n_patches = x.shape[1]
        grid_size = int(n_patches ** 0.5)
        x = einops.rearrange(x, 'b (h w) c -> b h w c', h=grid_size, w=grid_size)
        x = x.unflatten(dim=0, sizes=(B, T))
        return x

    def postprocess(self, x):
        if self.data_mean is not None:
            x = self.feature_inverse_transform(x)
        return x

    def forward(self, x: torch.Tensor, deterministic=False) -> VAEOutput:
        f = self.preprocess(x)
        f = einops.rearrange(f, "b t h w c -> (b t) h w c")
        return self.vae(f, deterministic=deterministic)