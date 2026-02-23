import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import os
import sys
from contextlib import nullcontext

current_file_path = os.path.abspath(__file__)
try_dir = os.path.dirname(current_file_path)
vae_root = os.path.dirname(try_dir)
jepa_root = os.path.dirname(vae_root)
rae_src = os.path.join(vae_root, "models/rae/src")

critical_paths = [vae_root, jepa_root, os.path.join(jepa_root, "data_processing")]
for p in critical_paths:
    if p not in sys.path: sys.path.insert(0, p)

from data_processing.transforms import make_transforms
from models.vae.dino_raw_vae import DINORawVAE

device = "cuda" if torch.cuda.is_available() else "cpu"
img_path = "/home/lizijia/Gen-JEPA/vae/try/n01580077_1018.JPEG"
cfg_path = "/home/lizijia/Gen-JEPA/vae/models/configs/kubric/vae/convnext_isotropic_base.yaml"
out_path = "/home/lizijia/Gen-JEPA/vae/try/n01580077_dino_stats.npz"

cfg = OmegaConf.load(cfg_path)
model = DINORawVAE(**cfg.model).to(device).eval()

img = np.array(Image.open(img_path).convert("RGB"))[None, ...]  # [T=1,H,W,C]
tfm = make_transforms(img_size=224, random_horizontal_flip=False, random_resize_scale=(1.0,1.0), hwc=True, do_255_to_1=True)
x = tfm(img).unsqueeze(0).to(device)  # [B=1,T=1,C,H,W]

amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
with amp_ctx:
    with torch.no_grad():
        f = model.extract_features(x.flatten(end_dim=1), reshape=False)  # [BT,N,C]
        f = f.reshape(-1, f.shape[-1]).float().cpu().numpy()             # [N,C]

mean = f.mean(axis=0)
std = f.std(axis=0) + 1e-6
np.savez(out_path, mean=mean, std=std)
print("saved:", out_path, mean.shape, std.shape)
