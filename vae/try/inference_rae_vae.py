import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
import ssl

# ==========================================
# 0. SSL 与环境配置
# ==========================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for env_key in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]:
    if env_key in os.environ: del os.environ[env_key]

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ==========================================
# 1. 路径配置
# ==========================================
VAE_BASE = "/home/lizijia/Gen-JEPA/vae"
DATA_PROC = "/home/lizijia/Gen-JEPA/data_processing"
RAE_SRC = "/home/lizijia/Gen-JEPA/vae/models/rae/src"
JEPA_ROOT = "/home/lizijia/Gen-JEPA"

sys.path = [p for p in sys.path if p not in [VAE_BASE, DATA_PROC, RAE_SRC, JEPA_ROOT]]
sys.path.insert(0, VAE_BASE)
sys.path.insert(1, DATA_PROC)
sys.path.insert(2, JEPA_ROOT)

# ==========================================
# 2. 模块导入
# ==========================================
try:
    from models.vae.dino_raw_vae import DINORawVAE
    from data_processing.utils import init_data
    from data_processing.transforms import make_transforms
except ImportError as e:
    print(f"❌ VAE/Data Import Error: {e}")
    sys.exit(1)

sys.path.insert(0, RAE_SRC)
try:
    from stage1 import RAE  
except ImportError as e:
    print(f"❌ RAE Import Error: {e}")
    sys.exit(1)
finally:
    if RAE_SRC in sys.path: sys.path.remove(RAE_SRC)

# ==========================================
# 3. 自定义解码函数 (Fixed)
# ==========================================
def custom_rae_decode(model, z):
    # 1. 形状调整 [B, C, H, W] -> [B, N, C]
    if model.reshape_to_2d:
        b, c, h, w = z.shape
        n = h * w
        z = z.view(b, c, n).transpose(1, 2) 

    # 2. 前向传播 (RAE Decoder 会自动加上 CLS token)
    output = model.decoder(z, drop_cls_token=False).logits
    
    # 3. 切除 CLS Token
    if output.shape[1] == model.decoder.num_patches + 1:
        output = output[:, 1:, :] 

    # 4. Unpatchify (Patch -> Image)
    x_rec = model.decoder.unpatchify(output)
    
    # 5. 图像空间反归一化 (ImageNet Mean/Std -> Pixel [0,1])
    x_rec = x_rec * model.encoder_std.to(x_rec.device) + model.encoder_mean.to(x_rec.device)
    
    return x_rec

# ==========================================
# 4. 可视化工具
# ==========================================
def perform_pca_vis(feat1, feat2=None, n_components=3):
    if feat2 is not None:
        combined = np.concatenate([feat1, feat2], axis=0)
        pca = PCA(n_components=n_components)
        pca.fit(combined)
        vis1 = pca.transform(feat1)
        vis2 = pca.transform(feat2)
        all_vals = np.concatenate([vis1, vis2])
        v_min, v_max = all_vals.min(axis=0), all_vals.max(axis=0)
        denom = v_max - v_min + 1e-8
        return (vis1 - v_min) / denom, (vis2 - v_min) / denom
    else:
        pca = PCA(n_components=n_components)
        vis = pca.transform(feat1)
        return (vis - vis.min(0)) / (vis.max(0) - vis.min(0) + 1e-8)

def plot_results(results_list, save_path="vae_rae_eval.png"):
    n = len(results_list)
    if n == 0: return
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1: axes = axes[None, :]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, res in enumerate(results_list):
        img = res['orig_img'].permute(1, 2, 0).cpu().float().numpy()
        img = np.clip(img * std + mean, 0, 1)
        axes[i, 0].imshow(img); axes[i, 0].set_title("Original Input"); axes[i, 0].axis('off')
        
        H = W = int(res['pca_orig'].shape[0]**0.5)
        axes[i, 1].imshow(res['pca_orig'].reshape(H, W, 3)); axes[i, 1].set_title("PCA: GT Features"); axes[i, 1].axis('off')
        axes[i, 2].imshow(res['pca_recon'].reshape(H, W, 3)); axes[i, 2].set_title("PCA: VAE Features"); axes[i, 2].axis('off')
        
        axes[i, 3].imshow(np.clip(res['rae_dec_orig'].permute(1, 2, 0).cpu().float().numpy(), 0, 1))
        axes[i, 3].set_title("RAE Decode (GT -> Image)"); axes[i, 3].axis('off')
        
        axes[i, 4].imshow(np.clip(res['rae_dec_recon'].permute(1, 2, 0).cpu().float().numpy(), 0, 1))
        axes[i, 4].set_title("RAE Decode (VAE -> Image)"); axes[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Visualization saved to: {save_path}")

# ==========================================
# 5. 主程序
# ==========================================
@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # --- 路径 ---
    vae_config_path = "/home/lizijia/Gen-JEPA/vae/models/configs/kubric/vae/rae_vae_base.yaml"
    vae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/models/vae/checkpoints/rae_vae/best_vae.pth" 
    
    rae_decoder_config_path = "/home/lizijia/Gen-JEPA/vae/models/rae/configs/decoder/ViTXL/config.json"
    rae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/models/rae/models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    
    dataset_stats_path = "/home/lizijia/Gen-JEPA/vae/data/push_t_dino_b_rae_3000batch.npz"
    rae_stats_path = "/home/lizijia/Gen-JEPA/vae/models/rae/stats/stat.pt"
    test_data_path = "/home/lizijia/Gen-JEPA/vae/data/pusht_noise"

    # --- 1. Load Stats ---
    print(f"Loading Stats...")
    try:
        raw_stats = torch.load(rae_stats_path, map_location='cpu')
    except:
        raw_stats = {}

    def parse_stat(val, default, target_dim=768):
        if val is None: 
            return torch.full((target_dim,), float(default), device=device)
        t = val.to(device).float()
        if t.numel() == 1: return torch.full((target_dim,), t.item(), device=device)
        if t.numel() > target_dim: t = t.view(-1, target_dim).mean(dim=0)
        return t

    rae_mean_1d = parse_stat(raw_stats.get('mean'), 0.0)
    if raw_stats.get('var') is not None:
        rae_var_1d = parse_stat(raw_stats.get('var'), 1.0)
    elif raw_stats.get('std') is not None:
        rae_var_1d = parse_stat(raw_stats.get('std'), 1.0) ** 2
    else:
        rae_var_1d = parse_stat(None, 1.0)
    
    rae_std_1d = torch.sqrt(rae_var_1d + 1e-5)

    # --- 2. Load VAE ---
    print("Loading VAE...")
    vae_conf = OmegaConf.load(vae_config_path)
    model_args = OmegaConf.to_container(vae_conf.model, resolve=True)
    for k in ['feature_stats', 'rae_stats_path']: 
        if k in model_args: del model_args[k]

    vae_model = DINORawVAE(**model_args, feature_stats=dataset_stats_path, rae_stats_path=None).to(device).eval()
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_model.vae.load_state_dict(vae_ckpt['state_dict'] if 'state_dict' in vae_ckpt else vae_ckpt, strict=False)

    # --- 3. Init RAE ---
    print("Loading RAE...")
    rae_model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-base',
        encoder_input_size=224,
        encoder_params={'dinov2_path': 'facebook/dinov2-base'}, 
        decoder_config_path=rae_decoder_config_path, 
        decoder_patch_size=16,
        reshape_to_2d=True, 
        normalization_stat_path=None,
        pretrained_decoder_path=None 
    ).to(device).eval()

    # --- 4. Load RAE Decoder Weights (Silent) ---
    if os.path.exists(rae_ckpt_path):
        rae_state = torch.load(rae_ckpt_path, map_location=device)
        if 'state_dict' in rae_state: 
            rae_state = rae_state['state_dict']
        
        # 修正 key: 移除 "decoder." 前缀，因为我们要直接 load 进 rae_model.decoder
        new_state_dict = {}
        for k, v in rae_state.items():
            if k.startswith("decoder."):
                new_state_dict[k.replace("decoder.", "")] = v
            else:
                new_state_dict[k] = v
        
        rae_model.decoder.load_state_dict(new_state_dict, strict=False)
        print("✅ RAE Decoder weights loaded.")
    else:
        print(f"❌ Error: Checkpoint not found: {rae_ckpt_path}")
        sys.exit(1)

    # --- 5. Data & Inference ---
    print("Loading Data...")
    transform = make_transforms(img_size=224)
    ret = init_data(
        data_paths=[test_data_path], batch_size=1, dataset_type="custom",
        num_workers=0, transform=transform,
        frameskip=5, num_hist=3, num_pred=1, rank=0, world_size=1
    )
    val_loader = ret[5]

    print("Running Inference...")
    results = []
    
    for i, batch in enumerate(val_loader):
        if i >= 5: break
        
        img = batch[0]['visual'].to(device)
        if len(img.shape) == 5: img = img[:, 0]
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # A. Get Raw Features
            gt_feat_raw = vae_model.preprocess(img.unsqueeze(1)).squeeze(1)
            vae_out_raw = vae_model.vae(gt_feat_raw, deterministic=True).sample
            
            # B. Normalize Manually
            gt_feat_norm = (gt_feat_raw - rae_mean_1d) / rae_std_1d
            vae_feat_norm = (vae_out_raw - rae_mean_1d) / rae_std_1d
            
            # C. Prepare Input
            gt_input = gt_feat_norm.permute(0, 3, 1, 2)
            vae_input = vae_feat_norm.permute(0, 3, 1, 2)
            
            # D. Decode
            dec_gt = custom_rae_decode(rae_model, gt_input)
            dec_vae = custom_rae_decode(rae_model, vae_input)
        
        f_gt = gt_feat_raw[0].reshape(-1, 768).float().cpu().numpy()
        f_vae = vae_out_raw[0].reshape(-1, 768).float().cpu().numpy()
        v_gt, v_vae = perform_pca_vis(f_gt, f_vae)
        
        results.append({
            'orig_img': img[0].float().cpu(),
            'pca_orig': v_gt, 'pca_recon': v_vae,
            'rae_dec_orig': dec_gt[0].float().cpu(), 
            'rae_dec_recon': dec_vae[0].float().cpu()
        })
        print(f"  - Sample {i+1} Processed")

    plot_results(results, save_path="vae_rae_eval.png")

if __name__ == "__main__":
    main()