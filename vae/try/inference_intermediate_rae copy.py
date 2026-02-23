import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
import random
import ssl

# ==========================================
# 0. SSL and Environment Setup
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
# 1. path setup
# ==========================================
current_file_path = os.path.abspath(__file__)
try_dir = os.path.dirname(current_file_path)
vae_root = os.path.dirname(try_dir)
jepa_root = os.path.dirname(vae_root)
rae_src = os.path.join(vae_root, "models/rae/src")

critical_paths = [vae_root, jepa_root, os.path.join(jepa_root, "data_processing")]
for p in critical_paths:
    if p not in sys.path: sys.path.insert(0, p)

# ==========================================
# 2. module imports
# ==========================================
try:
    from models.vae.dino_raw_vae import DINORawVAE
    from data_processing.utils import init_test_data 
    from data_processing.transforms import make_transforms
except ImportError as e:
    print(f"❌ VAE/Data Import Error: {e}")
    sys.exit(1)

sys.path.insert(0, rae_src)
try:
    from stage1 import RAE  
except ImportError as e:
    print(f"❌ RAE Import Error: {e}")
    sys.exit(1)
finally:
    if rae_src in sys.path: sys.path.remove(rae_src)

# ==========================================
# 3. helper functions
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

def plot_results(results_list, save_path="vae_rae_eval_kubric.png"):
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
        axes[i, 1].imshow(res['pca_orig'].reshape(H, W, 3)); axes[i, 1].set_title("PCA: GT Raw"); axes[i, 1].axis('off')
        axes[i, 2].imshow(res['pca_recon'].reshape(H, W, 3)); axes[i, 2].set_title("PCA: VAE Recon"); axes[i, 2].axis('off')
        
        axes[i, 3].imshow(np.clip(res['rae_dec_orig'].permute(1, 2, 0).cpu().float().numpy(), 0, 1))
        axes[i, 3].set_title("RAE (GT -> Img)"); axes[i, 3].axis('off')
        
        axes[i, 4].imshow(np.clip(res['rae_dec_recon'].permute(1, 2, 0).cpu().float().numpy(), 0, 1))
        axes[i, 4].set_title("RAE (VAE -> Img)"); axes[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Visualization saved to: {save_path}")

# ==========================================
# 4. Main
# ==========================================
@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # --- path ---
    vae_config_path = "/home/lizijia/Gen-JEPA/vae/models/configs/kubric/vae/dino_vae_prediction1.yaml"
    vae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/try/vae_checkpoints/kubric.pt"

    # vae_config_path = "/home/lizijia/Gen-JEPA/vae/models/configs/kubric/vae/convnext_isotropic_base.yaml"
    # vae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/try/vae_checkpoints/best_vae.pth"

    rae_config_path = "/home/lizijia/Gen-JEPA/vae/models/rae/configs/stage1/pretrained/DINOv2-B.yaml"
    rae_decoder_config_path = "/home/lizijia/Gen-JEPA/vae/models/rae/configs/decoder/ViTXL/config.json"
    rae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/models/rae/models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    
    test_data_path = "/home/lizijia/Gen-JEPA/vae/data/pusht_noise"

    # ==========================================
    # 1. Loading VAE model & Config
    # ==========================================
    print(f"Loading VAE Config...")
    config = OmegaConf.load(vae_config_path)
    model_params = config.model if 'model' in config else config
    rae_stats_path = model_params.get('rae_stats_path', None)

    vae_model = DINORawVAE(**model_params).to(device).eval()

    # ==========================================
    # 2. Loading VAE Checkpoint
    # ==========================================
    print(f"Loading VAE Checkpoint: {vae_ckpt_path}")
    full_checkpoint = torch.load(vae_ckpt_path, map_location=device)
    
    if 'model_state_dict' in full_checkpoint:
        state_dict = full_checkpoint['model_state_dict']
    else:
        state_dict = full_checkpoint

    new_state_dict = {k.replace("vae.", ""): v for k, v in state_dict.items() if k.startswith("vae.") or "vae" not in k}
    missing, unexpected = vae_model.vae.load_state_dict(new_state_dict, strict=False)
    
    if len([k for k in missing if "encoder" in k and "backbone" not in k]) == 0:
        print("✅ VAE Weights loaded successfully.")
    else:
        print(f"⚠️ Warning: Possible missing VAE layers.")

    # ==========================================
    # 3. Loading RAE
    # ==========================================
    print("Loading RAE from dedicated config...")
    rae_yaml_cfg = OmegaConf.load(rae_config_path)
    rae_params = OmegaConf.to_container(rae_yaml_cfg.stage_1.params, resolve=True)
    
    rae_params['decoder_config_path'] = rae_decoder_config_path
    if rae_stats_path:
        rae_params['normalization_stat_path'] = rae_stats_path 
    
    # initialize RAE
    rae_model = RAE(**rae_params).to(device).eval()

    if os.path.exists(rae_ckpt_path):
        rae_st = torch.load(rae_ckpt_path, map_location=device)
        if 'state_dict' in rae_st: rae_st = rae_st['state_dict']
        new_st = {k.replace("decoder.", ""): v for k, v in rae_st.items() if k.startswith("decoder.")}
        rae_model.decoder.load_state_dict(new_st, strict=False)
        print("✅ RAE Decoder weights loaded via YAML config.")
    else:
        print(f"❌ RAE Checkpoint not found.")
        sys.exit(1)

    # ==========================================
    # 4. lcoading data
    # ==========================================
    transform = make_transforms(img_size=224)
    test_dataset, test_traj, test_loader = init_test_data(
        test_data_path=test_data_path,
        batch_size=1, 
        num_workers=4,
        transform=transform
    )

    # ==========================================
    # 5. inference
    # ==========================================
    print("Running Inference...")
    results = []
    
    all_indices = list(range(len(test_loader)))
    selected_indices = random.sample(all_indices, min(2, len(all_indices)))
    
    for idx in selected_indices:
        batch = test_loader.dataset[idx]
        obs = batch[0]
        img = obs['visual'] 
        
        if len(img.shape) == 3: img = img.unsqueeze(0)
        if len(img.shape) == 4 and img.shape[0] > 1: img = img[0:1]
        if img.ndim == 4: img = img.unsqueeze(0)
        img = img.to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # A. VAE Forward
            target_feat_vae_norm = vae_model.preprocess(img).squeeze(1) 
            vae_out = vae_model(img)
            recon_feat_vae_norm = vae_out.sample.squeeze(1) 

            # B. convert to RAE space 
            # (这部分帮你完成了从 VAE stats 切换到 RAE stats 的对齐)
            gt_rae_input = vae_model.convert_to_rae_space(target_feat_vae_norm, target_dim=768)
            vae_rae_input = vae_model.convert_to_rae_space(recon_feat_vae_norm, target_dim=768)

            # C. RAE decode
            # 💡 直接调用官方 decode。内部逻辑: 
            #    (1) RAE反归一化 (2) Transformer解码 (3) 图片RGB反归一化
            dec_gt = rae_model.decode(gt_rae_input.permute(0, 3, 1, 2))
            dec_vae = rae_model.decode(vae_rae_input.permute(0, 3, 1, 2))

        # D. For PCA visualization, use raw dino features
        gt_raw = vae_model.postprocess(target_feat_vae_norm)[0, ..., -768:]
        vae_raw = vae_model.postprocess(recon_feat_vae_norm)[0, ..., -768:]
        
        f_gt = gt_raw.reshape(-1, 768).float().cpu().numpy()
        f_vae = vae_raw.reshape(-1, 768).float().cpu().numpy()
        v_gt, v_vae = perform_pca_vis(f_gt, f_vae)

        results.append({
            'orig_img': img[0, 0].float().cpu(),
            'pca_orig': v_gt,
            'pca_recon': v_vae,
            'rae_dec_orig': dec_gt[0].float().cpu(),
            'rae_dec_recon': dec_vae[0].float().cpu()
        })
        print(f"  - Sample {idx} Processed")

    plot_results(results, save_path="vae_rae_eval_kubric.png")

if __name__ == "__main__":
    main()