import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
from omegaconf import OmegaConf
import ssl
from PIL import Image
from contextlib import nullcontext

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

def print_tensor_stats(name, tensor):
    t = tensor.detach().float()
    print(
        f"[debug] {name}: mean={t.mean().item():.4f}, "
        f"std={t.std().item():.4f}, min={t.min().item():.4f}, max={t.max().item():.4f}"
    )

def load_vae_state_dict_compatible(vae_model, full_checkpoint):
    target_state = vae_model.vae.state_dict()
    target_keys = set(target_state.keys())

    # Candidate containers in priority order.
    state_candidates = []
    if isinstance(full_checkpoint, dict):
        for key in ("ema_state_dict", "model_state_dict", "state_dict"):
            maybe_sd = full_checkpoint.get(key, None)
            if isinstance(maybe_sd, dict):
                state_candidates.append((key, maybe_sd))
    if isinstance(full_checkpoint, dict) and any(torch.is_tensor(v) for v in full_checkpoint.values()):
        state_candidates.append(("raw", full_checkpoint))
    elif not isinstance(full_checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(full_checkpoint)}")

    def transformed_maps(sd):
        yield ("as_is", sd)
        prefixes = ("vae.", "module.vae.", "model.vae.", "module.", "model.")
        for prefix in prefixes:
            stripped = {k[len(prefix):]: v for k, v in sd.items() if isinstance(k, str) and k.startswith(prefix)}
            if stripped:
                yield (f"strip:{prefix}", stripped)

    best = None
    for src_name, sd in state_candidates:
        for tf_name, mapped in transformed_maps(sd):
            filtered = {}
            shape_mismatch = 0
            for k, v in mapped.items():
                if k not in target_keys:
                    continue
                if not torch.is_tensor(v):
                    continue
                if tuple(v.shape) != tuple(target_state[k].shape):
                    shape_mismatch += 1
                    continue
                filtered[k] = v
            matched = len(filtered)
            if best is None or matched > best["matched"]:
                best = {
                    "source": src_name,
                    "transform": tf_name,
                    "state_dict": filtered,
                    "matched": matched,
                    "shape_mismatch": shape_mismatch,
                }

    if best is None or best["matched"] == 0:
        raise RuntimeError("No compatible VAE weights found in checkpoint.")

    print(
        f"[debug] VAE weight source={best['source']} "
        f"transform={best['transform']} matched={best['matched']}/{len(target_keys)} "
        f"shape_mismatch={best['shape_mismatch']}"
    )
    missing, unexpected = vae_model.vae.load_state_dict(best["state_dict"], strict=False)
    if best["matched"] != len(target_keys):
        raise RuntimeError(
            f"Incomplete VAE load: matched={best['matched']}/{len(target_keys)} "
            f"(source={best['source']}, transform={best['transform']})"
        )
    return missing, unexpected

def plot_results(results_list, save_path="vae_rae_eval_imagenet_single.png"):
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

    # vae_config_path = "/home/lizijia/Gen-JEPA/vae/models/configs/cityscapes/vae/raw/convnext_isotropic_large.yaml"
    # vae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/try/vae_checkpoints/cityscape.pt"
    
    rae_config_path = "/home/lizijia/Gen-JEPA/vae/models/rae/configs/stage1/pretrained/DINOv2-B.yaml"
    rae_decoder_config_path = "/home/lizijia/Gen-JEPA/vae/models/rae/configs/decoder/ViTXL/config.json"
    rae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/models/rae/models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"

    # ==========================================
    # 1. Loading VAE model & Config
    # ==========================================
    print(f"Loading VAE Config...")
    config = OmegaConf.load(vae_config_path)
    model_params = config.model if 'model' in config else config
    rae_stats_path = model_params.get('rae_stats_path', None)

    cfg_ckpt_path = config.get("ckpt_path", None)
    if (vae_ckpt_path is None) and cfg_ckpt_path is not None:
        vae_ckpt_path = cfg_ckpt_path

    vae_model = DINORawVAE(**model_params).to(device).eval()

    # ==========================================
    # 2. Loading VAE Checkpoint
    # ==========================================
    print(f"Loading VAE Checkpoint: {vae_ckpt_path}")
    full_checkpoint = torch.load(vae_ckpt_path, map_location=device)
    missing, unexpected = load_vae_state_dict_compatible(vae_model, full_checkpoint)
    
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
    # 4. Preparing ImageNet Single Image Data
    # ==========================================
    test_image_path = "/home/lizijia/Gen-JEPA/vae/try/n01580077_1018.JPEG"
    
    # changing format for VideoTransform
    raw_pil = Image.open(test_image_path).convert("RGB")
    img_np = np.array(raw_pil)        # to [H, W, 3] numpy array
    img_input = img_np[np.newaxis, ...] # to [T=1, H, W, C=3]

    print(f"Input Numpy Shape before transform: {img_input.shape}")

    transform = make_transforms(
        img_size=224,
        random_horizontal_flip=False,    # no random flip
        random_resize_scale=(1.0, 1.0),  # no random resize
        hwc=True,                        # the input is [T, H, W, C]
        do_255_to_1=True                 # the input is in range 0-255, need to be normalized to 0-1
    )

    img_tensor = transform(img_input)    # [T=1, C=3, H=224, W=224]
    
    # add batch dimension to [B=1, T=1, C=3, H=224, W=224]
    img = img_tensor.unsqueeze(0).to(device)

    # ==========================================
    # 5. inference
    # ==========================================
    print("Running Inference...")
    results = []
    remove_dino_affine_for_rae = True
    print(f"[debug] remove_dino_affine_for_rae={remove_dino_affine_for_rae}")
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with amp_ctx:
        # A. VAE Forward
        target_feat_vae_norm = vae_model.preprocess(img).squeeze(1) 
        vae_out = vae_model(img, deterministic=True)
        recon_feat_vae_norm = vae_out.sample.squeeze(1) 

        # B. convert to RAE space 
        gt_rae_input = vae_model.convert_to_rae_space(
            target_feat_vae_norm,
            target_dim=768,
            remove_dino_norm_affine=remove_dino_affine_for_rae,
        )
        vae_rae_input = vae_model.convert_to_rae_space(
            recon_feat_vae_norm,
            target_dim=768,
            remove_dino_norm_affine=remove_dino_affine_for_rae,
        )

    # C. RAE decode
    dec_gt = rae_model.decode(gt_rae_input.permute(0, 3, 1, 2).float())
    dec_vae = rae_model.decode(vae_rae_input.permute(0, 3, 1, 2).float())
    print_tensor_stats("gt_rae_input", gt_rae_input)
    print_tensor_stats("vae_rae_input", vae_rae_input)
    print_tensor_stats("dec_gt", dec_gt)
    print_tensor_stats("dec_vae", dec_vae)

    print("[debug] MSE Loss between dec_gt and dec_vae: " + str(F.mse_loss(dec_gt, dec_vae).item()))

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
    print("✅ Single Image Processed")

    plot_results(results, save_path="vae_rae_eval_imagenet_single.png")

if __name__ == "__main__":
    main()
