import os
import sys
import torch
import hydra
import logging
import numpy as np
import matplotlib.pyplot as plt
import einops
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from PIL import Image
import torch.nn.parallel

# --- Project Imports ---
from vae.models.predictor.utilities.ddp import cleanup, setup_ddp
from vae.models.predictor.utilities.seed import seed_everything
from vae.models.predictor.autoregressive_dino_foresight_latent import AutoregressiveDINOForesightLatent
from data_processing.utils import init_data
from data_processing.transforms import make_inverse_transforms, make_transforms

# --- 补充导入：用于生成和 DINO-WM ---
from vae.models.predictor.train import sample
try:
    from clone.dino_wm.plan import load_model
except ImportError:
    print("Warning: Could not import 'plan'. Make sure DINO-WM source code is in PYTHONPATH.")
    def load_model(*args, **kwargs): pass

config_path = os.environ.get("CONFIG_PATH", "../configs/kubric")
config_name = os.environ.get("CONFIG_NAME", "dino_foresight_cfm.yaml")

# ==============================================================================
# 1. DINO-WM Inference Class
# ==============================================================================
class PushTInference:
    def __init__(self, checkpoint_path, config_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(config_path)
        print(f"[DINO-WM] Loading model from {checkpoint_path}...")
        
        self.model = load_model(Path(checkpoint_path), self.cfg, num_action_repeat=1, device=self.device)
        self.model = self.model.to(self.device).to(torch.bfloat16)
        self.model.eval()

    @torch.no_grad()
    def predict_next_frame(self, context_imgs, context_proprio, history_actions, target_action):
        context_imgs = context_imgs.to(self.device).to(torch.bfloat16)
        context_proprio = context_proprio.to(self.device).to(torch.bfloat16)
        full_actions = torch.cat([history_actions, target_action], dim=1).to(self.device).to(torch.bfloat16)
        
        obs_input = {"visual": context_imgs, "proprio": context_proprio}
        
        # Rollout
        z_obses, _ = self.model.rollout(obs_input, full_actions)
        pred_z_visual = z_obses['visual'][:, -1:, ...] # Last frame
        pred_z_proprio = z_obses['proprio'][:, -1:, ...]
        
        # Decode
        decoded_out, _ = self.model.decode_obs({
            "visual": pred_z_visual,
            "proprio": pred_z_proprio
        })
        
        pred_img = decoded_out['visual'] 
        return pred_img, pred_z_visual

# ==============================================================================
# 2. PCA Visualization Tool (修复：支持独立的 PCA)
# ==============================================================================
def apply_pca_visualization(gt_features, pred_features_dict):
    results = {}

    # 辅助函数：安全获取 H, W, C
    def get_spatial_info(f):
        if len(f.shape) == 3:
            return f.shape # h, w, c
        elif len(f.shape) == 2:
            l, c = f.shape
            h = int(np.sqrt(l))
            return h, h, c
        else:
            raise ValueError(f"Unexpected shape: {f.shape}")

    # --- 1. 处理主组 (GT 和 User Model) ---
    # 它们共享同一个 PCA (基于 GT 训练)，以便直观比较重建质量
    flat_gt = []
    shapes_gt = []
    
    for feat in gt_features:
        h, w, c = get_spatial_info(feat)
        flat_gt.append(feat.reshape(-1, c))
        shapes_gt.append((h, w))
    
    # 训练主 PCA
    fit_data_main = np.concatenate(flat_gt, axis=0).astype(np.float32)
    pca_main = PCA(n_components=3)
    pca_main.fit(fit_data_main)
    
    # 计算归一化范围
    main_trans = pca_main.transform(fit_data_main)
    p_min_main = main_trans.min(axis=0)
    p_max_main = main_trans.max(axis=0)
    range_main = p_max_main - p_min_main + 1e-6
    
    def process_with_pca(data, h, w, pca_model, p_min, range_val):
        trans = pca_model.transform(data.reshape(-1, data.shape[-1]).astype(np.float32))
        norm = (trans - p_min) / range_val
        return norm.reshape(h, w, 3).clip(0, 1)

    # 转换 GT
    results['gt'] = []
    curr = 0
    for h, w in shapes_gt:
        chunk = fit_data_main[curr : curr + h*w]
        # 这里直接复用已经 transform 过的部分会更快，但为了代码逻辑统一，重新调用 process
        results['gt'].append(process_with_pca(chunk, h, w, pca_main, p_min_main, range_main))
        curr += h*w

    # 转换 User Model (假设它和 GT 维度一致)
    if 'user' in pred_features_dict:
        results['user'] = []
        for feat in pred_features_dict['user']:
            h, w, c = get_spatial_info(feat)
            results['user'].append(process_with_pca(feat, h, w, pca_main, p_min_main, range_main))

    # --- 2. 处理 DINO-WM (独立 PCA) ---
    if 'wm' in pred_features_dict:
        results['wm'] = []
        wm_feats = pred_features_dict['wm']
        
        # 收集所有 WM 特征用于训练它自己的 PCA
        flat_wm = []
        shapes_wm = []
        for feat in wm_feats:
            h, w, c = get_spatial_info(feat)
            flat_wm.append(feat.reshape(-1, c))
            shapes_wm.append((h, w))
            
        fit_data_wm = np.concatenate(flat_wm, axis=0).astype(np.float32)
        
        # 训练 WM 专用 PCA
        pca_wm = PCA(n_components=3)
        pca_wm.fit(fit_data_wm)
        
        wm_trans = pca_wm.transform(fit_data_wm)
        p_min_wm = wm_trans.min(axis=0)
        p_max_wm = wm_trans.max(axis=0)
        range_wm = p_max_wm - p_min_wm + 1e-6
        
        # 转换 WM
        curr = 0
        for h, w in shapes_wm:
            chunk = fit_data_wm[curr : curr + h*w]
            results['wm'].append(process_with_pca(chunk, h, w, pca_wm, p_min_wm, range_wm))
            curr += h*w
            
    return results

# ==============================================================================
# 3. Main Comparison Logic
# ==============================================================================
@torch.no_grad()
def run_comparison(user_model, dino_wm_model, dataloader, device, save_path):
    user_model.eval()
    inv_transform = make_inverse_transforms(img_size=224)
    inv_transform.mean = inv_transform.mean.to(device)
    inv_transform.std = inv_transform.std.to(device)

    # 1. Get Data
    batch = next(iter(dataloader))
    data_dict = batch[0]
    action_batch = batch[1]

    visual = data_dict['visual'][0:1].to(device) 
    proprio = data_dict['proprio'][0:1].to(device)
    actions = action_batch[0:1].to(device) 
    
    raw_model = user_model.module if hasattr(user_model, "module") else user_model

    print(f"Input Visual Shape: {visual.shape}")

    # ========================================
    # A. User Model Prediction
    # ========================================
    print("Running User Model...")
    user_preds_pca = []
    context_len = 3 
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        gt_dino_feats = raw_model.vae.preprocess(visual) # [B, T, h, w, c]
        
        z_gt_flat = einops.rearrange(gt_dino_feats, "b t h w c -> (b t) h w c")
        z_dist = raw_model.vae.vae.encode(z_gt_flat)
        z_latent = z_dist.sample()
        z_latent = einops.rearrange(z_latent, "(b t) h w c -> b t h w c", b=visual.shape[0])
        
        z_ctx_input = z_latent[:, :context_len]
        
        for sample_idx in range(3):
            z_out_traj, _ = sample(
                raw_model,
                z_ctx=z_ctx_input,
                action=actions,
                proprio=proprio,
                steps=50
            )
            pred_latent = z_out_traj[:, -1]
            rec_dino = raw_model.vae.vae.decode(pred_latent)
            user_preds_pca.append(rec_dino[0].float().cpu().numpy())

    gt_dino_list = [f.float().cpu().numpy() for f in gt_dino_feats[0]]

    # ========================================
    # B. DINO-WM Prediction
    # ========================================
    print("Running DINO-WM...")
    wm_rgb, wm_latent = dino_wm_model.predict_next_frame(
        visual[:, :3], 
        proprio[:, :3], 
        actions[:, :2], 
        actions[:, 2:3]
    )
    
    wm_rgb_tensor = wm_rgb[0, 0] 
    if wm_rgb_tensor.min() < -1.5: 
        wm_rgb_restored = inv_transform(wm_rgb_tensor.unsqueeze(0))[0]
    else:
        wm_rgb_restored = wm_rgb_tensor
    wm_rgb_np = wm_rgb_restored.permute(1, 2, 0).float().cpu().numpy().clip(0, 1)
    
    wm_feat_raw = wm_latent[0, 0].float().cpu().numpy() 

    # ========================================
    # C. Visualization
    # ========================================
    print("Applying PCA...")
    pca_results = apply_pca_visualization(
        gt_features=gt_dino_list, 
        pred_features_dict={
            'user': user_preds_pca,
            'wm': [wm_feat_raw] 
        }
    )
    
    gt_rgb_all = inv_transform(visual[0])
    gt_rgb_np = gt_rgb_all.permute(0, 2, 3, 1).cpu().float().numpy().clip(0, 1)

    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    plt.suptitle("Comparison: User Model vs DINO-WM")

    # Row 1: GT RGB
    for i in range(4):
        axs[0, i].imshow(gt_rgb_np[i])
        axs[0, i].set_title(f"GT RGB T{i}")
        axs[0, i].axis('off')
        
    # Row 2: PCA (GT & WM)
    axs[1, 0].imshow(pca_results['gt'][2])
    axs[1, 0].set_title("GT PCA T2 (Last Ctx)")
    
    axs[1, 1].imshow(pca_results['gt'][3])
    axs[1, 1].set_title("GT PCA T3 (Target)")
    
    axs[1, 2].imshow(wm_rgb_np)
    axs[1, 2].set_title("DINO-WM (RGB)")
    
    axs[1, 3].imshow(pca_results['wm'][0])
    axs[1, 3].set_title("DINO-WM (PCA)")

    # Row 3: Ours PCA
    axs[2, 0].imshow(pca_results['user'][0])
    axs[2, 0].set_title("Ours PCA Sample 1")
    
    axs[2, 1].imshow(pca_results['user'][1])
    axs[2, 1].set_title("Ours PCA Sample 2")
    
    axs[2, 2].imshow(pca_results['user'][2])
    axs[2, 2].set_title("Ours PCA Sample 3")
    axs[2, 3].axis('off')

    plt.tight_layout()
    print(f"Saving comparison to {save_path}")
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# 4. Main Entry
# ==============================================================================
@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    rank, world_size, local_rank, device = setup_ddp()
    seed_everything(cfg.training.seed + rank)

    if rank == 0:
        print(f"--- Comparison Mode ---")
        print(f"Checkpoint: {cfg.resume.ckpt}")

    # 1. Initialize User Model (Standard)
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    if "vae" in cfg:
        user_model = AutoregressiveDINOForesightLatent(**model_kwargs, vae=cfg.vae)
    else:
        raise NotImplementedError("Model type not supported.")

    user_model = user_model.to(device).bfloat16()
    user_model.eval()

    if not cfg.resume.ckpt:
        raise ValueError("Error: Must provide resume.ckpt path!")

    # 2. Load Weights
    ckpt_path = cfg.resume.ckpt
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    state_dict = ckpt
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "ema_state_dict" in ckpt: 
        print("Note: Loading EMA weights.")
        state_dict = ckpt["ema_state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        new_state_dict[name] = v
    
    user_model.load_state_dict(new_state_dict, strict=False)
    user_model = torch.nn.parallel.DistributedDataParallel(user_model, device_ids=[local_rank])

    # 3. Load DINO-WM
    dino_wm_ckpt = "/home/lizijia/Gen-JEPA/clone/dino_wm/checkpoints/outputs/pusht/checkpoints/model_latest.pth"
    dino_wm_cfg = "/home/lizijia/Gen-JEPA/clone/dino_wm/checkpoints/outputs/pusht/hydra.yaml"
    dino_wm = PushTInference(dino_wm_ckpt, dino_wm_cfg, device=device)

    # 4. Data Loading
    img_size = 224
    transform = make_transforms(img_size=img_size)
    (_, _, _, _, _, val_dataloader, _, _) = init_data(
        data_paths=cfg.data.paths,
        batch_size=1,
        num_workers=4,           
        dataset_type="custom",
        frameskip=cfg.data.frameskip,
        num_hist=cfg.model.context_frames,
        num_pred=cfg.model.prediction_frames, 
        transform=transform,
        pin_mem=True,        
        rank=rank,             
        world_size=world_size
    )

    # 5. Run Comparison
    if rank == 0:
        output_dir = Path(cfg.experiment_path) / cfg.name / "comparisons"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        step_num = ckpt.get('step', 'unknown') if isinstance(ckpt, dict) else 'unknown'
        save_file = output_dir / f"compare_step_{step_num}.jpg"
        
        run_comparison(user_model, dino_wm, val_dataloader, device, save_file)
        
    cleanup()

if __name__ == "__main__":
    main()