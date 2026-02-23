import os
import torch
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import einops

# --- 复用 train.py 的组件 ---
from vae.models.predictor.utilities.ddp import cleanup, setup_ddp
from vae.models.predictor.utilities.seed import seed_everything
from vae.models.predictor.autoregressive_dino_foresight_latent import AutoregressiveDINOForesightLatent
from data_processing.utils import init_data
from data_processing.transforms import make_inverse_transforms, make_transforms

# --- 直接导入你修改好的函数 ---
from vae.models.predictor.train import evaluate, plot_samples

# 设置默认配置路径
config_path = os.environ.get("CONFIG_PATH", "../configs/kubric")
config_name = os.environ.get("CONFIG_NAME", "dino_foresight_cfm.yaml")

@torch.inference_mode()
def check_vae_reconstruction(model, batch, device):
    """
    [诊断工具] VAE 重构检查
    用于确认 VAE 是否工作正常，以及 Latent 是否包含细节信息。
    """
    # ==========================================
    # 1. 准备模型和工具
    # ==========================================
    raw_model = model.module if hasattr(model, "module") else model
    model_dtype = next(raw_model.parameters()).dtype
    inv_transform = make_inverse_transforms(img_size=224) 

    # ==========================================
    # 2. 准备数据
    # ==========================================
    # 提取第一帧 [1, 1, C, H, W]
    if isinstance(batch, (list, tuple)):
        if isinstance(batch[0], dict):
            x = batch[0]['visual'][0:1, 0:1, ...].to(device=device, dtype=model_dtype)
        else:
            x = batch[0][0:1, 0:1, ...].to(device=device, dtype=model_dtype)
    else:
        x = batch[0:1, 0:1, ...].to(device=device, dtype=model_dtype)

    print(f"\n--- [Diagnosis] Feature Statistics ---")
    
    with torch.amp.autocast('cuda', enabled=True, dtype=model_dtype):
        # A. 获取 GT DINO 特征
        # 注意：这里调用 preprocess，确保输入归一化是对的
        feat_gt = raw_model.vae.preprocess(x) 
        
        # B. VAE 重构
        feat_flat = einops.rearrange(feat_gt, "b t h w c -> (b t) h w c")
        inner_vae = raw_model.vae.vae if hasattr(raw_model.vae, 'vae') else raw_model.vae
        
        # 使用 mode() 取均值
        dist = inner_vae.encode(feat_flat)
        z = dist.mode() 
        feat_recon_flat = inner_vae.decode(z)
        feat_recon = einops.rearrange(feat_recon_flat, "(b t) h w c -> b t h w c", b=1, t=1)

    # ==========================================
    # 3. 数值诊断
    # ==========================================
    gt_np = feat_gt.float().cpu().numpy().reshape(-1, 3072)
    recon_np = feat_recon.float().cpu().numpy().reshape(-1, 3072)

    print(f"GT    Range: [{gt_np.min():.2f}, {gt_np.max():.2f}] | Std: {gt_np.std():.4f}")
    print(f"Recon Range: [{recon_np.min():.2f}, {recon_np.max():.2f}] | Std: {recon_np.std():.4f}")
    
    std_ratio = recon_np.std() / (gt_np.std() + 1e-6)
    print(f"Ratio (Recon/GT): {std_ratio:.2f}")

    # ==========================================
    # 4. 可视化处理
    # ==========================================
    
    # A. 统一 PCA (Shared) - 展示真实差异
    pca = PCA(n_components=3)
    pca.fit(gt_np) 
    
    gt_pca_shared = pca.transform(gt_np)
    recon_pca_shared = pca.transform(recon_np)
    
    # 归一化 (以 GT 为基准)
    p_min, p_max = gt_pca_shared.min(axis=0), gt_pca_shared.max(axis=0)
    range_val = p_max - p_min + 1e-6
    
    gt_vis_shared = (gt_pca_shared - p_min) / range_val
    recon_vis_shared = (recon_pca_shared - p_min) / range_val
    
    # B. 独立 PCA (Independent) - 强行拉伸对比度
    r_min, r_max = recon_pca_shared.min(axis=0), recon_pca_shared.max(axis=0)
    recon_vis_indep = (recon_pca_shared - r_min) / (r_max - r_min + 1e-6)

    # C. RGB 还原
    img_tensor = x[0].float().cpu() # [1, C, H, W]
    img_rgb = inv_transform(img_tensor)
    img_rgb = torch.clamp(img_rgb, 0, 1)
    img_rgb = img_rgb[0].permute(1, 2, 0).numpy() # [H, W, C]

    # ==========================================
    # 5. 绘图
    # ==========================================
    h, w = feat_gt.shape[2], feat_gt.shape[3]
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original RGB
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original RGB")
    axs[0].axis('off')
    
    # 2. GT PCA
    axs[1].imshow(gt_vis_shared.reshape(h, w, 3).clip(0, 1))
    axs[1].set_title("GT Feature PCA")
    axs[1].axis('off')
    
    # 3. Recon PCA (Shared Scale)
    axs[2].imshow(recon_vis_shared.reshape(h, w, 3).clip(0, 1))
    axs[2].set_title(f"Recon (True Scale)\nstd_ratio={std_ratio:.2f}")
    axs[2].axis('off')
    
    # 4. Recon PCA (Normalized)
    axs[3].imshow(recon_vis_indep.reshape(h, w, 3).clip(0, 1))
    axs[3].set_title("Recon (Auto-Contrast)\nCheck structure here!")
    axs[3].axis('off')
    
    plt.tight_layout()
    return fig

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    rank, world_size, local_rank, device = setup_ddp()
    seed_everything(cfg.training.seed + rank)

    if rank == 0:
        print(f"--- Evaluation Mode (Clean Version) ---")
        print(f"Checkpoint: {cfg.resume.ckpt}")

    # =========================================================
    # 1. 初始化模型 (标准方式)
    # =========================================================
    # 安全地解包参数，避免 action_dim 重复报错 (如果之前改了 train.py 这步也是安全的)
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    # 如果 cfg.model 里有 action_dim，这里就自动带上了；如果没有，模型会去读默认值
    
    if "vae" in cfg:
        model = AutoregressiveDINOForesightLatent(**model_kwargs, vae=cfg.vae)
    else:
        raise NotImplementedError("Model type not supported.")

    model = model.to(device).bfloat16()
    model.eval()

    if not cfg.resume.ckpt:
        raise ValueError("Error: Must provide resume.ckpt path!")

    # =========================================================
    # 2. 加载权重 (标准方式)
    # =========================================================
    ckpt_path = cfg.resume.ckpt
    print(f"Loading weights from {ckpt_path}...")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 兼容性处理：提取 model_state_dict
    state_dict = ckpt
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "ema_state_dict" in ckpt: # 也可以选择加载 EMA 权重进行评估
        print("Note: Loading EMA weights for evaluation.")
        state_dict = ckpt["ema_state_dict"]

    # 处理 DDP 的 'module.' 前缀问题
    # 如果 checkpoint 有 module. 但当前模型没有 (或者反之)，需要调整 key
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:] # 去掉 module.
        new_state_dict[name] = v
    
    # 加载权重
    # strict=True 是最佳实践，因为现在 key 应该完全匹配了
    # 但如果为了容错，可以改回 strict=False 并打印 missing_keys
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    if rank == 0:
        print(f"Load Result: {msg}")
        if len(msg.missing_keys) > 0:
            print(f"⚠️ Warning: Missing keys: {msg.missing_keys[:5]} ...")
            # 只有当 missing 的不是 VAE 核心层时才忽略
            if any("encoder" in k for k in msg.missing_keys):
                print("❌ CRITICAL: VAE keys seem to be missing! Check your checkpoint.")

    # =========================================================
    # 3. 准备数据
    # =========================================================
    img_size = 224
    transform = make_transforms(img_size=img_size)
    
    # 注意：这里 num_pred=1 是为了可视化方便，你可以根据需要改回 cfg.model.prediction_frames
    (_, _, _, _, _, val_dataloader, _, viz_val_loader) = init_data(
        data_paths=cfg.data.paths,
        batch_size=cfg.training.batch_size,          
        num_workers=4,           
        dataset_type="custom",
        frameskip=cfg.data.frameskip,
        num_hist=cfg.model.context_frames,
        num_pred=cfg.model.prediction_frames, 
        transform=transform,
        pin_mem=True,        
        persistent_workers=True, 
        rank=rank,             
        world_size=world_size,
        n_rollout=None 
    )

    # =========================================================
    # 4. 执行评估与可视化 (仅 Rank 0)
    # =========================================================
    
    # A. 计算指标 (全员参与)
    print("Running evaluation metrics...")
    eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader
    # mean_future_mse, min_of_k_mse, mean_of_k_mse = evaluate(model, eval_loader, device)
    
    if rank == 0:
        # print(f"\n[Evaluation Result]")
        # print(f"Mean Future MSE: {mean_future_mse.item():.6f}")
        # print(f"Min of K MSE:    {min_of_k_mse.item():.6f}")

        print("\n" + "="*40)
        print("Generating visualization samples...")
        
        save_dir = Path(cfg.experiment_path) / cfg.name / "eval_results"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 取一个 Batch
        viz_batch = next(iter(val_dataloader))

        # 1. 专项检查 VAE 重构 (Diagnosis)
        print("Running VAE reconstruction check...")
        try:
            fig = check_vae_reconstruction(model, viz_batch, device)
            step_num = ckpt.get('step', 0) if isinstance(ckpt, dict) else 0
            vae_debug_path = save_dir / f"vae_recon_check_{step_num}.png"
            fig.savefig(vae_debug_path)
            print(f"VAE debug image saved to: {vae_debug_path}")
            plt.close(fig) 
        except Exception as e:
            print(f"Error in VAE check: {e}")
        
        # 2. 运行完整的 plot_samples (Prediction)
        print("Plotting prediction samples...")
        pil_image = plot_samples(model, viz_batch, device)
        
        step_num = ckpt.get('step', 0) if isinstance(ckpt, dict) else 0
        save_path = save_dir / f"eval_prediction_{step_num:08d}.jpg"
        pil_image.save(save_path)
        print(f"Sample image saved to: {save_path}")

    cleanup()

if __name__ == "__main__":
    main()