import torch
import numpy as np
from tqdm import tqdm
import os
import sys
from omegaconf import OmegaConf

# --- 1. 环境路径设置 (确保能导入你的项目模块) ---
current_file_path = os.path.abspath(__file__)
vae_dir = os.path.dirname(current_file_path)
project_root = "/home/lizijia/Gen-JEPA" # 你的项目根目录

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "vae"))

# 导入你的模块
from models.vae.dino_raw_vae import DINORawVAE
from data_processing.utils import init_data 
from data_processing.transforms import make_transforms

def compute_and_save_stats():
    # ================= 配置区域 =================
    device = torch.device("cuda")
    
    # 配置文件路径
    config_path = "/home/lizijia/Gen-JEPA/vae/models/configs/kubric/vae/rae_vae_base.yaml"
    # 输出文件保存路径
    save_path = "/home/lizijia/Gen-JEPA/vae/data/push_t_dino_b_rae_3000batch.npz"
    # 数据集路径
    data_path = "/home/lizijia/Gen-JEPA/vae/data/pusht_noise"
    
    # 采样设置
    MAX_BATCHES = 3000  # 采样批次上限
    BATCH_SIZE = 64     # 增大 Batch Size 以加快速度
    FRAMESKIP = 5       # 跳帧采样，增加数据多样性
    # ===========================================

    print("🚀 正在初始化模型...")
    config = OmegaConf.load(config_path)
    
    # 初始化时不加载 feature_stats，因为我们就是要算它
    # 重点：转为 bfloat16 以适配新显卡
    model = DINORawVAE(**config.model).to(device).bfloat16()
    model.eval()

    print("🚀 正在加载数据...")
    # 使用标准 transform
    transform = make_transforms(img_size=224)
    
    n_hist = 3
    n_pred = 1
    
    # 获取 DataLoader
    # init_data 返回: (dataset, val_dataset, traj_dataset, val_traj_dataset, train_loader, ...)
    # 我们只需要第 5 个返回值: train_loader
    return_values = init_data(
        data_paths=[data_path],
        batch_size=BATCH_SIZE,
        dataset_type="custom",
        num_workers=8,
        frameskip=FRAMESKIP,
        num_hist=n_hist,
        num_pred=n_pred,
        transform=transform, # 别忘了传入 transform
        rank=0, 
        world_size=1
    )
    loader = return_values[4] # 取出 train_loader

    # --- 初始化累加器 (使用 float64 避免精度丢失) ---
    # DINOv2 Base = 768
    feature_dim = 768 
    
    # sum_x: 特征和, sum_x2: 特征平方和, n_total: 总像素点数
    sum_x = torch.zeros(feature_dim, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros(feature_dim, dtype=torch.float64, device=device)
    n_total = 0

    print(f"🚀 开始计算 DINO 统计量 (Max Batches: {MAX_BATCHES})...")

    with torch.no_grad():
        # 强制使用 BF16 混合精度上下文
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            
            for i, batch in enumerate(tqdm(loader, total=MAX_BATCHES)):
                if i >= MAX_BATCHES:
                    break
                
                # 1. 获取图像 [B, T, 3, 224, 224]
                # obs 是字典，通常在 batch[0]
                obs = batch[0]
                imgs = obs['visual'].to(device)
                
                # 合并 B 和 T 维度 -> [B*T, 3, 224, 224]
                imgs = imgs.flatten(0, 1)
                
                # 2. 提取特征
                # DINOv2 输出: [N, Tokens, Channels] (例如 [B*T, 256, 768])
                feat = model.extract_features(imgs)
                
                # 3. 展平空间维度 -> [Total_Pixels, Channels]
                feat_flat = feat.reshape(-1, feat.shape[-1])
                
                # 4. 累加计算 (转为 float64 进行累加，保证精度)
                # 注意：这里 feat 是 bf16，转 double 再算
                feat_double = feat_flat.to(torch.float64)
                
                sum_x += feat_double.sum(dim=0)
                sum_x2 += (feat_double ** 2).sum(dim=0)
                n_total += feat_double.shape[0]

    print("📊 正在计算最终均值和标准差...")
    
    # --- 最终计算 ---
    # Mean = Sum / N
    mean = sum_x / n_total
    
    # Var = E[X^2] - (E[X])^2
    # Std = sqrt(Var)
    variance = (sum_x2 / n_total) - (mean ** 2)
    
    # 处理可能的负值（数值误差导致极小的负数）
    variance = torch.clamp(variance, min=0.0)
    std = torch.sqrt(variance)

    # 转回 cpu numpy float32 保存
    final_mean = mean.cpu().numpy().astype(np.float32)
    final_std = std.cpu().numpy().astype(np.float32)

    # --- 保存 ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, mean=final_mean, std=final_std)
    
    print("=" * 40)
    print(f"✅ 统计量已保存至: {save_path}")
    print(f"📈 统计样本数 (Pixels): {n_total}")
    print(f"🧩 特征维度: {final_mean.shape[0]}")
    print(f"   Mean (前5位): {final_mean[:5]}")
    print(f"   Std  (前5位): {final_std[:5]}")
    print("=" * 40)

if __name__ == "__main__":
    compute_and_save_stats()