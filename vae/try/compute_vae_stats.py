import torch
import numpy as np
import os
import sys
import ssl
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

# --- 1. 环境与路径配置 ---
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["WANDB_MODE"] = "offline" 

# 自动处理路径，确保能导入 models 和 data_processing
current_file_path = os.path.abspath(__file__)
vae_dir = os.path.dirname(current_file_path)   
jepa_root = "/home/lizijia/Gen-JEPA" # 建议直接写死你的根目录

if jepa_root not in sys.path:
    sys.path.insert(0, jepa_root)

# 导入你的模块
from models.vae.dino_raw_vae import DINORawVAE
from data_processing.utils import init_data
from data_processing.transforms import make_transforms

def compute_latent_stats():
    # ========================== 配置区域 ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 配置文件与 DINO 统计量路径
    config_path = "/home/lizijia/Gen-JEPA/vae/configs/convnext_isotropic_base.yaml"
    dino_stats_path = "/home/lizijia/Gen-JEPA/vae/data/push_t_dino_b_3000batch.npz"
    
    # 2. VAE 权重路径 (请指向你训练好的 best 权重)
    vae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/try/vae_checkpoints/kubric.pt" 
    
    # 3. 数据集路径
    data_path = "/home/lizijia/Gen-JEPA/vae/data/pusht_noise"
    
    # 4. 输出路径
    save_path = "/home/lizijia/Gen-JEPA/vae/data/vae_pusht_stats_3000batch.npz"

    # 5. 随机抽样控制：300个 batch 对于 PushT 来说已经足够覆盖全局分布
    MAX_BATCHES = 3000  
    BATCH_SIZE = 64
    # =============================================================

    print(f"🚀 正在初始化数据加载器 (使用 Predictor 训练参数)...")
    transform = make_transforms(img_size=224)

    # 复用 Predictor 的数据切片逻辑
    (
        dataset, _, _, _, 
        train_loader, _, _, _
    ) = init_data(
        data_paths=[data_path],
        batch_size=BATCH_SIZE,          
        num_workers=8,           
        dataset_type="custom",
        frameskip=5,
        num_hist=3,
        num_pred=1, 
        transform=transform,
        pin_mem=True,        
        persistent_workers=True, 
        rank=0,
        world_size=1,
        n_rollout=None # 如果数据集很大，不限制 rollout，靠 MAX_BATCHES 截断
    )

    print(f"✅ 数据加载器就绪。全集总样本数: {len(dataset)}")

    # --- 初始化 VAE 模型 ---
    print(f"🚀 正在从 {config_path} 加载 VAE 模型结构...")
    config = OmegaConf.load(config_path)
    model = DINORawVAE(**config, feature_stats=dino_stats_path)
    
    # 加载已训练好的权重
    if not os.path.exists(vae_ckpt_path):
        print(f"❌ 错误: 找不到权重文件 {vae_ckpt_path}，请检查路径。")
        return
        

    # --- 自适应加载逻辑 ---
    print(f"📂 正在加载权重: {vae_ckpt_path}")
    state_dict = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
    # 注意：根据你的保存逻辑，可能需要 state_dict['model_state_dict'] 或直接 state_dict
    
    # 逻辑 A：如果权重里有 'model_state_dict' (作者的格式)
    if 'model_state_dict' in state_dict:
        # 尝试加载到整个 model (DINORawVAE)，因为它包含 vae. 前缀
        msg = model.load_state_dict(state_dict['model_state_dict'], strict=False)
        print(f"✅ 以作者模式加载成功: {msg}")
        
    # 逻辑 B：如果权重里没有那个 Key (你原来代码保存的格式)
    else:
        # 提取字典内容
        actual_dict = state_dict.get('state_dict', state_dict)
        # 尝试直接加载到 model.vae
        msg = model.vae.load_state_dict(actual_dict, strict=False)
        print(f"✅ 以原生模式加载成功: {msg}")
    
    model.to(device).bfloat16() # 转换为 bf16
    model.eval()
    
    # --- 开始抽样提取 ---
    print(f"🚀 开始随机抽样计算 (上限: {MAX_BATCHES} batches)...")
    latents_accumulator = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, desc="Sampling Latents")):
            if i >= MAX_BATCHES:
                print(f"\n✋ 已达到抽样上限 ({MAX_BATCHES} batches)，停止提取。")
                break
                
            obs = batch[0] # obs 字典
            
            # 提取 Latents [B*T, 16, 16, 16]
            images = obs['visual'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # 这样内部 extract_features 出来的结果会自动适配 bf16
                output = model(images)
            z = output.latents
            
            # 展平所有维度，只保留 Channel 维度以计算统计量
            # 形状变换: [(B*T*16*16), 16]
            z_flat = rearrange(z, "b h w c -> (b h w) c").to(torch.float32)
            
            latents_accumulator.append(z_flat.cpu())

    # --- 计算统计量 ---
    print(f"📊 正在拼接数据并计算统计量 (这可能需要一分钟)...")
    all_z = torch.cat(latents_accumulator, dim=0) 
    
    mean = all_z.mean(dim=0).numpy()
    std = all_z.std(dim=0).numpy()

    # --- 保存结果 ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, mean=mean, std=std)

    print("\n" + "="*50)
    print(f"✅ 统计量计算完成！")
    print(f"📍 保存路径: {save_path}")
    print(f"📈 参与统计的数据点数 (Pixels): {all_z.shape[0]}")
    print(f"📈 隐空间通道数 (Channels): {mean.shape[0]}")
    print(f"   - Mean (Avg): {mean.mean():.4f}")
    print(f"   - Std  (Avg): {std.mean():.4f}")
    print("="*50)
    print("🚀 现在你可以使用这个 .npz 文件开始训练 Predictor 了。")

if __name__ == "__main__":
    compute_latent_stats()