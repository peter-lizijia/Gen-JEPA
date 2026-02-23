import torch
import os
import sys
from omegaconf import OmegaConf

# --- 简单配置 ---
vae_config_path = "/home/lizijia/Gen-JEPA/vae/models/configs/kubric/vae/dino_vae_prediction1.yaml"
vae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/try/vae_checkpoints/kubric.pt"
vae_stats_path = "/home/lizijia/Gen-JEPA/vae/data/push_t_dino_b_3000batch.npz"

current_file_path = os.path.abspath(__file__)
try_dir = os.path.dirname(current_file_path)
vae_root = os.path.dirname(try_dir)
sys.path.insert(0, vae_root)
try:
    from models.vae.dino_raw_vae import DINORawVAE
except ImportError:
    print("❌ 无法导入 DINORawVAE")
    sys.exit(1)

def debug_kubric_loading():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 强制使用 bfloat16 以匹配高性能 GPU 算子
    dtype = torch.bfloat16
    print(f"🚀 开始审计 Checkpoint: {os.path.basename(vae_ckpt_path)} (使用 {dtype})")

    # 1. 初始化模型
    config = OmegaConf.load(vae_config_path)
    model_params = config.model if 'model' in config else config
    model = DINORawVAE(**model_params).to(device).to(dtype).eval()

    # 2. 采样关键层权重检查
    layer_name = "vae.decoder_backbone.blocks[-1].norm"
    try:
        layer_to_check = model.vae.decoder_backbone.blocks[-1].norm
        initial_weight_sum = layer_to_check.weight.abs().sum().item()
    except:
        layer_to_check = None
        initial_weight_sum = 0
    
    print(f"📊 {layer_name} 初始权重 Sum (随机): {initial_weight_sum:.6f}")

    # 3. 加载 Checkpoint
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    new_state_dict = {}
    vae_prefix_count = 0
    for k, v in state_dict.items():
        if k.startswith("vae."):
            new_state_dict[k.replace("vae.", "")] = v
            vae_prefix_count += 1
        else:
            new_state_dict[k] = v
    
    # 转换为 bfloat16 加载
    new_state_dict = {k: v.to(dtype) for k, v in new_state_dict.items()}
    missing, unexpected = model.vae.load_state_dict(new_state_dict, strict=False)
    
    # 再次检查权重是否变化
    loaded_weight_sum = layer_to_check.weight.abs().sum().item() if layer_to_check else 0
    print(f"📊 {layer_name} 加载后权重 Sum: {loaded_weight_sum:.6f}")
    
    if abs(loaded_weight_sum - initial_weight_sum) < 1e-6:
        print(f"❌ 严重警告: 权重在加载后没有变化！")
    else:
        print(f"✅ 权重已更新 (Diff: {abs(loaded_weight_sum - initial_weight_sum):.6f})")

    # 4. 数值分布探针 (带 autocast)
    print("\n🔬 执行数值分布检查 (Forward Pass)...")
    import numpy as np
    
    # 造一个伪数据
    dummy_img = torch.randn(1, 1, 3, 224, 224).to(device).to(dtype)
    
    if os.path.exists(vae_stats_path):
        s = np.load(vae_stats_path)
        vae_mean = torch.from_numpy(s['mean']).to(device).to(dtype)
        vae_std = torch.from_numpy(s['std']).to(device).to(dtype)
    else:
        vae_mean, vae_std = torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)

    with torch.amp.autocast('cuda', dtype=dtype):
        with torch.no_grad():
            # 获取特征
            target_norm = model.preprocess(dummy_img).squeeze(1)
            out = model(dummy_img)
            recon_norm = out.sample.squeeze(1)
            
            t_sub = target_norm[..., -768:]
            r_sub = recon_norm[..., -768:]
            
            # 反归一化逻辑
            m_sub = vae_mean[..., -768:] if vae_mean.numel() > 1 else vae_mean
            s_sub = vae_std[..., -768:] if vae_std.numel() > 1 else vae_std
                
            t_raw = t_sub * s_sub + m_sub
            r_raw = r_sub * s_sub + m_sub

            print("\n📈 数值对比:")
            print(f"  [1] VAE 归一化空间 (Norm Space):")
            print(f"      Target Mean: {t_sub.mean().item():.4f}, Std: {t_sub.std().item():.4f}")
            print(f"      Recon  Mean: {r_sub.mean().item():.4f}, Std: {r_sub.std().item():.4f}")
            
            print(f"  [2] 原始特征空间 (Raw Space):")
            print(f"      Target Mean: {t_raw.mean().item():.4f}, Std: {t_raw.std().item():.4f}")
            print(f"      Recon  Mean: {r_raw.mean().item():.4f}, Std: {r_raw.std().item():.4f}")
            
            print(f"  [3] 核心数值量级 (L1 Error): {torch.abs(r_raw - t_raw).mean().item():.4f}")

if __name__ == "__main__":
    debug_kubric_loading()