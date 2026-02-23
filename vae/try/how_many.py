import sys
import os
import torch

# 1. 设置路径（和你 online_train_vae.py 逻辑一致）
current_file_path = os.path.abspath(__file__)
jepa_root = "/home/lizijia/Gen-JEPA" # 直接用你的根目录
if jepa_root not in sys.path:
    sys.path.insert(0, jepa_root)

from data_processing.utils import init_data
from data_processing.transforms import make_transforms

def check_stats():
    # 模拟你的 transform
    img_size = 224
    transform = make_transforms(img_size=img_size)

    print("--- 🔍 正在扫描数据集，请稍候... ---")
    
    # 调用 init_data，参数必须和你运行中的脚本完全一致
    (
        dataset, val_dataset, traj_dataset, val_traj_dataset, 
        _, _, _, _
    ) = init_data(
        data_paths=["/home/lizijia/Gen-JEPA/vae/data/pusht_noise"],
        batch_size=64,          
        num_workers=4, # 检查数据不需要太多 worker
        dataset_type="custom",
        frameskip=1,
        num_hist=3,
        num_pred=6,
        transform=transform,
        rank=0,
        world_size=1
    )

    print("\n" + "="*40)
    print("📊 PUSHT 数据集统计报告")
    print("="*40)
    
    # 1. 轨迹统计 (视频数量)
    print(f"🎬 视频轨迹 (Trajectories):")
    print(f"   - 训练集: {len(traj_dataset)} 个视频")
    print(f"   - 验证集: {len(val_traj_dataset)} 个视频")
    
    # 2. 样本统计 (切分后的 Sample 数量)
    print(f"\n🧩 样本总数 (Total Samples):")
    print(f"   - 训练集: {len(dataset)} 个样本")
    print(f"   - 验证集: {len(val_dataset)} 个样本")
    
    # 3. 计算配比
    ratio = len(val_dataset) / (len(dataset) + len(val_dataset)) * 100
    print(f"\n📐 验证集占比: {ratio:.2f}%")
    
    # 4. 每个 Batch 的覆盖情况
    steps_per_epoch = len(dataset) // 64
    print(f"\n⏱️ 训练节奏 (Batch Size = 64):")
    print(f"   - 每个 Epoch 约包含: {steps_per_epoch} 个 Steps")
    print(f"   - 如果 val_every=1000，则每个 Epoch 验证约 {steps_per_epoch/1000:.1f} 次")
    print("="*40)

if __name__ == "__main__":
    check_stats()