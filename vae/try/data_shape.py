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

    # 模拟你的 transform
img_size = 224
transform = make_transforms(img_size=img_size)

(
    dataset, val_dataset, traj_dataset, val_traj_dataset, 
    train_dataloader, val_dataloader, dist_sampler, viz_val_loader
) = init_data(
      data_paths=["/home/lizijia/Gen-JEPA/vae/data/pusht_noise"],
      batch_size=16,          
      num_workers=8,           
      dataset_type="custom",
      frameskip=1,
      num_hist=3,
      num_pred=6, 
      transform=transform,
      pin_mem=True,        
      persistent_workers=True, 
      rank=0,
      world_size=1,
      n_rollout=1 
    )

# --- 变量名探测代码 ---
# 1. 获取一个 Batch
raw_batch = next(iter(train_dataloader))

print("\n" + "📊" * 15)
print("  Push-T Batch Index Analysis")
print("📊" * 15)

# 2. 遍历元组/列表中的每一个元素
for i, data in enumerate(raw_batch):
    if isinstance(data, dict):
        print(f"\n[索引 {i}] - 类型: 字典 (dict)")
        for k, v in data.items():
            print(f"   ┗━ 键: '{k}' | Shape: {list(v.shape)}")
    elif torch.is_tensor(data):
        print(f"\n[索引 {i}] - 类型: Tensor")
        print(f"   ┗━ Shape: {list(data.shape)}")
    else:
        print(f"\n[索引 {i}] - 类型: {type(data)}")

print("\n" + "="*40)