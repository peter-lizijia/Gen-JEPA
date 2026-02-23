import torch
from utils import init_data

from transforms import make_transforms

# 1. image transform
img_size = 224
transform = make_transforms(img_size=img_size)

# 2. initialize data loaders for online training
(
    dataset,            # 原始训练集对象
    val_dataset,        # 验证集对象
    traj_dataset,       # 包含 action_mean/std 的重要统计对象
    val_traj_dataset,
    data_loader,        # <--- 你的核心：训练用实时加载器
    val_data_loader,    # 验证用加载器
    dist_sampler,
    viz_val_loader
) = init_data(
    data_paths=["/home/lizijia/data/pusht_noise"], # 替换为你的 mp4 所在目录
    batch_size=8,
    dataset_type="custom",
    num_workers=16,         # 建议设为 16，充分利用 32 核 CPU 解码
    frameskip=5,            # 每隔 5 帧取一次
    num_hist=3,             # 历史帧数
    num_pred=1,             # 预测帧数
    transform=transform,
    rank=0,                 # 单卡训练设为 0
    world_size=1            # 单卡训练设为 1
)

# 需要改成正式运行版：+yaml，分布式

device = torch.device("cuda")

# 开启实时读取循环
for epoch in range(10):
    # 如果是分布式训练，需要 set_epoch 保证 shuffle 随机性
    dist_sampler.set_epoch(epoch) 
    
    for i, batch in enumerate(data_loader):
        # 实时解包 (Push-T 包含 obs, action, state, reward)
        obs, action, state, reward = batch
        
        # 此时的 obs['visual'] 已经是 [8, 4, 3, 224, 224] 的 Tensor
        # 8 是 batch_size, 4 是帧数 (3 hist + 1 pred)
        
        # 搬运到 GPU
        images = obs['visual'].to(device, non_blocking=True)
        actions = action.to(device, non_blocking=True)
        
        # 运行你的 VAE 或 JEPA 训练逻辑...
        
        if i % 10 == 0:
            print(f"Batch {i} 实时解码完成并喂入 GPU")