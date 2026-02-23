import os
import sys
# 1. 动态获取根目录路径
# 你的脚本在 /home/lizijia/Gen-JEPA/vae/try/online_train_vae.py
# 我们需要把 /home/lizijia/Gen-JEPA/vae 放入路径，这样才能 import models
# 我们还需要把 /home/lizijia/Gen-JEPA 放入路径，这样才能 import data_processing

current_file_path = os.path.abspath(__file__) # 得到脚本绝对路径
try_dir = os.path.dirname(current_file_path)   # /vae/try
vae_root = os.path.dirname(try_dir)            # /vae
jepa_root = os.path.dirname(vae_root)          # /Gen-JEPA

# 2. 将这些关键路径插入 sys.path 的最前面
# 顺序很重要：越具体的路径放越前面
critical_paths = [
    vae_root,         # 优先级最高：为了找到 models.vae.dino_raw_vae
    jepa_root,        # 为了找到 data_processing.utils
    os.path.join(jepa_root, "data_processing") # 为了 utils.py 内部能找到 src
]

for p in critical_paths:
    if p not in sys.path:
        sys.path.insert(0, p)


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
from einops import rearrange
import random

# 导入你的模块
from models.vae.dino_raw_vae import DINORawVAE
from data_processing.utils import init_test_data 
from data_processing.transforms import make_transforms

def visualize_multi_features_pca(original_feats, reconstructed_feats, save_path="vae_test_comparison.png"):
    """
    一次性对比多组特征
    original_feats/reconstructed_feats: List of [1, H, W, C] tensors
    """
    n = len(original_feats)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    
    for i in range(n):
        # 准备数据 [256, 3072]
        feat1 = original_feats[i][0].view(-1, original_feats[i].shape[-1]).float().cpu().numpy()
        feat2 = reconstructed_feats[i][0].view(-1, reconstructed_feats[i].shape[-1]).float().cpu().numpy()
        
        # 联合降维以保持颜色坐标系一致
        combined = np.concatenate([feat1, feat2], axis=0)
        pca = PCA(n_components=3)
        pca.fit(combined)
        
        def process_pca(f):
            low_dim = pca.transform(f)
            low_dim = (low_dim - low_dim.min()) / (low_dim.max() - low_dim.min() + 1e-8)
            return low_dim.reshape(16, 16, 3)

        vis1 = process_pca(feat1)
        vis2 = process_pca(feat2)

        axes[i, 0].imshow(vis1)
        axes[i, 0].set_title(f"Sample {i+1}: Original DINO")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(vis2)
        axes[i, 1].set_title(f"Sample {i+1}: VAE Recon")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Comparison figure saved to: {save_path}")
    plt.show()

@torch.no_grad()
def test_vae_performance(test_data_path, config_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模型
    config = OmegaConf.load(config_path)
    stats_path = "/home/lizijia/Gen-JEPA/vae/data/push_t_stats.npz" 
    model = DINORawVAE(**config, feature_stats=stats_path).to(device)
    
    # 加载权重
    print(f"⌛ Loading Checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.vae.load_state_dict(state_dict)
    model.eval()

    # 2. 初始化测试数据
    # 注意：这里调用你提供的 init_test_data 或类似的函数
    transform = make_transforms(img_size=224)
    test_dataset, test_traj, test_loader = init_test_data(
        test_data_path=test_data_path,
        batch_size=1, # 设为1方便随机取帧
        num_workers=4,
        transform=transform
    )

    # 3. 随机抽取 3 个样本
    print(f"🎲 Randomly sampling 3 frames from test set...")
    all_indices = list(range(len(test_loader.dataset)))
    selected_indices = random.sample(all_indices, 3)
    
    orig_list = []
    recon_list = []

    for idx in selected_indices:
        # 获取单个样本
        batch = test_loader.dataset[idx]
        # batch[0] 通常是 obs，取决于你的 Dataset 返回值
        obs, action, state, _ = batch 
        img_tensor = obs['visual'].unsqueeze(0).to(device) # [1, T, 3, 224, 224]

        # 前向传播
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            target_feat_raw = model.preprocess(img_tensor)
            target_feat = rearrange(target_feat_raw, "b t h w c -> (b t) h w c")
            
            output = model(img_tensor)
            recon_feat = output.sample 

            orig_list.append(target_feat)
            recon_list.append(recon_feat)

    # 4. 可视化
    visualize_multi_features_pca(orig_list, recon_list)

if __name__ == "__main__":
    test_vae_performance(
        test_data_path="/home/lizijia/Gen-JEPA/vae/data/pusht_noise", # 确保路径正确
        config_path="/home/lizijia/Gen-JEPA/vae/configs/convnext_isotropic_base.yaml",
        ckpt_path="/home/lizijia/Gen-JEPA/vae/try/checkpoints/best_vae.pth" # 使用你保存的最佳模型
    )