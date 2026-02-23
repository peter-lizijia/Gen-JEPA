from contextlib import nullcontext

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from omegaconf import OmegaConf
import os
import ssl
from PIL import Image
import torchvision.transforms as T

# 确保你的项目路径在 PYTHONPATH 中，或者该脚本与 models 文件夹在同一层级
from models.vae.dino_raw_vae import DINORawVAE 

# 解决 DINOv2 权重下载时的 SSL 证书验证问题
ssl._create_default_https_context = ssl._create_unverified_context

class PushTImageDataset(Dataset):
    def __init__(self, root):
        self.root = root
        # 搜寻 png, jpg, jpeg 格式的图片
        self.image_files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()

        # DINOv2 推荐的标准输入尺寸
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

def compute_stats(image_dataset, config_path, save_path="data/push_t_stats.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载配置
    config = OmegaConf.load(config_path)
    
    # 2. 实例化模型
    # 使用 **config 将 yaml 里的字段解包给 DINORawVAE 的 __init__ 参数
    print(f"正在初始化 DINORawVAE (Backbone: {config.dinov2_variant})...")
    model = DINORawVAE(**config).to(device)
    model.eval()

    # 因为数据集很小，batch_size=1 最稳妥，num_workers=0 防止多进程报错
    loader = DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_features = []

    print(f"开始提取特征 (期望维度: {config.input_dim})...")
    
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with amp_ctx:
        with torch.no_grad():
            for imgs in tqdm(loader):
                imgs = imgs.to(device)
                
                # 使用作者提供的函数：它会自动获取 intermediate_layers [2, 5, 8, 11]
                # 并进行拼接，输出形状为 [B, Tokens, 1152]
                features = model.extract_features(imgs) 
                
                # 将空间维度和 batch 维度展平，只保留通道维度 [N, 1152]
                features_flat = features.cpu().numpy().reshape(-1, features.shape[-1])
                all_features.append(features_flat)

    # 合并所有样本的特征
    all_features_stack = np.concatenate(all_features, axis=0)

    # 3. 计算通道维度的均值和标准差
    print("正在计算均值和标准差...")
    mean = np.mean(all_features_stack, axis=0)
    std = np.std(all_features_stack, axis=0)

    # 4. 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, mean=mean, std=std)
    
    print("-" * 30)
    print(f"✅ 统计量计算完成并保存至: {save_path}")
    print(f"特征总数 (Tokens): {all_features_stack.shape[0]}")
    print(f"特征通道数: {mean.shape[0]}")
    print(f"Mean (前3维): {mean[:3]}")
    print(f"Std  (前3维): {std[:3]}")
    print("-" * 30)

if __name__ == "__main__":
    # --- 路径配置 ---
    # 修改为你的随机样本图片路径
    IMG_ROOT = "/home/lizijia/Gen-JEPA/vae/try/imagenet_stats"
    # 修改为你的配置文件路径
    CONFIG_PATH = "/home/lizijia/Gen-JEPA/vae/configs/convnext_isotropic_base.yaml"
    # 输出文件路径
    SAVE_PATH = "/home/lizijia/Gen-JEPA/vae/try/imagenet_stats/stats.npz"

    # 运行
    if not os.path.exists(IMG_ROOT):
        print(f"❌ 找不到图片目录: {IMG_ROOT}")
    else:
        dataset = PushTImageDataset(root=IMG_ROOT)
        compute_stats(dataset, CONFIG_PATH, SAVE_PATH)