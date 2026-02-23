import torch
import torch.nn.functional as F
from einops import rearrange
from torch.amp import autocast
import os

class VAETrainer:
    def __init__(self, model, lr=2e-4, kl_weight=1e-6):  #后续修改参数yaml
        self.model = model
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(self.model.vae.parameters(), lr=lr)
        self.kl_weight = kl_weight

        self.best_val_loss = float('inf')
        
        # 确保状态正确
        self.model.dino_v2.eval() # 老师永远不学
        self.model.vae.train()    # 学生努力学

    def train_step(self, images):
        """
        images: [B, T, 3, 224, 224] 的原始图像 Tensor
        """
        images = images.to(self.device) 
    
        # 建议将所有前向传播都包裹在 autocast 中
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # 1. 获取 Target (DINO 特征)
            with torch.no_grad():
                # 提取特征 + 归一化
                target_feat_raw = self.model.preprocess(images) 
                # 维度转换：[B, T, H, W, C] -> [B*T, H, W, C]
                target_feat = rearrange(target_feat_raw, "b t h w c -> (b t) h w c")

            # 2. VAE 前向传播
            # 这里的 self.model(images) 内部包含 Encoder 和 Decoder
            # 在 autocast 下，它们会跑在极快的 bfloat16 算子上
            output = self.model(images) 
            
            # 3. 提取结果
            recon_feat = output.sample  # VAE 重构出的特征
            kl_loss = output.kl.mean()
            
            # 4. 计算损失
            recon_loss = F.mse_loss(recon_feat, target_feat)
            total_loss = recon_loss + self.kl_weight * kl_loss
        
        # --- 结束 autocast ---

        # 5. 反向传播 (backward 通常在 autocast 外部)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }
    

    @torch.no_grad()
    def val_step(self, val_loader):
        """
        在验证集上运行一个完整的 Epoch 并返回平均 Loss
        """
        self.model.vae.eval()  # 切换到评估模式
        total_recon = 0.0
        total_kl = 0.0
        n = 0

        for batch in val_loader:
            # 这里的解包逻辑需与你的 online_train_vae.py 保持一致
            obs, action, state, _ = batch 
            images = obs['visual'].to(self.device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # 1. 获取 Target (逻辑与 train_step 完全一致)
                target_feat_raw = self.model.preprocess(images) 
                target_feat = rearrange(target_feat_raw, "b t h w c -> (b t) h w c")

                # 2. VAE 前向传播
                output = self.model(images) 
                
                # 3. 计算损失
                recon_loss = F.mse_loss(output.sample, target_feat)
                kl_loss = output.kl.mean()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n += 1

        self.model.vae.train()  # 验证结束，切换回训练模式
        
        return {
            "val/recon_loss": total_recon / n,
            "val/kl_loss": total_kl / n,
            "val/total_loss": (total_recon + self.kl_weight * total_kl) / n
        }

    def save_checkpoint(self, path, is_best=False):
        """
        path: 常规 checkpoint 的保存路径
        is_best: 是否同时保存为最佳模型
        """
        # 提取文件夹路径
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # 保存当前权重
        torch.save(self.model.vae.state_dict(), path)
        
        # 如果是最佳模型，额外存一份固定命名的文件
        if is_best:
            best_path = os.path.join(directory, "best_vae.pth")
            torch.save(self.model.vae.state_dict(), best_path)
            print(f"🌟 Found better model! Updated to: {best_path}")
        
        print(f"✅ VAE saved to: {path}")