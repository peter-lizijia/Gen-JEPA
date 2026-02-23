import ssl
import sys
import os

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "True"
os.environ["CURL_CA_BUNDLE"] = ""

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
from data_processing.utils import init_data
from omegaconf import OmegaConf
import torch.distributed as dist
import wandb

from data_processing.transforms import make_transforms

from models.vae.dino_raw_vae import DINORawVAE
from train_vae import VAETrainer


def main():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    save_dir = "/home/lizijia/Gen-JEPA/vae/models/vae/checkpoints/rae_vae"
    os.makedirs(save_dir, exist_ok=True)
    # 2. 初始化 wandb
    # project: 你的项目名称
    # name: 这一次运行的实验名称（如果不设置，wandb 会自动生成搞笑的名称）
    wandb.init(
        project="gen-jepa_rae", 
        name="vae_pusht_batch64_workers8_val",
        config={
            "learning_rate": 2e-4,
            "architecture": "DINORawVAE",
            "dataset": "PushT",
            "batch_size": 64,
        }
    )
    # 2. image transform
    img_size = 224
    transform = make_transforms(img_size=img_size)

    device = torch.device("cuda")


    # 3. initialize data loaders
    print("--- 🚀 Initializing Data Loaders... ---")
    (
        dataset, val_dataset, traj_dataset, val_traj_dataset, 
        data_loader, val_data_loader, dist_sampler, viz_val_loader
    ) = init_data(
        data_paths=["/home/lizijia/Gen-JEPA/vae/data/pusht_noise"],
        batch_size=64,          
        num_workers=8,           
        dataset_type="custom",
        frameskip=5,
        num_hist=3,
        num_pred=1,
        transform=transform,
        pin_mem=True,        
        persistent_workers=True, 
        rank=0,
        world_size=1
    )

    # --- 步骤 4 细化 ---
    print("--- 🚀 Loading Config... ---",flush=True)
    config = OmegaConf.load("/home/lizijia/Gen-JEPA/vae/models/configs/kubric/vae/rae_vae_base.yaml")
    
    print("--- 🚀 Creating DINORawVAE model... ---",flush=True)
    model = DINORawVAE(**config.model)
    
    model = model.to(device)

    # --- 步骤 5 细化 ---
    print("--- 🚀 Instantiating VAETrainer... ---",flush=True)
    vae_trainer = VAETrainer(model, lr=2e-4)



    val_every = 1000   # 每 1000 个 batch 记录一次验证集 Loss
    save_every = 5000  # 每 2000 个 batch 保存一次普通 Checkpoint
    global_step = 0    

    print("--- 🚀 Training Loop Started ---", flush=True)
    for epoch in range(3):
        if dist_sampler is not None:
            dist_sampler.set_epoch(epoch)
        
        model.vae.train() 
        for i, batch in enumerate(data_loader):
            # 1. 训练步
            obs, action, state, _ = batch
            images = obs['visual'].to(device, non_blocking=True)
            loss_dict = vae_trainer.train_step(images)
            global_step += 1
            
            # 2. 实时记录训练 Loss
            wandb.log({
                "train/total_loss": loss_dict["total_loss"],
                "train/recon_loss": loss_dict["recon_loss"],
                "train/kl_loss": loss_dict["kl_loss"],
                "global_step": global_step,
            })
            
            # 3. --- 核心修改：定期跑验证并记录 ---
            if global_step % val_every == 0:
                print(f"--- 🔍 Step {global_step}: Running Validation... ---", flush=True)
                
                # 调用你刚才在 train_vae.py 写好的 val_step
                val_metrics = vae_trainer.val_step(val_data_loader)
                
                # 更新最佳模型逻辑
                current_val_loss = val_metrics["val/recon_loss"]
                is_best = current_val_loss < vae_trainer.best_val_loss
                if is_best:
                    vae_trainer.best_val_loss = current_val_loss
                
                # 将验证 Loss 发送到 WandB (关键点：使用相同的 global_step 对齐)
                wandb.log({
                    **val_metrics, 
                    "val/best_recon_loss": vae_trainer.best_val_loss,
                    "global_step": global_step
                })
                
                # 保存 Checkpoint
                ckpt_path = os.path.join(save_dir, f"vae_step_{global_step}.pth")
                vae_trainer.save_checkpoint(ckpt_path, is_best=is_best)
                
                print(f"Step {global_step} | Val Recon Loss: {current_val_loss:.6f} | Best: {vae_trainer.best_val_loss:.6f}", flush=True)

            # 4. 普通保存逻辑（如果没到验证步但也到了保存步）
            elif global_step % save_every == 0:
                normal_ckpt_path = os.path.join(save_dir, f"vae_step_{global_step}.pth")
                vae_trainer.save_checkpoint(normal_ckpt_path, is_best=False)

            if global_step % 100 == 0:
                print(f"Step {global_step} | Train Loss: {loss_dict['recon_loss']:.6f}", flush=True)

        # 每个 Epoch 结束强制运行一次验证并保存
        print(f"--- 🏁 Epoch {epoch} End: Final Validation... ---", flush=True)
        val_metrics = vae_trainer.val_step(val_data_loader)
        is_best = val_metrics["val/recon_loss"] < vae_trainer.best_val_loss
        if is_best:
            vae_trainer.best_val_loss = val_metrics["val/recon_loss"]
        
        epoch_ckpt_path = os.path.join(save_dir, f"vae_epoch_{epoch}_end.pth")
        vae_trainer.save_checkpoint(epoch_ckpt_path, is_best=is_best)
    
    print("--- ✅ Training Complete! ---", flush=True)


if __name__ == "__main__":
    main()