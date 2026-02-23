#!/bin/bash

#SBATCH --job-name=vae_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32             # 提升至 16，匹配 num_workers
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1                   # 保持 1 张 GPU
#SBATCH --mem=128G                      # 建议提升到 64G，视频解码缓存比较吃内存
#SBATCH --time=24:00:00                # 建议增加限时，10个epoch可能需要几个小时
#SBATCH --output=/home/lizijia/Gen-JEPA/vae/logs/%j_vae_train.out
#SBATCH --error=/home/lizijia/Gen-JEPA/vae/logs/%j_vae_train.err


# 2. 激活 Conda 环境
source "/home/lizijia/miniconda3/etc/profile.d/conda.sh"
conda activate jepa-wms

# 3. 设置项目路径
# 确保 Python 能找到 app 文件夹
export PYTHONPATH=/home/lizijia/Gen-JEPA:/home/lizijia/Gen-JEPA/data_processing:$PYTHONPATH

# 4. 运行离线测试脚本
# 我们直接运行刚才写的测试脚本
cd /home/lizijia/Gen-JEPA/vae/try
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python online_train_vae.py