#!/bin/bash
#SBATCH --job-name=compute_stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00             
#SBATCH --output=/home/lizijia/Gen-JEPA/vae/logs/%j_compute_stats.out
#SBATCH --error=/home/lizijia/Gen-JEPA/vae/logs/%j_compute_stats.err

# 1. 创建必要的目录
mkdir -p /home/lizijia/Gen-JEPA/vae/data
mkdir -p /home/lizijia/Gen-JEPA/vae/logs

# 2. 激活环境
source "/home/lizijia/miniconda3/etc/profile.d/conda.sh"
conda activate jepa-wms

# 3. 设置项目路径 (关键：确保能找到 data_processing 和 models)
export PYTHONPATH=/home/lizijia/Gen-JEPA/vae:/home/lizijia/Gen-JEPA:/home/lizijia/Gen-JEPA/data_processing:$PYTHONPATH

# 4. 运行脚本
# 确保你在脚本里把 save_path 设置为 /home/lizijia/Gen-JEPA/vae/data/push_t_stats.npz
python /home/lizijia/Gen-JEPA/vae/try/get_sta_dino.py
# python /home/lizijia/Gen-JEPA/vae/try/compute_dino_stats.py