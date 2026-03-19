#!/bin/bash

#SBATCH --job-name=vae_eval_final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lizijia/Gen-JEPA/vae/logs/%j_vae_final.out
#SBATCH --error=/home/lizijia/Gen-JEPA/vae/logs/%j_vae_final.err

# === 1. 激活环境 ===
source "/home/lizijia/miniconda3/etc/profile.d/conda.sh"
conda activate jepa-wms


# === 3. 清理路径 ===
# 彻底清空 PYTHONPATH，防止外部变量干扰 Python 脚本内部的路径逻辑
unset PYTHONPATH

# === 4. 运行 ===
echo "🚀 Job started..."
cd /home/lizijia/Gen-JEPA/vae/try

unset SSL_CERT_FILE

python /home/lizijia/Gen-JEPA/vae/try/inference_intermediate_rae.py

echo "✅ Job finished."