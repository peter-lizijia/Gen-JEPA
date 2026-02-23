#!/bin/bash

#SBATCH --job-name=split_pusht_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16              # 切分脚本主要是 CPU 密集型（Tensor 计算和文件移动）
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:0                   # 处理数据不需要 GPU，设为 0 节省资源
#SBATCH --mem=128G                      # 必须足够大以同时加载多个大 Tensor
#SBATCH --time=02:00:00                # 处理 1000 个视频通常 1 小时内搞定
#SBATCH --output=/home/lizijia/Gen-JEPA/data_processing/logs/%j_data_split.out
#SBATCH --error=/home/lizijia/Gen-JEPA/data_processing/logs/%j_data_split.err

# 2. 激活 Conda 环境
source "/home/lizijia/miniconda3/etc/profile.d/conda.sh"
conda activate jepa-wms

# 3. 设置路径
# 确保包含脚本所在的目录
export PYTHONPATH=/home/lizijia/Gen-JEPA:/home/lizijia/Gen-JEPA/data_processing:$PYTHONPATH

# 4. 运行切分脚本
cd /home/lizijia/Gen-JEPA/data_processing

echo "开始执行数据集物理切分..."
python split_test_set.py
echo "数据集切分任务已完成。"