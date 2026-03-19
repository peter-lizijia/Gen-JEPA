#!/bin/bash

#SBATCH --job-name=wind_demo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/lizijia/Gen-JEPA/stochastic_pusht/gym-pusht/logs/%j_wind_demo.out
#SBATCH --error=/home/lizijia/Gen-JEPA/stochastic_pusht/gym-pusht/logs/%j_wind_demo.err

# 1. 激活 Conda 环境
source "/home/lizijia/miniconda3/etc/profile.d/conda.sh"
conda activate jepa-wms

# 2. 设置项目路径
export PYTHONPATH=/home/lizijia/Gen-JEPA:/home/lizijia/Gen-JEPA/stochastic_pusht/gym-pusht:$PYTHONPATH

# 3. 进入工程目录并创建日志目录
cd /home/lizijia/Gen-JEPA/stochastic_pusht/gym-pusht
mkdir -p logs

# 4. 运行视频采样脚本
python make_wind_demo.py
