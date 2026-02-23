#!/bin/bash

#SBATCH --job-name=pusht_preproc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32            # 核心改动：给足 CPU 动力
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1                  # 单卡够用了
#SBATCH --mem=64G                     
#SBATCH --time=04:00:00
#SBATCH --output=/home/lizijia/Gen-JEPA/data_processing/logs/%j_preproc.out
#SBATCH --error=/home/lizijia/Gen-JEPA/data_processing/logs/%j_preproc.err

# 2. 环境激活
source "/home/lizijia/miniconda3/etc/profile.d/conda.sh"
conda activate jepa-wms

# 3. 设置项目路径
export PYTHONPATH=$PYTHONPATH:/home/lizijia/Gen-JEPA/data_processing

# 4. 执行预处理
cd /home/lizijia/Gen-JEPA/data_processing
python preprocess_offline.py