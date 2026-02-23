#!/bin/bash

#SBATCH --job-name=gen_jepa_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16             # 对应 num_workers: 8，通常设为 2x
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1 
#SBATCH --mem=128G 
#SBATCH --time=72:00:00                
#SBATCH --output=/home/lizijia/Gen-JEPA/vae/logs/%j_gen_jepa_train_test.out
#SBATCH --error=/home/lizijia/Gen-JEPA/vae/logs/%j_gen_jepa_train_test.err

# 1. 激活环境
source "/home/lizijia/miniconda3/etc/profile.d/conda.sh"
conda activate jepa-wms

# 2. 设置项目根目录和 Python 路径
# 确保项目根目录在 PYTHONPATH 中，这样内部引用才不会报错
export PRJ_ROOT=/home/lizijia/Gen-JEPA
export PYTHONPATH=$PRJ_ROOT:$PRJ_ROOT/data_processing:$PYTHONPATH

# 3. 设置 Hydra 配置文件路径的环境变量 (对应你代码中的 os.environ.get)
export CONFIG_PATH="$PRJ_ROOT/vae/models/configs/kubric"
export CONFIG_NAME="dino_foresight_cfm"

# 4. 运行训练脚本

cd /home/lizijia/Gen-JEPA

torchrun --nproc_per_node=1 --master_port=29508 vae/models/predictor/train.py
# torchrun --nproc_per_node=1 --master_port=29502 /home/lizijia/Gen-JEPA/vae/models/predictor/eval.py
