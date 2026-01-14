#!/bin/bash
#SBATCH -J Pangu_Infer
#SBATCH -p vip_gpu_ailab
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -A ai4earth
#SBATCH --qos=gpugpu
#SBATCH -c 4
#SBATCH -o logs/pangu/%j.out
#SBATCH -e logs/pangu/%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Date:   $(date)"

# 1. 激活 Pangu 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pangu_env

# 2. 检查 GPU
echo "[Hardware Check]"
nvidia-smi
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"

# 3. 运行推理 (例如预测 72 小时 = 3天)
echo "[Starting Inference]"
python -u src/pangu/inference.py --lead-time 72

echo "Job Finished at \$(date)"