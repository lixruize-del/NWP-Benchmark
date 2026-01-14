#!/bin/bash
#SBATCH -J AIFS_Prod
#SBATCH -p vip_gpu_ailab
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -A ai4earth
#SBATCH --qos=gpugpu
#SBATCH -c 4
#SBATCH -o logs/aifs/%j.out
#SBATCH -e logs/aifs/%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Date:   $(date)"
echo "Dir:    $(pwd)"
echo "========================================"

# 激活 Conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aifs_env

# [DEPLOYMENT FIX] 强制加载 Conda 环境的 C++ 标准库
# 解决 ImportError: /usr/lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found
export LD_PRELOAD=/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/libstdc++.so.6

# [DEPLOYMENT FIX] 注入 NVIDIA CUDA 12 运行时库路径
# 解决 ImportError: libcupti.so.12 / libcublas.so.12 not found
export LD_LIBRARY_PATH=/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/cublas/lib:/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/curand/lib:/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/bingxing2/ailab/scxlab0094/.conda/envs/aifs_v2/lib/python3.12/site-packages/nvidia/cufft/lib:\$LD_LIBRARY_PATH

# 硬件检查
echo "[Hardware Check]"
nvidia-smi
echo ""

# 运行推理
echo "[Starting Inference]"
python -u src/aifs/inference.py --lead-time 24

echo "Job Finished at $(date)"