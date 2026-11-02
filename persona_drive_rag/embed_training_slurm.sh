#!/bin/bash
#SBATCH --job-name=rag_train
#SBATCH --output=slurm_logs/rag_train_%j.out
#SBATCH --error=slurm_logs/rag_train_%j.err
#SBATCH --time=01:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:A30:4
#SBATCH -A YOUR_SLURM_ACCOUNT_GPU   # #SBATCH lines cannot read env vars; edit or pass -A on the sbatch CLI
#SBATCH --mem=48G

mkdir -p slurm_logs

echo "================================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Start Time   : $(date)"
echo "================================================================"

source ~/.bashrc
conda activate "${PERSONADRIVE_CONDA_ENV:-/path/to/conda/envs/personadrive}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled

echo "Python       : $(which python)"
echo "PyTorch CUDA : $(python -c 'import torch; print(torch.cuda.device_count(), "GPUs")')"

python embed_training.py \
    --jsonl_path triplets/triplets.jsonl \
    --image_emb_path embeddings/image_embeddings.pt \
    --frame_index_path embeddings/frame_ids.json \
    --batch_size 512 \
    --epochs 20 \
    --lr 1e-4

echo "================================================================"
echo "Job completed at: $(date)"
echo "================================================================"