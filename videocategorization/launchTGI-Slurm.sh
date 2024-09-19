#!/bin/bash
#SBATCH --job-name=tgi-tests
#SBATCH --partition hopper-prod
#SBATCH --gpus=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH -o slurm/logs/%x_%j.out
#SBATCH --qos=high

export HF_TOKEN=XXXXX
export PORT=1456
srun --container-image='ghcr.io#huggingface/text-generation-inference' \
     --container-env=HUGGING_FACE_HUB_TOKEN,PORT \
     --container-mounts="/scratch:/data" \
     --container-workdir='/usr/src' \
     --no-container-mount-home \
     --qos normal \
     --gpus=8 \
     /usr/local/bin/text-generation-launcher --model-id meta-llama/Meta-Llama-3.1-70B-Instruct