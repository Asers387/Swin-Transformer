#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00
#SBATCH --job-name=vhr-mim
#SBATCH --output=%x-%j.out

cd ~/projects/Swin-Transformer
docker build -t vhr-mim .

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -e WANDB_API_KEY=$WANDB_API_KEY --rm --ipc host \
	--mount type=bind,source=/home/daduc/projects/Swin-Transformer,target=/code \
	--mount type=bind,source=/home/daduc/data,target=/data \
	--mount type=bind,source=/dev/shm,target=/dev/shm \
	vhr-mim torchrun sweep_simmim_pt_run.py \
		--data-path /data/vhr-silva \
		--split-path splits/forests_snowless/vhr-silva \
                --sweep-id mlvyc2ob
