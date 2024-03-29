#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=31G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/csedu-scratch/project/xxx/handcrafted_weight_sponging/slurm_logs/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/xxx/handcrafted_weight_sponging/slurm_logs/%j-%x.err
#SBATCH --job-name=HWSponge

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/xxx/venv/bin/activate

python /ceph/csedu-scratch/project/xxx/handcrafted_weight_sponging/main.py \
    --model="VGG16" --dataset="MNIST" --max_epoch=20 --batch_size=512 \
    --load --threshold=0.05 --learning_rate=0.01
    
deactivate