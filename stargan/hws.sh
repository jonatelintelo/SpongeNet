#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/csedu-scratch/project/jlintelo/stargan/slurm_logs/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/jlintelo/stargan/slurm_logs/%j-%x.err
#SBATCH --job-name=HWS_GAN

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/jlintelo/venv/bin/activate

python /ceph/csedu-scratch/project/jlintelo/stargan/main.py \
    --mode hws --selected_attrs Black_Hair Young \
    --test_attribute Black_Hair
    
deactivate