#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/csedu-scratch/project/jlintelo/stargan/slurm_logs/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/jlintelo/stargan/slurm_logs/%j-%x.err
#SBATCH --job-name=Save_Images

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/jlintelo/venv/bin/activate

python /ceph/csedu-scratch/project/jlintelo/stargan/main.py \
    --mode save_real_images

deactivate