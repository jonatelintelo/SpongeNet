#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/csedu-scratch/project/jlintelo/slurm_logs/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/jlintelo/slurm_logs/%j-%x.err
#SBATCH --job-name=SpoGAN_Eval

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/jlintelo/venv/bin/activate

python /ceph/csedu-scratch/project/jlintelo/stargan/eval.py \
    --mode test --selected_attrs Black_Hair Male Young --c_dim 3 \
    --fake_models 0 1 2 3 4 5

deactivate