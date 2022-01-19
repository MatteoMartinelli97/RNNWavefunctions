#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=jobname
#SBATCH --output=Results/output_name.out
#SBATCH --gres=gpu:1      		    # Number of GPUs (per node)
#SBATCH --cpus-per-task=1                   # Number of CPUs
#SBATCH --mem=20G               	    # memory (per node)
#SBATCH --partition=gpu   		    # Using only a GPU


export PATH=/pkgs/anaconda3/bin:$PATH
./pkgs/anaconda3/etc/profile.d/conda.sh
module load tensorflow-gpu-36

python run_job.py
