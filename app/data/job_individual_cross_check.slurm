#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 24
#SBATCH --time=2:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A spinquest_standard

module load miniforge cuda/11.4.2 cudnn/8.2.4.15
module load gcc/11.4.0 lhapdf/6.5.4 
source activate lhapdf-tf-py11
export PYTHONPATH=~/.conda/envs/lhapdf-tf-py11/lib/python3.11/site-packages:$PYTHONPATH

python3 individual_cross-check_script.py