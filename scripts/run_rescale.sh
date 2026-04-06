#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem=24GB
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=recale-eor
#SBATCH --cpus-per-task=4
#SBATCH --output=/lustre/aoc/projects/hera/Validation/validation-sim/logs/rescale-eor-%J.out

source ~/miniconda3/bin/activate
conda activate h6c

time python rescale_eor.py
