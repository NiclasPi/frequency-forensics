#!/bin/bash

#SBATCH --partition=A40short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -e out/%j.err
#SBATCH -o out/%j.out

module load Miniforge3/24.1.2-0 CUDA/12.4.0
source ~/.bashrc
conda activate Forensics

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

python -m freqdect.prepare_dataset ./data/source --train-size 20000 --val-size 4000 --test-size 6000 --raw --binary-classification --crop

echo "SBATCH completed"
