#!/bin/bash

#SBATCH --partition=A40short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=1
#SBATCH -e out/%j.err
#SBATCH -o out/%j.out

module load Miniforge3/24.1.2-0 CUDA/12.4.0
source ~/.bashrc
conda activate Forensics

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src:$(pwd)/src/freqdect

python -O -m freqdect.train_classifier \
  --dataset-root-dir /home/npillath/datasets/image \
  --dataset-positive afhqv2_11k_512 ffhq_11k_512 \
  --dataset-negative afhqv2_stylegan3_11k_512 ffhq_stylegan3_11k_512 \
  --model cnn --features raw

echo "SBATCH completed"
