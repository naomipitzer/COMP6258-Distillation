#!/bin/bash -l
#SBATCH --account=[account, e.g "lyceum"]
#SBATCH -p [account]
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[username]@soton.ac.uk
#SBATCH --time=10:00:00

module load conda/py3-latest
conda activate dpdl

python buffer.py --dataset=Tiny --model=ConvNetD4 --train_epochs=102 --num_experts=1 --buffer_path=results_10_F --data_path=/scratch/np5n22/tiny-imagenet-200 --save_interval 1 --lr_teacher=1e-2


# NOTE: Accuracy is as described in the paper
