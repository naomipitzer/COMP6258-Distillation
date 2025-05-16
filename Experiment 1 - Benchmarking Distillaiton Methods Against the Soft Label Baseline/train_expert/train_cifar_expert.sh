#!/bin/bash -l
#SBATCH --account=[account, e.g "lyceum"]
#SBATCH -p [account]
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[username]@soton.ac.uk
#SBATCH --time=2:00:00

module load conda/py3-latest
conda activate dpdl

# CIFAR 10
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=164 --num_experts=1 --buffer_path=results_10_F --data_path=../data --save_interval 1 --lr_teacher=5e-4


# CIFAR 100
python buffer.py --dataset=CIFAR100 --model=ConvNet --train_epochs=125 --num_experts=1 --buffer_path=results_10_F --data_path=../data --save_interval 1 --lr_teacher=1e-2


# NOTE: CIFAR100 reaches expected accuracy, CIFAR10 does not as per what was documented in the paper
# cifar10 and cifar100 must be in the current directory