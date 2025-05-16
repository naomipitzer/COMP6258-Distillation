#!/bin/bash -l
#SBATCH --account=lyceum
#SBATCH -p lyceum
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=np5n22@soton.ac.uk
#SBATCH --time=10:00:00

module load conda/py3-latest
conda activate dpdl

echo "Tiny IPC 1"
python train_student.py --dataset=Tiny --data_path=/scratch/np5n22/tiny-imagenet-200 --expert_path=tinynetwork.pt --ipc=1 --use_soft_labels --teacher_model=ConvNetD4 --student_model=ConvNetD4 --expert_epoch=11

echo "Tiny IPC 10"
python train_student.py --dataset=Tiny --data_path=/scratch/np5n22/tiny-imagenet-200 --expert_path=tinynetwork.pt --ipc=10 --use_soft_labels --lr=0.1 --teacher_model=ConvNetD4 --student_model=ConvNetD4 --expert_epoch=50

echo "Tiny IPC 50"
python train_student.py --dataset=Tiny --data_path=/scratch/np5n22/tiny-imagenet-200 --expert_path=tinynetwork.pt --ipc=50 --use_soft_labels --lr=0.1  --teacher_model=ConvNetD4 --student_model=ConvNetD4 --expert_epoch=90

echo "Tiny IPC 100"
python train_student.py --dataset=Tiny --data_path=/scratch/np5n22/tiny-imagenet-200 --expert_path=tinynetwork.pt --ipc=100 --use_soft_labels --lr=0.1 --teacher_model=ConvNetD4 --student_model=ConvNetD4 --expert_epoch=102