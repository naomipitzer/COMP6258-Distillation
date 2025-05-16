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


#CIFAR-10
echo "CIFAR10 IPC 1"
python train_student.py --dataset=CIFAR10 --data_path=cifar10 --expert_path=nozcacif10model.pt --ipc=1 --use_soft_labels --student_model=ConvNet --student_model=ConvNet --expert_epoch=20

echo "CIFAR10 IPC 10"
python train_student.py --dataset=CIFAR10 --data_path=cifar10 --expert_path=nozcacif10model.pt --ipc=10 --use_soft_labels --student_model=ConvNet --student_model=ConvNet --expert_epoch=44

echo "CIFAR 10 IPC 50"
python train_student.py --dataset=CIFAR10 --data_path=cifar10 --expert_path=nozcacif10model.pt --ipc=50 --use_soft_labels --student_model=ConvNet --student_model=ConvNet --expert_epoch=128


#CIFAR-100
echo "CIFAR 100 IPC 1"
python train_student.py --dataset=CIFAR100 --data_path=cifar100 --expert_path=cif100nozca.pt --ipc=1 --use_soft_labels --student_model=ConvNet --student_model=ConvNet --expert_epoch=13

echo "CIFAR100 IPC 10"
python train_student.py --dataset=CIFAR100 --data_path=cifar100 --expert_path=cif100nozca.pt --ipc=10 --use_soft_labels --student_model=ConvNet --student_model=ConvNet --expert_epoch=36

echo "CIFAR100 IPC 50"
python train_student.py --dataset=CIFAR100 --data_path=cifar100 --expert_path=cif100nozca.pt --ipc=50 --use_soft_labels --student_model=ConvNet --student_model=ConvNet --expert_epoch=125

echo "CIFAR100 IPC 100"
python train_student.py --dataset=CIFAR100 --data_path=cifar100 --expert_path=cif100nozca.pt --ipc=100 --use_soft_labels --student_model=ConvNet --student_model=ConvNet --expert_epoch=125


# NOTE: We do not specify learning rate in this file, as the default already applies to the CIFAR datasets
# cifar100 and cifar10 files must be in the current directory