# No Distillation Reproduction (COMP6258 Project)

This repo reproduces the label swap ablation experiment from:

**"No Distillation: Improving Dataset Distillation by Removing Distillation"**  
https://github.com/sunnytqin/no-distillation/tree/main

We verify their claim that only the top 2–3 softmax entries in the soft labels carry meaningful semantic information by measuring how student accuracy drops when these entries are perturbed.

---

## 📁 Project Structure

.
├── no_distillation/          # Cloned official repo
├── swap_labels.py            # Perturbs soft-label entry at k-th position
├── train_student.py          # Trains student on distilled data
├── evaluate_student.py       # Evaluates student accuracy
├── run_swaps.sh              # Runs ablation experiment (IPC 1, 10, 50)
├── restructure_val.py        # Fixes TinyImageNet val folder (one-time setup)
├── bashScripts.txt           # All training + experiment commands
└── README.md

---

## Setup

1. Clone this repo and set up Python:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision
    ```

2. Download and restructure TinyImageNet:

    ```bash
    python restructure_val.py --path ./tiny-imagenet-200
    ```

3. Train the expert model and generate distilled labels (see `bashScripts.txt`):

    ```bash
    PYTHONPATH=./no_distillation ./venv/bin/python -m train_expert.buffer \
      --dataset Tiny \
      --subset imagenette \
      --model ConvNet \
      --num_experts 1 \
      --train_epochs 20 \
      --batch_train 32 \
      --batch_real 32 \
      --data_path ./tiny-imagenet-200 \
      --buffer_path ./distilled \
      --save_interval 1
    ```

    This will generate:
    - `./distilled/Tiny_ConvNet_IPC1.pt`
    - `./distilled/Tiny_ConvNet_IPC10.pt`
    - `./distilled/Tiny_ConvNet_IPC50.pt`

4. Generate distilled soft labels for each IPC:
# IPC = 1
PYTHONPATH=./no_distillation ./venv/bin/python generate_soft_labels.py \
  --expert_buffer ./buffers/Tiny/ConvNet/replay_buffer_1.pt \
  --epoch 11 \
  --ipc 1 \
  --data_path ./tiny-imagenet-200 \
  --output ./distilled/Tiny_ConvNet_IPC1.pt \
  --device cpu

# IPC = 10
PYTHONPATH=./no_distillation ./venv/bin/python generate_soft_labels.py \
  --expert_buffer ./buffers/Tiny/ConvNet/replay_buffer_1.pt \
  --epoch 11 \
  --ipc 10 \
  --data_path ./tiny-imagenet-200 \
  --output ./distilled/Tiny_ConvNet_IPC10.pt \
  --device cpu

# IPC = 50
PYTHONPATH=./no_distillation ./venv/bin/python generate_soft_labels.py \
  --expert_buffer ./buffers/Tiny/ConvNet/replay_buffer_1.pt \
  --epoch 11 \
  --ipc 50 \
  --data_path ./tiny-imagenet-200 \
  --output ./distilled/Tiny_ConvNet_IPC50.pt \
  --device cpu

5. Train Student on a Distilled Set 
PYTHONPATH=./no_distillation ./venv/bin/python train_student.py \
  --distilled ./distilled/Tiny_ConvNet_IPC1.pt \
  --model convnetd4 \
  --batch 64 \
  --epochs 100 \
  --lr 1e-3 \
  --device cpu

6. Evaluate Student Model (Manual Example)
PYTHONPATH=./no_distillation ./venv/bin/python evaluate_student.py \
  --student student.pth \
  --model convnetd4 \
  --batch 64 \
  --device cpu
---





## Run the Experiment

To reproduce the soft-label swap ablation (Figure 3 in the paper):

```bash
bash run_swaps.sh
