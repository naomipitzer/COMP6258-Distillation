#!/usr/bin/env bash
set -e

# common hyperparams
MODEL=resnet18
BATCH=64
EPOCHS=100
LR=1e-3
DEVICE=cpu   # or cuda


ORIG=./distilled/Tiny_ConvNet_IPC1.pt
OUT=swap_results_IPC1.txt
> "$OUT"

for k in 1 2 5 10 20 50 100 150 200; do
  SWAP=./distilled/Tiny_ConvNet_IPC1_swap${k}.pt

  # 1) swap labels
  ./swap_labels.py --input "$ORIG" --swap "$k" --output "$SWAP"

  # 2) train student
  PYTHONPATH=./no_distillation ./venv/bin/python train_student.py \
    --distilled "$SWAP" \
    --model     $MODEL \
    --batch     $BATCH \
    --epochs    $EPOCHS \
    --lr        $LR \
    --device    $DEVICE

  # 3) eval and append
  PYTHONPATH=./no_distillation ./venv/bin/python evaluate_student.py \
    --student student.pth \
    --model   $MODEL \
    --batch   $BATCH \
    --device  $DEVICE \
    >> "$OUT"

  echo "IPC=1 swap $k done"
done

echo
echo "Results for IPC=1"
column -t "$OUT"
echo


ORIG=./distilled/Tiny_ConvNet_IPC10.pt
OUT=swap_results_IPC10.txt
> "$OUT"

for k in 1 2 5 10 20 50 100 150 200; do
  SWAP=./distilled/Tiny_ConvNet_IPC10_swap${k}.pt

  ./swap_labels.py --input "$ORIG" --swap "$k" --output "$SWAP"

  PYTHONPATH=./no_distillation ./venv/bin/python train_student.py \
    --distilled "$SWAP" \
    --model     $MODEL \
    --batch     $BATCH \
    --epochs    $EPOCHS \
    --lr        $LR \
    --device    $DEVICE

  PYTHONPATH=./no_distillation ./venv/bin/python evaluate_student.py \
    --student student.pth \
    --model   $MODEL \
    --batch   $BATCH \
    --device  $DEVICE \
    >> "$OUT"

  echo "IPC=10 swap $k done"
done

echo
echo "Results for IPC=10"
column -t "$OUT"


ORIG=./distilled/Tiny_ConvNet_IPC50.pt
OUT=swap_results_IPC50.txt
> "$OUT"

for k in 1 2 5 10 20 50 100 150 200; do
  SWAP=./distilled/Tiny_ConvNet_IPC50_swap${k}.pt

  ./swap_labels.py --input "$ORIG" --swap "$k" --output "$SWAP"

  PYTHONPATH=./no_distillation ./venv/bin/python train_student.py \
    --distilled "$SWAP" \
    --model     $MODEL \
    --batch     $BATCH \
    --epochs    $EPOCHS \
    --lr        $LR \
    --device    $DEVICE

  PYTHONPATH=./no_distillation ./venv/bin/python evaluate_student.py \
    --student student.pth \
    --model   $MODEL \
    --batch   $BATCH \
    --device  $DEVICE \
    >> "$OUT"

  echo "IPC=50 swap $k done"
done

echo
echo "Results for IPC=50"
column -t "$OUT"
