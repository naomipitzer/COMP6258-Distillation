#!/usr/bin/env python3
import os, csv, re

STUDENT_SCRIPT = "softlabel/nodistill.py"
EXPERT_DIR     = "train_expert/experts20/Tiny/ConvNet"
DATA_PATH      = "data/tiny-imagenet-200"
LOG_CSV        = "student_accuracy_log.csv"

# write header
with open(LOG_CSV, "w", newline="") as f:
    csv.writer(f).writerow(["Expert_Epoch","Mean_Test_Acc","Std_Test_Acc"])

for epoch in range(50):
    print(f"\n→ Running student for expert epoch {epoch} ←\n")

    cmd = (
        f"python {STUDENT_SCRIPT} "
        f"--dataset=Tiny "
        f"--data_path={DATA_PATH} "
        f"--batch_real=256 "
        f"--expert_path={EXPERT_DIR} "
        f"--teacher_label "
        f"--teacher_model=ConvNet "
        f"--student_model=ConvNet "
        f"--ipc=1 "
        f"--max_expert_epoch={epoch} "
        f"--epoch_eval_train=1000 "
        f"--optimizer=SGD "
        f"--lr_net=0.01 "
        f"--num_eval=5 "
        f"--batch_train=256 "
        f"--dsa=True "
        f"--dsa_strategy=color_crop_cutout_flip_scale_rotate "
        f"--temp=1.0"
    )

    tmp_out = f"tmp_out_epoch_{epoch}.log"
    os.system(f"{cmd} > {tmp_out} 2>&1")

    mean_acc = std_acc = None
    with open(tmp_out) as f:
        for L in f:
            m = re.search(r"mean\s*=\s*([\d\.]+)\s*std\s*=\s*([\d\.]+)", L)
            if m:
                mean_acc = float(m.group(1))
                std_acc  = float(m.group(2))

    print(f"→ Epoch {epoch} → mean={mean_acc}  std={std_acc}")

    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([epoch, mean_acc, std_acc])
