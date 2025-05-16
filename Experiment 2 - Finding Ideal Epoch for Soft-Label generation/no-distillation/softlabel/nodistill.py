#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import time  # 🔍 for timing
from utils import get_dataset, get_network, evaluate_synset
from reparam_module import ReparamModule
from torch.utils.data import DataLoader, Subset

def main(args):
    start_total = time.time()  # 🔍 Start total timer

    # device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    # load dataset
    print("→ Loading dataset...")
    channel, im_size, num_classes, _, _, _, \
      dst_train, dst_test, testloader, _, _, _ = \
      get_dataset(args.dataset, args.data_path,
                  args.batch_real, args=args)

    # load expert buffer
    if args.teacher_label:
        buf = args.expert_path
        if os.path.isdir(buf):
            buf = os.path.join(buf, "replay_buffer_0.pt")
        print("→ Loading expert buffer from:", buf)
        trajectory = torch.load(buf, map_location='cpu')[0]
        target_params = trajectory[args.max_expert_epoch]

        # build a ReparamModule for soft-labels
        label_net   = get_network(
            args.teacher_model, channel, num_classes, im_size, dist=False
        ).to(args.device)
        label_net   = ReparamModule(label_net)
        flat_params = torch.cat([p.reshape(-1) for p in target_params], 0)
        flat_params = flat_params.detach().requires_grad_(False)
        label_net.eval()

    # build synthetic images (ipc per class)
    print("→ Generating synthetic images...")
    image_syn = torch.zeros(
      (num_classes * args.ipc, channel, im_size[0], im_size[1]),
      dtype=torch.float
    )
    for c in range(num_classes):
        idxs = np.random.choice(
            [i for i,(_,lab) in enumerate(dst_train) if lab == c],
            args.ipc, replace=False
        )
        loader = DataLoader(Subset(dst_train, idxs), batch_size=args.ipc)
        image_syn[c*args.ipc:(c+1)*args.ipc] = next(iter(loader))[0]

    # generate labels
    print("→ Generating soft/hard labels...")
    if args.teacher_label:
        with torch.no_grad():
            logits    = label_net(image_syn.to(args.device),
                                  flat_param=flat_params)
            label_syn = F.softmax(logits * args.temp, dim=-1)
    else:
        hard      = np.repeat(np.arange(num_classes), args.ipc)
        label_syn = torch.tensor(hard, dtype=torch.long).to(args.device)

    # evaluate student multiple times
    print("→ Training and evaluating student models...")
    accs = []
    for i in range(args.num_eval):
        print(f"→ Evaluation run {i+1}/{args.num_eval}")
        eval_start = time.time()  # 🔍 start timing for each eval

        student = get_network(
            args.student_model, channel, num_classes, im_size, dist=True
        )
        _, _, test_acc = evaluate_synset(
            i, student, image_syn, label_syn, testloader, args
        )
        accs.append(test_acc)

        eval_end = time.time()  # 🔍 end timing for eval
        print(f"✓ Run {i+1} completed in {eval_end - eval_start:.2f}s with acc: {test_acc:.4f}")

    # final result
    m, s = float(np.mean(accs)), float(np.std(accs))
    print(f"\n Final Result over {len(accs)} runs → mean = {m:.4f}  std = {s:.4f}")

    end_total = time.time()  # 🔍 End total timer
    print(f"⏱️ Total time elapsed: {(end_total - start_total)/60:.2f} minutes")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--dataset',       type=str,   default='Tiny')
    p.add_argument('--data_path',     type=str,   required=True)
    p.add_argument('--batch_real',    type=int,   default=256)
    # expert
    p.add_argument('--expert_path',   type=str,   required=True,
                   help="folder containing replay_buffer_0.pt or the .pt file itself")
    p.add_argument('--teacher_label', action='store_true')
    p.add_argument('--teacher_model', type=str,   default='ConvNet')
    # synthetic
    p.add_argument('--ipc',           type=int,   default=1)
    # student
    p.add_argument('--student_model', type=str,   default='ConvNet')
    p.add_argument('--epoch_eval_train',
                                  type=int,   default=1000)
    p.add_argument('--optimizer',     type=str,   default='SGD')
    p.add_argument('--lr_net',        type=float, default=0.01)
    p.add_argument('--num_eval',      type=int,   default=5)
    p.add_argument('--batch_train',   type=int,   default=256)
    # which expert epoch
    p.add_argument('--max_expert_epoch',
                                  type=int,   default=10)
    # augmentation & temperature (unused except for soft labels)
    p.add_argument('--dsa',           type=str,   default='True')
    p.add_argument('--dsa_strategy',  type=str,
                   default='color_crop_cutout_flip_scale_rotate')
    p.add_argument('--zca',           action='store_true',
                   help="dummy flag for compatibility")
    p.add_argument('--temp',          type=float, default=1.0)

    args = p.parse_args()
    main(args)
