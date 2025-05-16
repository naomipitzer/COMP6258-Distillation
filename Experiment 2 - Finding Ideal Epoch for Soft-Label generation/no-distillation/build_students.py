import os, sys, argparse
import torch
import numpy as np
import time
from collections import defaultdict
from torch.utils.data import DataLoader, Subset, TensorDataset


sys.path.insert(0, os.path.join(os.getcwd(), 'softlabel'))
from utils import get_dataset, get_network, evaluate_synset
from nodistill import ReparamModule

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate student models on synthetic soft labels from different expert epochs"
    )
    parser.add_argument('--expert_path', required=True,
                        help="Path to saved trajectory .pt file (contains list of checkpoint params)")
    parser.add_argument('--data_path', required=True,
                        help="Root path to dataset (e.g., data/tiny-imagenet-200)")
    parser.add_argument('--ipc', type=int, default=10,
                        help="Images per class for synthetic set")
    parser.add_argument('--batch_train', type=int, default=16,
                        help="Batch size for student training")
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help="Number of epochs to train each student")
    parser.add_argument('--student_model', type=str, default='ConvNet',
                        help="Model architecture for student")
    parser.add_argument('--teacher_model', type=str, default='ConvNet',
                        help="Model architecture of expert")
    parser.add_argument('--temp', type=float, default=1.0,
                        help="Temperature for soft labels")
    parser.add_argument('--epochs', type=int, nargs='+', default=[0, 10, 20],
                        help="List of expert epochs to evaluate (e.g. 0 10 20)")
    parser.add_argument('--all_epochs', action='store_true', default=False,
                        help="Use all expert epochs in trajectory")
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help="Optimizer for student training: SGD or AdamW")
    parser.add_argument('--lr_net', type=float, default=0.01,
                        help="Learning rate for student training")
    parser.add_argument('--zca', action='store_true', default=False,
                        help="Apply ZCA whitening to data (not used by default)")
    args = parser.parse_args()

    # Choosing device (MPS on Apple silicon, else CUDA/CPU)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] Using device: {device}")

    # Loading dataset
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {args.data_path}")
    print("→ [DEBUG] About to call get_dataset()")
    dataset_info = get_dataset('Tiny', args.data_path, args.ipc, args=args)
    print("→ [DEBUG] get_dataset() returned successfully")
    channel, im_size, num_classes = dataset_info[0], dataset_info[1], dataset_info[2]
    dst_train, dst_test, testloader = dataset_info[6], dataset_info[7], dataset_info[8]

    # Loading expert trajectory
    full_traj = torch.load(args.expert_path, map_location='cpu')[0]
    if args.all_epochs:
        args.epochs = list(range(len(full_traj)))

    # Precomputing class→indices map
    class_to_indices = defaultdict(list)
    for idx, (_, lab) in enumerate(dst_train):
        class_to_indices[lab].append(idx)

    # Generating synthetic images
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating synthetic images (ipc={args.ipc})")
    image_syn = torch.zeros((num_classes * args.ipc, channel, im_size[0], im_size[1]),
                             dtype=torch.float)
    for c in range(num_classes):
        idxs = np.random.choice(class_to_indices[c], args.ipc, replace=False)
        imgs = next(iter(DataLoader(Subset(dst_train, idxs), batch_size=args.ipc)))[0]
        image_syn[c * args.ipc : (c + 1) * args.ipc] = imgs
    image_syn = image_syn.to(device)

    results = []
    total = len(args.epochs)
    for idx, epoch in enumerate(args.epochs, 1):
        start_ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{start_ts}] Starting expert epoch {epoch} ({idx}/{total})")
        target_params = full_traj[epoch]

        # Building label network
        label_net = get_network(args.teacher_model, channel, num_classes, im_size, dist=False).to(device)
        label_net = ReparamModule(label_net)
        # concatenate and move to device
        flat_params = torch.cat([p.reshape(-1) for p in target_params], 0).to(device)
        flat_params = flat_params.detach().requires_grad_(False)
        label_net.eval()

        # Soft-label generation
        with torch.no_grad():
            logits = label_net(image_syn, flat_param=flat_params)
            label_syn = torch.softmax(logits * args.temp, dim=-1)

        # Preparing student args
        ns_args = argparse.Namespace(
            device=device,
            ipc=args.ipc,
            batch_train=args.batch_train,
            epoch_eval_train=args.epoch_eval_train,
            optimizer=args.optimizer,
            lr_net=args.lr_net,
            num_eval=1,
            dsa=False,
            teacher_label=False,
            zca=args.zca,
            temp=args.temp,
            exp_num=idx,
            selection_strategy=None
        )

        # Training & evaluating student
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training student on epoch {epoch}")
        student = get_network(args.student_model, channel, num_classes, im_size, dist=True)
        _, _, test_acc = evaluate_synset(0, student,
                                        image_syn.cpu(), label_syn.cpu(),
                                        testloader, ns_args)
        end_ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{end_ts}] Completed expert epoch {epoch} ({idx}/{total}): test_acc={test_acc:.4f}")
        results.append((epoch, test_acc))

    # Final summary
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Summary of results:")
    for e, acc in results:
        print(f"  Epoch {e}: {acc:.4f}")

if __name__ == '__main__':
    main()
