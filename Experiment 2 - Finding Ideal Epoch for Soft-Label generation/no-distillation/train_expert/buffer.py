import os, sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.append('../')
sys.path.append('../softlabel/')
from softlabel.utils import get_dataset, get_network, get_daparam, TensorDataset, epoch, ParamDiffAug
import copy
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    # Parse and set up
    args.dsa = True if args.dsa == 'True' else False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dsa_param = ParamDiffAug()
    args.decoder = None

    # Load data
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = \
        get_dataset(args.dataset, args.data_path, args.batch_real, args=args)

    print('Hyper-parameters: \n', args.__dict__)

    # Set up save directory
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet" or args.dataset == "ImageNet64":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset == "Tiny" and args.subset != "imagenette":
        save_dir = os.path.join(save_dir, args.subset)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # Build real dataset tensors
    images_all, labels_all = [], []
    indices_class = [[] for _ in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        img, lbl = dst_train[i]
        images_all.append(torch.unsqueeze(img, dim=0))
        labels_all.append(class_map[int(lbl)])
    for idx, lab in enumerate(labels_all):
        indices_class[lab].append(idx)
    images_all = torch.cat(images_all, dim=0).to('cpu')
    labels_all = torch.tensor(labels_all, dtype=torch.long, device='cpu')
    for c in range(num_classes):
        print(f'class c = {c}: {len(indices_class[c])} real images')
    for ch in range(channel):
        print(f'real images channel {ch}, mean = {images_all[:,ch].mean():.4f}, std = {images_all[:,ch].std():.4f}')

    # Training setup
    criterion = nn.CrossEntropyLoss().to(args.device)
    dst_train_tensor = TensorDataset(copy.deepcopy(images_all), copy.deepcopy(labels_all))
    trainloader = torch.utils.data.DataLoader(dst_train_tensor, batch_size=args.batch_train, shuffle=True, num_workers=0)
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'
    print('DC augmentation parameters: \n', args.dc_aug_param)

    # Loop over experts
    for it in range(args.num_experts):
        teacher_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        teacher_net.train()
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2)

        # Record initial parameters
        timestamps = [[p.detach().cpu() for p in teacher_net.parameters()]]

        # Learning rate schedule
        lr_schedule = [args.train_epochs // 2 + 1]

        # Epoch loop
        for e in range(args.train_epochs):
            train_loss, train_acc = epoch('train', trainloader, teacher_net, teacher_optim, criterion, args, aug=True)
            if e == args.train_epochs - 1 or e % 1 == 0:
                test_loss, test_acc = epoch('test', testloader, teacher_net, None, criterion, args, aug=False)
                print(f'Itr: {it}\tEpoch: {e}\tTrain Acc: {train_acc:.5f}\tTest Acc: {test_acc:.5f}')
            print(datetime.now(), f'Itr: {it}\tEpoch: {e}\tTrain Acc: {train_acc:.5f}')

            # Append parameters after this epoch
            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            # Save replay buffer for this epoch
            save_path = os.path.join(save_dir, f'replay_buffer_{it}_epoch_{e:02d}.pt')
            print(f'Saving {save_path}')
            torch.save(timestamps.copy(), save_path)

            # Learning rate decay if needed
            if e in lr_schedule and args.decay:
                new_lr = args.lr_teacher * 0.1
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=new_lr, momentum=args.mom, weight_decay=args.l2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Expert Models')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--subset', type=str, default='imagenette')
    parser.add_argument('--model', type=str, default='ConvNet')
    parser.add_argument('--num_experts', type=int, default=1)
    parser.add_argument('--lr_teacher', type=float, default=0.01)
    parser.add_argument('--teacher_label', action='store_true', default=False)
    parser.add_argument('--selection_strategy', type=str, default='random')
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--dsa', type=str, default='True', choices=['True','False'])
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--buffer_path', type=str, default='./buffers')
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--save_interval', type=int, default=1)
    args = parser.parse_args()
    main(args)
