#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student',   type=str, required=True,
                        help="path to student.pth")
    parser.add_argument('--model',     type=str, default='resnet18',
                        choices=['resnet18','resnet50'])
    parser.add_argument('--data_path', type=str, default='./tiny-imagenet-200',
                        help="root of tiny-imagenet-200")
    parser.add_argument('--batch',     type=int, default=32)
    parser.add_argument('--device',    type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.model == 'resnet18':
        net = models.resnet18()
    else:
        net = models.resnet50()
    net.fc = nn.Linear(net.fc.in_features, 200)
    net.load_state_dict(torch.load(args.student, map_location=device))
    net.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    val_root = os.path.join(args.data_path, 'val', 'images')
    valset = datasets.ImageFolder(val_root, transform=transform)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch, shuffle=False, num_workers=4)

    # gets accuracy
    correct = 0
    total   = 0
    with torch.no_grad():
        for x,y in valloader:
            x,y = x.to(device), y.to(device)
            preds = net(x).argmax(1)
            correct += (preds==y).sum().item()
            total   += y.size(0)

    print(f'Student accuracy: {correct/total*100:.2f}%')

if __name__=='__main__':
    main()
