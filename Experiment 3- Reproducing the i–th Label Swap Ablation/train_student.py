#!/usr/bin/env python3
import torch, argparse
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distilled',   required=True,
                        help='.pt file with (images, soft_labels)')
    parser.add_argument('--model',       default='resnet18',
                        choices=['resnet18','resnet50'])
    parser.add_argument('--batch',       type=int, default=32)
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--device',      default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    images, soft_labels = torch.load(args.distilled, map_location='cpu')
    ds = TensorDataset(images, soft_labels)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # choose student
    if args.model=='resnet18':
        net = models.resnet18(num_classes=soft_labels.size(1))
    else:
        net = models.resnet50(num_classes=soft_labels.size(1))
    net = net.to(device)

    # loss = KLDiv( student_logits, soft_labels ) with temperature=1
    criterion = nn.KLDivLoss(reduction='batchmean')
    opt       = optim.Adam(net.parameters(), lr=args.lr)

    for e in range(1, args.epochs+1):
        net.train()
        total_loss = 0.0
        for x, y_soft in loader:
            x = x.to(device)
            y_soft = y_soft.to(device)

            logits = net(x)
            log_p  = nn.functional.log_softmax(logits, 1)
            loss   = criterion(log_p, y_soft)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()*x.size(0)

        print(f"Epoch {e:02d}/{args.epochs} — avg KL loss: {total_loss/len(ds):.4f}")

    torch.save(net.state_dict(), 'student.pth')
    print("student model saved to student.pth")

if __name__=='__main__':
    main()
