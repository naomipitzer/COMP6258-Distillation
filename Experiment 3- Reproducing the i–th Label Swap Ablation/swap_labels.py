#!/usr/bin/env python
import torch, argparse

def swap_labels(labels, swap_idx):
    N,C = labels.shape
    out = labels.clone()
    vals, idxs = labels.sort(dim=1, descending=True)
    s = swap_idx - 1
    pos_k    = idxs[:, s]    # the i-th largest
    pos_last = idxs[:, -1]   # the smallest
    for i in range(N):
        a,b = pos_k[i].item(), pos_last[i].item()
        out[i,a], out[i,b] = out[i,b], out[i,a]
    return out

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',  type=str, required=True)
    p.add_argument('--swap',   type=int, required=True, help='i-th label to swap')
    p.add_argument('--output', type=str, required=True)
    args = p.parse_args()

    imgs, labs = torch.load(args.input)
    labs_swapped = swap_labels(labs, args.swap)
    torch.save((imgs, labs_swapped), args.output)
    print(f"→ saved swapped labels to {args.output}")
