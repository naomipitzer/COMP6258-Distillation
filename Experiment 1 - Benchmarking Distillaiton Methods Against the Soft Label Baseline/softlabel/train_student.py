# Written by us, with some utility functions adapted from https://github.com/sunnytqin/no-distillation/tree/main


# Import libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from tqdm import tqdm
import os
import datetime

from utils import get_dataset, get_network
from reparam_module import ReparamModule


# Logger
def setup_logger(name):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"logs/{name}_{timestamp}.log"
    log = open(path, "w")

    def log_print(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()
    return log_print


# Sample images by class based on Image per class (IPC) value - using default selection (Random) as was documented in the paper
def sample_images_by_class(dataset, ipc, num_classes):
    images = []
    indices_by_class = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        indices_by_class[label].append(idx)
    for c in range(num_classes):
        idxs = random.sample(indices_by_class[c], ipc)
        subset = torch.utils.data.Subset(dataset, idxs)
        loader = DataLoader(subset, batch_size=ipc)
        for batch in loader:
            images.append(batch[0]) 
            break
    return torch.cat(images, dim=0)


# Evaluates the  model on the test/validation set
# Compares hard label to ground truth to compute accuracy
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100. * correct / total


# Training function - trains the student model and calculates test accuracy at every epoch
def train_student(student, trainloader, testloader, device, epochs, lr, log_print):
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) # Momentum, weight decay taken from utils.py
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 4, gamma=0.3) # Scheduler taken from utils.py
    criterion = nn.CrossEntropyLoss() # Loss function taken from utils.py
    
    best = 0.0
    for epoch in tqdm(range(epochs)):
        student.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device) # Loads image and expert's soft labels
            loss = criterion(student(x), y) # Calculates loss using student-generated soft labels and expert-generated soft labels
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(student, testloader, device)
        log_print(f"Epoch {epoch+1}/{epochs} - Test Acc: {acc:.2f}%")
        best = max(best, acc) # stores best accuracy throughout training
        scheduler.step()

    log_print(f"Best Test Accuracy: {best:.2f}%")
    final_acc = evaluate(student, testloader, device)
    log_print(f"\n Test Accuracy after training: {final_acc:.2f}%")


# Loads model parameters from file (given specific expert epoch)
def load_flat_params(path, epoch, log_print):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, list):
        expert_0 = checkpoint[0]
        if epoch < 0:
            epoch = len(expert_0) - 1
        log_print(f"Expert 0, loading epoch {epoch}")
        param_list = expert_0[epoch]
        if isinstance(param_list, list):
            flat = torch.cat([p.flatten() for p in param_list]).detach()
            return flat
        else:
            raise TypeError("Expected a list of tensors for expert epoch entry.")
    else:
        raise TypeError("Checkpoint is not a list of experts.")

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True) # Either CIFAR10, CIFAR100, or Tiny
    parser.add_argument('--data_path', type=str, required=True) # Where the dataset is stored
    parser.add_argument('--expert_path', type=str, required=True) # Where the expert parameters are stored
    parser.add_argument('--expert_epoch', type=int, default=-1) # What expert epoch to use for this run - (e.g, CIFAR10 IPC1 = expert epoch 20)
    parser.add_argument('--ipc', type=int, default=10) # Images per class
    parser.add_argument('--use_soft_labels', action='store_true') # Train with soft labels, true by default
    parser.add_argument('--student_model', type=str, default='ConvNet') # ConvNetD3, ConvNetD4, ConvNet
    parser.add_argument('--teacher_model', type=str, default='ConvNet') # ConvNetD3, ConvNetD4, ConvNet
    parser.add_argument('--epochs', type=int, default=300) # Epochs to train student for - we found that 300 was enough for models to converge
    parser.add_argument('--lr', type=float, default=0.01) # Learning rate to train student
    parser.add_argument('--zca', action='store_true') # Only used for get_datast() function, always False in our runs
    parser.add_argument('--temperature', type=float, default=1.0) # Temperature scaling for soft labels
    args = parser.parse_args()

    # Set seed (we experiment with multiple other seeds and calculate mean/standard deviation of runs)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set up - logger, device (for GPU support), dataset
    log_print = setup_logger(f"{args.dataset}_{args.student_model}_IPC{args.ipc}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}")

    log_print("Loading dataset...")
    ch, im_size, n_cls, _, _, _, dst_train, dst_test, testloader, *_ = get_dataset(
        args.dataset, args.data_path, 128, args=args
    )

    # Loads images per class
    log_print(f"Sampling {args.ipc} images/class...")
    image_syn = sample_images_by_class(dst_train, args.ipc, n_cls)

    log_print("Loading expert (flat param) model...")
    teacher = ReparamModule(get_network(args.teacher_model, ch, n_cls, im_size, dist=False)).to(device)
    flat_param = load_flat_params(args.expert_path, args.expert_epoch, log_print)


    # Using code adapted from https://github.com/sunnytqin/no-distillation/blob/main/softlabel/nodistill.py - Generate soft labels for all synthetic images
    if args.use_soft_labels:
        log_print("Generating soft labels with custom logic...")

        args.device = device
        args.temp = args.temperature
        args.max_expert_epoch = args.expert_epoch

        channel, num_classes = ch, n_cls
        expert_files = [args.expert_path]

        def assign_softlabels():
            file_idx = 0
            expert_idx = 0
            label_net = get_network(args.teacher_model, channel, num_classes, im_size, dist=False).to(args.device)

            label_syn_ensemble = []
            while True:
                if file_idx >= len(expert_files):
                    raise RuntimeError("No more expert files to load.")
                print("loading file {}".format(expert_files[file_idx]))
                buffer = torch.load(expert_files[file_idx])
                expert_trajectory = buffer[expert_idx]
                target_params = expert_trajectory[args.max_expert_epoch]

                if isinstance(target_params, list):
                    label_net = ReparamModule(label_net)
                    target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0).detach().requires_grad_(False)
                else:
                    label_net.load_state_dict(target_params['model'])
                label_net.eval()

                batch_labels = []
                SOFT_INIT_BATCH_SIZE = 100
                if image_syn.shape[0] > SOFT_INIT_BATCH_SIZE:
                    for indices in torch.split(torch.arange(image_syn.shape[0]), SOFT_INIT_BATCH_SIZE):
                        if isinstance(label_net, ReparamModule):
                            batch_labels.append(label_net(image_syn[indices].detach().to(args.device), flat_param=target_params))
                        else:
                            batch_labels.append(label_net(image_syn[indices].detach().to(args.device)).detach().cpu())
                    label_syn = torch.cat(batch_labels, dim=0)
                else:
                    if isinstance(label_net, ReparamModule):
                        label_syn = label_net(image_syn.detach().to(args.device), flat_param=target_params)
                    else:
                        label_syn = label_net(image_syn.detach().to(args.device))

                label_syn = torch.nn.functional.softmax(label_syn * args.temp, dim=-1)
                break  # no ensemble logic

            return label_syn.detach().requires_grad_(False)

        label_syn = assign_softlabels()
        train_set = TensorDataset(image_syn, label_syn)
    else:
        raise NotImplementedError("Only soft label mode is currently supported.")


    # Create trainloader from image_syn and labels_syn
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Train student
    log_print("Initialising student...")
    student = get_network(args.student_model, ch, n_cls, im_size, dist=False).to(device)
    log_print(f"Training for {args.epochs} epochs...")
    train_student(student, trainloader, testloader, device, args.epochs, args.lr, log_print)


if __name__ == '__main__':
    main()