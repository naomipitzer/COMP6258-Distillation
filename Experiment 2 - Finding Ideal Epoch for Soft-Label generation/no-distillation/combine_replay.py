import torch
import os

# Directory containing your per-epoch files
checkpoint_dir = 'train_expert/experts20/Tiny/ConvNet'

# Number of epochs you have saved (adjust if needed)
num_epochs = 50

epochs = []
for i in range(num_epochs):
    fname = os.path.join(checkpoint_dir, f'replay_buffer_0_epoch_{i:02d}.pt')
    data = torch.load(fname, map_location='cpu')
    # If each file is a list/tuple, extract the params list
    params = data[0] if isinstance(data, (list, tuple)) else data
    epochs.append(params)

# Save combined trajectory
out_path = os.path.join(checkpoint_dir, 'replay_buffer_0.pt')
torch.save([epochs], out_path)
print(f"Combined {num_epochs} epoch checkpoints into {out_path}")
