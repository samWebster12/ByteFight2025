from utils import TinyCNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

def load_dataset(save_dir="data"):
    X_list = []
    Y_list = []

    for fname in sorted(os.listdir(save_dir)):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(save_dir, fname))
            X_list.append(data["x"])
            Y_list.append(data["y"])

    X = np.concatenate(X_list, axis=0)  # (N, 6, H, W)
    Y = np.concatenate(Y_list, axis=0)  # (N,)
    print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")
    return X, Y

# Load data
X, Y = load_dataset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model first (not wrapped in DataParallel)
model = TinyCNN(input_shape=X.shape[1:])
model = model.to(device)

# Wrap with DataParallel for training on multiple GPUs
parallel_model = nn.DataParallel(model)

# Data preparation
X_torch = torch.from_numpy(X).float().to(device)
y_torch = torch.from_numpy(Y).float().unsqueeze(1).to(device)

# Training setup
optimizer = optim.Adam(parallel_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
NUM_EPOCHS = 1000

# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    parallel_model.train()
    optimizer.zero_grad()
    preds = parallel_model(X_torch)
    loss = loss_fn(preds, y_torch)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

# ✅ Save only the base model's weights — no "module." prefixes!
torch.save(model.state_dict(), "tiny_cnn_weights.pt")
