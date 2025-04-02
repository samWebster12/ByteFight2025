import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os


class TinyCNN(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64, input_shape=(6, 32, 32)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Get flattened size by doing a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flattened_dim = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


def load_dataset(save_dir="saved_chunks"):
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


# -------------------------------
# Main training loop
# -------------------------------
X, Y = load_dataset()
X_torch = torch.from_numpy(X).float()  # (N, 6, H, W)
y_torch = torch.from_numpy(Y).float().unsqueeze(1)  # (N, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wrap the model
model = TinyCNN(input_shape=X.shape[1:])
model = nn.DataParallel(model)   # 🔥 Use all available GPUs
model = model.to(device)

# Data
X_torch = torch.from_numpy(X).float().to(device)
y_torch = torch.from_numpy(Y).float().unsqueeze(1).to(device)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()
    preds = model(X_torch)   # Will auto split across GPUs
    loss = loss_fn(preds, y_torch)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
