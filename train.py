import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import os

class TinyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # single scalar output
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# 5) Putting it all together in a training loop
# ------------------------------------------------------------------
X_list = []
y_list = []

saved_boards_dir = "./saved_boards"  

for filename in os.listdir(saved_boards_dir):
    filepath = os.path.join(saved_boards_dir, filename)
    if not filename.endswith(".npz"):
        continue

    data = np.load(filepath)
    x = data['x'] # shape (6, H, W)
    label = data['y']  # shape (1,)

    # Convert shape (6, H, W) to either (6,H,W) or flatten it:
    # For example, flatten:
    x = x.flatten()

    X_list.append(x)
    y_list.append(label)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

if len(X) == 0:
    print("No data was generated. Please implement 'board_generator_fn'.")

# X shape -> (N, input_dim)
# y shape -> (N,)
N, input_dim = X.shape

# Build a small MLP
model = TinyNet(input_dim, hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Move data to PyTorch Tensors
X_torch = torch.from_numpy(X)  # shape (N, input_dim)
y_torch = torch.from_numpy(y).unsqueeze(1)  # shape (N, 1)

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    optimizer.zero_grad()

    preds = model(X_torch)        # (N, 1)
    loss = loss_fn(preds, y_torch)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss={loss.item():.4f}")

# Done! Now 'model' approximates the evaluation function.
