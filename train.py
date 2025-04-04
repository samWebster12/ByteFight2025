import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from utils import TinyCNN

# Load and normalize dataset
def load_dataset(save_dir="data"):
    X_list, Y_list = [], []
    for fname in sorted(os.listdir(save_dir)):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(save_dir, fname))
            X_list.append(data["x"])
            Y_list.append(data["y"])
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")
    return X, Y

X, Y = load_dataset()
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Normalize using train stats
mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
std = np.std(X_train, axis=(0, 2, 3), keepdims=True) + 1e-6
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN(input_shape=X_train.shape[1:]).to(device)
parallel_model = nn.DataParallel(model)

# DataLoaders
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float().unsqueeze(1)),
    batch_size=512, shuffle=True, pin_memory=True, num_workers=4
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float().unsqueeze(1)),
    batch_size=512, pin_memory=True, num_workers=4
)

# Optimizer, loss, scheduler
optimizer = optim.Adam(parallel_model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

# Training loop
train_losses, val_losses = [], []
NUM_EPOCHS = 1000

for epoch in range(1, NUM_EPOCHS + 1):
    parallel_model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = parallel_model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    parallel_model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(parallel_model(xb.to(device)), yb.to(device)).item()
                       for xb, yb in val_loader) / len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Test evaluation
parallel_model.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    Y_test_tensor = torch.from_numpy(Y_test).float().unsqueeze(1).to(device)
    test_preds = parallel_model(X_test_tensor)
    test_loss = loss_fn(test_preds, Y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "cnn_weights/v4.pt")

# Plot training/validation loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot of predictions
plt.scatter(Y_test_tensor.cpu().numpy(), test_preds.cpu().numpy(), alpha=0.3)
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.title("Test Predictions")
plt.grid(True)
plt.show()