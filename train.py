from utils import TinyCNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training set: X_train={X_train.shape}, Y_train={Y_train.shape}")
print(f"Testing set: X_test={X_test.shape}, Y_test={Y_test.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model first (not wrapped in DataParallel)
model = TinyCNN(input_shape=X_train.shape[1:])
model = model.to(device)

# Wrap with DataParallel for training on multiple GPUs
parallel_model = nn.DataParallel(model)

# Data preparation
X_torch = torch.from_numpy(X_train).float().to(device)
y_torch = torch.from_numpy(Y_train).float().unsqueeze(1).to(device)

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

with torch.no_grad():
    parallel_model.eval()
    X_test_torch = torch.from_numpy(X_test).float().to(device)
    y_test_torch = torch.from_numpy(Y_test).float().unsqueeze(1).to(device)
    test_preds = parallel_model(X_test_torch)
    test_loss = loss_fn(test_preds, y_test_torch)
    print(f"Test Loss: {test_loss.item():.4f}")

import matplotlib.pyplot as plt
plt.scatter(y_test_torch.cpu().numpy(), test_preds.cpu().numpy(), alpha=0.3)
plt.xlabel("True Y")
plt.ylabel("Predicted Y")
plt.title("Test Prediction Scatter")
plt.grid(True)
plt.show()


# ✅ Save only the base model's weights — no "module." prefixes!
torch.save(model.state_dict(), "cnn_weights/v4.pt")


# from utils import TinyCNN
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# from torch.utils.data import DataLoader, TensorDataset

# def load_dataset(save_dir="data"):
#     X_list = []
#     Y_list = []

#     for fname in sorted(os.listdir(save_dir)):
#         if fname.endswith(".npz"):
#             data = np.load(os.path.join(save_dir, fname))
#             X_list.append(data["x"])
#             Y_list.append(data["y"])

#     X = np.concatenate(X_list, axis=0)  # (N, 6, H, W)
#     Y = np.concatenate(Y_list, axis=0)  # (N,)
#     print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")
#     return X, Y

# def main():
#     # Load data
#     X, Y = load_dataset()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Initialize model first (not wrapped in DataParallel)
#     model = TinyCNN(input_shape=X.shape[1:])
#     model = model.to(device)

#     # Wrap with DataParallel for training on multiple GPUs
#     parallel_model = nn.DataParallel(model)

#     NUM_EPOCHS = 5000
#     LEARNING_RATE = 1e-3
#     BATCH_SIZE = 512

#     # Data preparation
#     X_torch = torch.from_numpy(X).float()
#     y_torch = torch.from_numpy(Y).float().unsqueeze(1)
#     dataset = TensorDataset(X_torch, y_torch)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


#     # Training setup
#     optimizer = optim.Adam(parallel_model.parameters(), lr=LEARNING_RATE)
#     loss_fn = nn.MSELoss()

#     # Training loop
#     for epoch in range(1, NUM_EPOCHS + 1):
#         parallel_model.train()
#         running_loss = 0.0

#         for batch_X, batch_y in dataloader:
#             batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
#             optimizer.zero_grad()
#             preds = parallel_model(batch_X)
#             loss = loss_fn(preds, batch_y)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         if epoch % 100 == 0:
#             avg_loss = running_loss / len(dataloader)
#             print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")


#     # ✅ Save only the base model's weights — no "module." prefixes!
#     torch.save(model.state_dict(), "cnn_weights/v4.pt")

# if __name__ == "__main__":
#     main()