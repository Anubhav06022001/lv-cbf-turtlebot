import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# --- Pathing ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Adjust imports for your project structure
from src.vocbf.vocbf_network import BarrierNet
from src.lvcbf.expectile import expectile_loss
from src.utils.checkpoint_utils import save_checkpoint, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Training Params ---
gamma = 0.5
tau = 0.8
lr = 0.0003
epochs = 45 
d_safe = 0.05
batch_size = 128

# --- Paths ---
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)
data_path = PROJECT_ROOT / "experiments" / "data_collection" / "dataset_vocbf.csv"

# ==========================================
# DATASET CLASS (For CSV State Data)
# ==========================================
class TurtleBotStateDataset(Dataset):
    def __init__(self, csv_file, models_dir):
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        states = df[['x', 'y', 'yaw', 'v', 'w']].values
        dists = df['min_dist'].values
        episodes = df['episode'].values

        # Calculate and save normalization factors
        self.x_mean = np.mean(states, axis=0)
        self.x_std = np.std(states, axis=0) + 1e-8
        np.save(str(models_dir / "tb3_state_mean.npy"), self.x_mean)
        np.save(str(models_dir / "tb3_state_std.npy"), self.x_std)

        norm_states = (states - self.x_mean) / self.x_std

        self.X, self.X_next, self.D = [], [], []

        # Create transitions (xt -> xt+1)
        for i in range(len(df) - 1):
            if episodes[i] == episodes[i+1]:  # Ensure it's the same episode
                self.X.append(norm_states[i])
                self.X_next.append(norm_states[i+1])
                self.D.append(dists[i])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.X_next = torch.tensor(np.array(self.X_next), dtype=torch.float32)
        self.D = torch.tensor(np.array(self.D), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_next[idx], self.D[idx]

# ==========================================
# TRAINING SETUP
# ==========================================
dataset = TurtleBotStateDataset(data_path, models_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

B_net = BarrierNet().to(DEVICE)
optimizer = torch.optim.Adam(B_net.parameters(), lr=lr)

checkpoint_path = models_dir / "tb3_vocbf.pt"

# Optional: Load checkpoint if resuming
start_epoch = load_checkpoint(B_net, optimizer, checkpoint_path, DEVICE)

print(f"Dataset loaded. Total transitions: {len(dataset)}")

# ==========================================
# TRAINING LOOP
# ==========================================
for epoch in range(epochs):
    losses = []
    B_net.train()

    for x, x_next, d in dataloader:
        x, x_next, d = x.to(DEVICE), x.to(DEVICE), d.to(DEVICE)

        ell = d - d_safe

        B = B_net(x).squeeze()
        with torch.no_grad():
            B_next = B_net(x_next).squeeze()

        target = (1 - gamma) * ell + gamma * torch.minimum(ell, B_next)
        diff = target - B
        value_loss = expectile_loss(diff, tau)

        # Boundary constraints
        unsafe = (d < d_safe)
        safe   = (d > 2.0) 
        
        anchor = 0
        if unsafe.any(): anchor += 0.05 * (B[unsafe] + 1).pow(2).mean()
        if safe.any(): anchor += 0.01 * (B[safe] - 1).pow(2).mean()

        loss = value_loss + anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d} | Loss: {np.mean(losses):.4f}")

    if (epoch + 1) % 5 == 0:
        save_checkpoint(B_net, optimizer, epoch, checkpoint_path)
        # pass # Enable if you have the save_checkpoint function implemented

# Save final model
final_model_path = models_dir / "tb3_vocbf.pt"
torch.save({'model_state_dict': B_net.state_dict()}, final_model_path)
print(f"Saved Baseline VOCBF model to {final_model_path}")