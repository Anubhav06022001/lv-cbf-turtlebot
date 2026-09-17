import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Fix relative pathing and module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up to src/ directory
sys.path.insert(0, str(PROJECT_ROOT))
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

# If your dataset class/network imports fail due to relative layouts, ensure project root is in path
ROOT_DIR = PROJECT_ROOT.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lvcbf.vision_cbf_network import VisionBarrierNet
from lvcbf.expectile import expectile_loss
from lvcbf.turtlebot_dataset import TurtleBotVisionDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Training params
gamma = 0.5
tau = 0.8
lr = 0.0003
epochs = 30 
d_safe = 0.05
batch_size = 128

# Paths
BASE_DIR = Path(__file__).resolve().parent
models_dir = BASE_DIR / "models"
os.makedirs(models_dir, exist_ok=True)

# Pointing to your newly collected dataset
data_path = ROOT_DIR / "experiments" / "data_collection" / "dataset_waypoint_ghosts.h5"

# ==========================================
# TRAINING SETUP
# ==========================================
dataset = TurtleBotVisionDataset(data_path, models_dir, load_to_ram=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

B_net = VisionBarrierNet().to(DEVICE)
optimizer = torch.optim.Adam(B_net.parameters(), lr=lr)

checkpoint_path = models_dir / "tb3_lvcbf_waypoint_ghost.pt"
start_epoch = load_checkpoint(B_net, optimizer, checkpoint_path, DEVICE)

print(f"Dataset loaded. Total transitions: {len(dataset)}")

# ==========================================
# TRAINING LOOP
# ==========================================
for epoch in range(epochs):
    losses = []

    for img, x, img_next, x_next, d in dataloader:
        img, x = img.to(DEVICE), x.to(DEVICE)
        img_next, x_next = img_next.to(DEVICE), x_next.to(DEVICE)
        d = d.to(DEVICE)

        ell = d - d_safe

        B = B_net(img, x).squeeze()
        with torch.no_grad():
            B_next = B_net(img_next, x_next).squeeze()

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

torch.save(B_net.state_dict(), models_dir / "tb3_lvcbf_waypoint_ghost_final.pt")
print(f"Saved Vision V-OCBF model to {models_dir / 'tb3_lvcbf_waypoint_ghost_final.pt'}")



##3 1st model isi code se train hua tha

# import os
# import sys
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from pathlib import Path

# from turtlebot_dataset import TurtleBotVisionDataset
# from vision_cbf_network import VisionBarrierNet
# from expectile import expectile_loss
# PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up to src/ directory
# sys.path.insert(0, str(PROJECT_ROOT))
# from utils.checkpoint_utils import save_checkpoint, load_checkpoint

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")

# # Training params
# gamma = 0.5
# tau = 0.8
# lr = 0.0003
# epochs = 30 
# d_safe = 0.05
# batch_size = 128

# # Paths
# BASE_DIR = Path(__file__).resolve().parent
# models_dir = BASE_DIR / "models"
# os.makedirs(models_dir, exist_ok=True)
# # Path to dataset in experiments folder
# PROJECT_ROOT_DIR = BASE_DIR.parent.parent  # Go from lvcbf -> src -> root
# data_path = PROJECT_ROOT_DIR / "experiments" / "data_collection" / "turtlebot_vision_dataset.h5"

# # ==========================================
# # TRAINING SETUP
# # ==========================================
# dataset = TurtleBotVisionDataset(data_path, models_dir, load_to_ram=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# B_net = VisionBarrierNet().to(DEVICE)
# optimizer = torch.optim.Adam(B_net.parameters(), lr=lr)

# checkpoint_path = models_dir / "tb3_lvcbf.pt"
# start_epoch = load_checkpoint(B_net, optimizer, checkpoint_path, DEVICE)

# print(f"Dataset loaded. Total transitions: {len(dataset)}")

# # ==========================================
# # TRAINING LOOP
# # ==========================================
# for epoch in range(epochs):
#     losses = []

#     for img, x, img_next, x_next, d in dataloader:
#         img, x = img.to(DEVICE), x.to(DEVICE)
#         img_next, x_next = img_next.to(DEVICE), x_next.to(DEVICE)
#         d = d.to(DEVICE)

#         ell = d - d_safe

#         B = B_net(img, x).squeeze()
#         with torch.no_grad():
#             B_next = B_net(img_next, x_next).squeeze()

#         target = (1 - gamma) * ell + gamma * torch.minimum(ell, B_next)
#         diff = target - B
#         value_loss = expectile_loss(diff, tau)

#         # Boundary constraints
#         unsafe = (d < d_safe)
#         safe   = (d > 2.0) 
        
#         anchor = 0
#         if unsafe.any(): anchor += 0.05 * (B[unsafe] + 1).pow(2).mean()
#         if safe.any(): anchor += 0.01 * (B[safe] - 1).pow(2).mean()

#         loss = value_loss + anchor

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())

#     if epoch % 5 == 0 or epoch == epochs - 1:
#         print(f"Epoch {epoch:03d} | Loss: {np.mean(losses):.4f}")

#     if (epoch + 1) % 5 == 0:
#         save_checkpoint(B_net, optimizer, epoch, checkpoint_path)

# torch.save(B_net.state_dict(), models_dir / "tb3_lvcbf_final.pt")
# print(f"Saved Vision V-OCBF model to {models_dir / 'tb3_lvcbf_final.pt'}")