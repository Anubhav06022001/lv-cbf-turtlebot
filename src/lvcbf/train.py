"""
Offline expectile Bellman training of VisionBarrierNet on the TurtleBot3
dataset. Structurally identical to your drone training script; only the
data path, checkpoint name, and (implicitly, via VisionBarrierNet) the
kinematic state dimension change (12D -> 6D).
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.lvcbf.vision_cbf_network import VisionBarrierNet
from src.lvcbf.expectile import expectile_loss
from src.lvcbf.dataset import VisionCBFDataset
from src.utils.checkpoint_utils import save_checkpoint, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Training params (identical to the drone script's values, per Table III)
gamma = 0.5      # matches paper's Bellman Discount (Table III); the drone
                  # training script you sent used 0.9 -- keep whichever you
                  # intend to report, this default follows the paper.
tau = 0.8         # matches paper's Expectile Level (Table III)
lr = 3e-4
epochs = 30
d_safe = 0.05     # matches TurtleBot eval_metrics.D_SAFE; scale to your robot's
                  # real collision radius + margin if different from the drone's.
batch_size = 128

models_dir = PROJECT_ROOT / "models" / "vision_vocbf"
os.makedirs(models_dir, exist_ok=True)

# ==========================================
# 2. TRAINING SETUP
# ==========================================
data_path = PROJECT_ROOT / "data" / "vision_vocbf" / "vision_dataset_turtlebot_v1.h5"
dataset = VisionCBFDataset(data_path, models_dir, load_to_ram=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

B_net = VisionBarrierNet().to(DEVICE)
optimizer = torch.optim.Adam(
    [p for p in B_net.parameters() if p.requires_grad], lr=lr
)  # frozen DINOv2 params are excluded automatically (requires_grad=False)

checkpoint_path = models_dir / "vision_vocbf_turtlebot.pt"
start_epoch = load_checkpoint(B_net, optimizer, checkpoint_path, DEVICE)

print(f"Dataset loaded. Total transitions: {len(dataset)}")

# ==========================================
# 3. TRAINING LOOP
# ==========================================
for epoch in range(start_epoch, epochs):
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

        # Boundary constraints (anchor to +-1 at the safety margins)
        unsafe = (d < d_safe)
        safe = (d > 2.0)

        anchor = 0
        if unsafe.any():
            anchor = anchor + 0.05 * (B[unsafe] + 1).pow(2).mean()
        if safe.any():
            anchor = anchor + 0.01 * (B[safe] - 1).pow(2).mean()

        loss = value_loss + anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d} | Loss: {np.mean(losses):.4f}")

    if (epoch + 1) % 5 == 0:
        save_checkpoint(B_net, optimizer, epoch, checkpoint_path)

torch.save({"model_state_dict": B_net.state_dict()}, checkpoint_path)
print(f"Saved TurtleBot Vision V-OCBF model to {checkpoint_path}")