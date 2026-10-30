import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Fix relative pathing and module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/
sys.path.insert(0, str(PROJECT_ROOT))
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

ROOT_DIR = PROJECT_ROOT.parent  # repo root
sys.path.insert(0, str(ROOT_DIR / "src"))

from lvcbf.vision_cbf_network import VisionBarrierNet
from lvcbf.expectile import expectile_loss
from lvcbf.turtlebot_dataset import TurtleBotVisionDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# TRAINING PARAMS
# ==========================================
gamma = 0.5
tau = 0.8
lr = 0.0003
epochs = 30
d_safe = 0.05
safe_thresh = 0.40
w_unsafe = 0.05
w_safe = 0.10
batch_size = 128
save_every = 5  # epochs

# ==========================================
# PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = "yawrand_x-3.5to0_v1"
models_dir = BASE_DIR / "models" / RUN_NAME   # stats, checkpoints, README all live here
os.makedirs(models_dir, exist_ok=True)

data_path = ROOT_DIR / "experiments" / "data_collection" / "dataset_waypoint_ghosts_yaw_random.h5"
checkpoint_path = models_dir / "ckpt_latest.pt"          # used for resume
final_path = models_dir / f"tb3_lvcbf_{RUN_NAME}_final.pt"

# Run README (written once, never overwritten)
readme_path = models_dir / "README.md"
if not readme_path.exists():
    readme_path.write_text(
        f"# Run: {RUN_NAME}\n\n"
        f"- Dataset: {data_path.name}\n"
        f"- Data collection: 500 episodes, MAX_STEPS=2000, start x in [-3.5, 0], "
        f"y in [-2.5, 2.5], random yaw, waypoint controller, ghost cylinders, GLFW 64x64 render\n"
        f"- Training: gamma={gamma}, tau={tau}, lr={lr}, epochs={epochs}, d_safe={d_safe}, "
        f"batch_size={batch_size}\n"
        f"- Anchors: unsafe (d<{d_safe}) weight {w_unsafe}, safe (d>{safe_thresh}) weight {w_safe}\n"
        f"- Files: ckpt_epochNNN.pt (every {save_every} epochs), ckpt_latest.pt (resume), "
        f"tb3_x_mean.npy / tb3_x_std.npy (state normalization), {final_path.name} (final weights)\n"
    )

# ==========================================
# TRAINING SETUP
# ==========================================
dataset = TurtleBotVisionDataset(data_path, models_dir, load_to_ram=True)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)

B_net = VisionBarrierNet().to(DEVICE)
optimizer = torch.optim.Adam(B_net.parameters(), lr=lr)

start_epoch = load_checkpoint(B_net, optimizer, checkpoint_path, DEVICE)
print(f"Dataset loaded. Total transitions: {len(dataset)}")

# ==========================================
# TRAINING LOOP
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

        # Boundary constraints
        unsafe = (d < d_safe)
        safe = (d > safe_thresh)

        anchor = 0
        if unsafe.any():
            anchor += w_unsafe * (B[unsafe] + 1).pow(2).mean()
        if safe.any():
            anchor += w_safe * (B[safe] - 1).pow(2).mean()

        loss = value_loss + anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Epoch {epoch:03d} | Loss: {np.mean(losses):.4f}")

    if (epoch + 1) % save_every == 0:
        save_checkpoint(B_net, optimizer, epoch, checkpoint_path)
        save_checkpoint(B_net, optimizer, epoch, models_dir / f"ckpt_epoch{epoch + 1:03d}.pt")

torch.save(B_net.state_dict(), final_path)
print(f"Saved Vision V-OCBF model to {final_path}")