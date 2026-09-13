import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.lvcbf.vision_cbf_network import VisionBarrierNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load your normalization stats
x_mean = np.load("src/lvcbf/models/tb3_x_mean.npy")
x_std  = np.load("src/lvcbf/models/tb3_x_std.npy")

# 2. Load your trained Vision CBF model
B_net = VisionBarrierNet(state_dim=5).to(DEVICE)
checkpoint = torch.load("models/tb3_lvcbf_apf.pt", map_location=DEVICE)
B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
B_net.eval()

vals = []

# 3. Read from your collected HDF5 dataset
# with h5py.File("experiments/data_collection/dataset_waypoint.h5", "r") as f:
with h5py.File("experiments/data_collection/dataset_apf.h5", "r") as f:
    print("HDF5 Keys:", list(f.keys()))
    # Limit to 5000 points to keep processing fast
    states = f["state"][:5000]
    images = f["rgb"][:5000] 
    
    for i in range(len(states)):
        # Format State
        x = (states[i] - x_mean) / x_std             
        x_torch = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        # Format Image (Assuming stored as HxWxC uint8, convert to CxHxW float)
        img = np.transpose(images[i], (2, 0, 1)) / 255.0
        img_torch = torch.tensor(img, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        # Predict B(x)
        b_val = B_net(img_torch, x_torch).item()
        vals.append(b_val)

# 4. Plotting (Your exact styling)
fig, ax = plt.subplots(figsize=(6,4))
plt.hist(vals, bins=60, color="#4C72B0", edgecolor="black")
plt.title("Barrier value distribution", fontsize=14, fontweight="bold")
plt.xlabel("B(x)", fontsize=15, fontweight="bold")
plt.ylabel("Count", fontsize=15, fontweight="bold")
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")

# zero barrier line
ax.axvline(0, color="black", linestyle="--", linewidth=1.5)

# safe and unsafe regions
ax.axvspan(min(vals), 0, color="red", alpha=0.08, label="Unsafe (B(x) < 0)")
ax.axvspan(0, max(vals), color="green", alpha=0.10, label="Safe (B(x) > 0)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
# plt.show()
plt.savefig("barrier_distribution.png", dpi=300, bbox_inches="tight")
print("Plot saved successfully as barrier_distribution.png")