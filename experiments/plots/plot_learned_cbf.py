import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2
from pathlib import Path

# MUST be set before mujoco imports
os.environ["MUJOCO_GL"] = "egl"
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvcbf.vision_cbf_network import VisionBarrierNet

plot_dir = PROJECT_ROOT / "experiments" / "results" / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XML_PATH = str(PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml")
MEAN_PATH = str(PROJECT_ROOT / "src/lvcbf/models/tb3_x_mean.npy")
STD_PATH = str(PROJECT_ROOT / "src/lvcbf/models/tb3_x_std.npy")
MODEL_PATH = str(PROJECT_ROOT / "models" / "tb3_lvcbf_apf_final.pt")

CYLINDER_RADIUS = 0.10
GOAL_POS = np.array([4.0, 0.0])

cylinders = [(1.0, 0.5), (1.0, -0.5), (2.0, 0.0), (2.0, 1.0), (2.0, -1.0), (3.0, 0.5), (3.0, -0.5)]

# Grid Resolution (90x60 = 5400 points). Adjust if you want it faster or higher quality.
NX, NY = 90, 60
x_vals = np.linspace(0, 4.5, NX)
y_vals = np.linspace(-1.5, 1.5, NY)
X, Y = np.meshgrid(x_vals, y_vals)
Z_learned = np.zeros_like(X)

def main():
    print(f"Loading Model to {DEVICE}...")
    x_mean = np.load(MEAN_PATH)
    x_std = np.load(STD_PATH)
    
    B_net = VisionBarrierNet(state_dim=5).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    B_net.eval()

    print("Initializing MuJoCo Renderer...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=128, width=128)
    
    # Hide the goal marker if it blocks the camera view
    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal")
    if goal_id != -1:
        model.geom_rgba[goal_id][3] = 0.0  # Make transparent for data collection

    total_points = NX * NY
    print(f"Evaluating {total_points} grid points. This will take a few minutes...")

    with torch.no_grad():
        for i in range(NY):
            for j in range(NX):
                px, py = X[i, j], Y[i, j]
                
                # Assume robot is facing the goal and moving at nominal speed
                yaw = np.arctan2(GOAL_POS[1] - py, GOAL_POS[0] - px)
                
                # Place robot in MuJoCo
                data.qpos[:3] = [px, py, 0.00]
                data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                mujoco.mj_forward(model, data)
                
                # Render Wrist Camera
                renderer.update_scene(data, camera="wrist_camera")
                rgb_arr = renderer.render()
                
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera="wrist_camera")
                depth_arr = np.expand_dims(renderer.render(), axis=-1)
                renderer.disable_depth_rendering()
                
                # Process Image
                rgb_resized = cv2.resize(rgb_arr, (64, 64))
                depth_norm = np.clip(cv2.resize(depth_arr, (64, 64)), 0.0, 3.0) / 3.0
                rgbd = np.concatenate([rgb_resized, np.expand_dims(depth_norm, axis=-1)], axis=-1)
                img_torch = torch.tensor(np.transpose(rgbd, (2, 0, 1)) / 255.0, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                # Process State
                state_5d = np.array([px, py, yaw, 0.21, 0.0])
                x_norm = (state_5d - x_mean) / x_std
                x_torch = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                # Predict B(x)
                B_val = B_net(img_torch, x_torch).item()
                Z_learned[i, j] = B_val
                
            if (i + 1) % 10 == 0:
                print(f"Processed row {i + 1}/{NY} ({(i + 1)/NY*100:.1f}%)")

    # --- Plotting ---
    print("Generating IEEE plot...")
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 5))

    # Contour map for learned B(x)
    contour = ax.contourf(X, Y, Z_learned, levels=50, cmap='RdYlGn', vmin=-1.0, vmax=1.0, alpha=0.8)
    
    # Solid black line where B(x) = 0 (The learned safety boundary)
    ax.contour(X, Y, Z_learned, levels=[0], colors='black', linewidths=2, linestyles='dashed')

    # Draw the physical cylinders for reference
    for cx, cy in cylinders:
        circle = patches.Circle((cx, cy), CYLINDER_RADIUS, facecolor='dimgray', edgecolor='black', zorder=3)
        ax.add_patch(circle)

    ax.plot(0.0, 0.0, 'bo', markersize=10, label='Start Region (Approx)', zorder=4)
    ax.plot(4.0, 0.0, 'b*', markersize=15, label='Goal', zorder=4)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Learned CBF Value $B(x)$')
    ax.set_title('Learned Environment Safety Map (Neural LV-CBF)')
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left')

    plt.tight_layout()
    plot_path = plot_dir / "learned_safety_map.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"Success! Saved IEEE-ready plot to {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()