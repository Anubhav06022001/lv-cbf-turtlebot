import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from pathlib import Path

# --- Setup Paths ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocbf.vocbf_network import BarrierNet

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN_PATH = PROJECT_ROOT / "models" / "tb3_state_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "tb3_state_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "tb3_vocbf.pt"

CYLINDER_RADIUS = 0.10
GOAL_POS = np.array([4.0, 0.0])
cylinders = [(1.0, 0.5), (1.0, -0.5), (2.0, 0.0), (2.0, 1.0), (2.0, -1.0), (3.0, 0.5), (3.0, -0.5)]

# Grid Resolution
NX, NY = 90, 60
x_vals = np.linspace(0, 4.5, NX)
y_vals = np.linspace(-1.5, 1.5, NY)
X, Y = np.meshgrid(x_vals, y_vals)
Z_learned = np.zeros_like(X)

def main():
    print(f"Loading VOCBF Model to {DEVICE}...")
    x_mean = np.load(MEAN_PATH)
    x_std = np.load(STD_PATH)
    
    B_net = BarrierNet(state_dim=5).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    B_net.eval()

    total_points = NX * NY
    print(f"Evaluating {total_points} grid points...")

    with torch.no_grad():
        for i in range(NY):
            for j in range(NX):
                px, py = X[i, j], Y[i, j]
                
                # Assume robot is facing the goal and moving at nominal speed
                yaw = np.arctan2(GOAL_POS[1] - py, GOAL_POS[0] - px)
                
                # Process State [x, y, yaw, v, w]
                state_5d = np.array([px, py, yaw, 0.15, 0.0])
                x_norm = (state_5d - x_mean) / x_std
                x_torch = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                # Predict B(x)
                B_val = B_net(x_torch).item()
                Z_learned[i, j] = B_val

    # --- Plotting ---
    print("Generating IEEE plot...")
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 5))

    contour = ax.contourf(X, Y, Z_learned, levels=50, cmap='RdYlGn', vmin=-1.0, vmax=1.0, alpha=0.8)
    ax.contour(X, Y, Z_learned, levels=[0], colors='black', linewidths=2, linestyles='dashed')

    for cx, cy in cylinders:
        circle = patches.Circle((cx, cy), CYLINDER_RADIUS, facecolor='dimgray', edgecolor='black', zorder=3)
        ax.add_patch(circle)

    ax.plot(0.0, 0.0, 'bo', markersize=10, label='Start Region (Approx)', zorder=4)
    ax.plot(4.0, 0.0, 'b*', markersize=15, label='Goal', zorder=4)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Learned CBF Value B(x)')
    ax.set_title('Learned Environment Safety Map (Baseline VOCBF)')
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left')

    plt.tight_layout()
    
    # Save in the same directory as this script (src/vocbf)
    plot_path = THIS_DIR / "vocbf_safety_map@0.15m_per_s.png"
    plt.show()
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    # print(f"Success! Saved IEEE-ready plot to {plot_path}")

if __name__ == "__main__":
    main()