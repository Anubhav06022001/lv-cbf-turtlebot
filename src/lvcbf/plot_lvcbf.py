import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2
import mujoco as mj
from mujoco.glfw import glfw
from pathlib import Path

# --- Setup Paths ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvcbf.vision_cbf_network import VisionBarrierNet

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN_PATH = PROJECT_ROOT / "models" / "tb3_state_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "tb3_state_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" /"lvcbf" / "tb3_lvcbf_balanced_forest_final.pt"
XML_PATH = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"

CYLINDER_RADIUS = 0.10
GOAL_POS = np.array([4.0, 0.0])
cylinders = [(1.0, 0.5), (1.0, -0.5), (2.0, 0.0), (2.0, 1.0), (2.0, -1.0), (3.0, 0.5), (3.0, -0.5)]

NX, NY = 90, 60
x_vals = np.linspace(0, 4.5, NX)
y_vals = np.linspace(-1.5, 1.5, NY)
X, Y = np.meshgrid(x_vals, y_vals)
Z_learned = np.zeros_like(X)

def yaw2quat(yaw):
    return np.array([np.cos(yaw/2.0), 0.0, 0.0, np.sin(yaw/2.0)])

def main():
    print(f"Loading Vision Model to {DEVICE}...")
    x_mean = np.load(MEAN_PATH)
    x_std = np.load(STD_PATH)
    
    B_net = VisionBarrierNet(state_dim=5).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    B_net.eval()

    # --- Initialize MuJoCo for rendering ---
    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(128, 128, "Hidden", None, None)
    glfw.make_context_current(window)

    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, int(mj.mjtFontScale.mjFONTSCALE_150.value))
    
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
    offscreen_cam = mj.MjvCamera()
    offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    offscreen_cam.fixedcamid = cam_id
    read_vport = mj.MjrRect(0, 0, 128, 128)

    print(f"Rendering images and evaluating {NX * NY} grid points (This may take a minute)...")

    with torch.no_grad():
        for i in range(NY):
            for j in range(NX):
                px, py = X[i, j], Y[i, j]
                yaw = np.arctan2(GOAL_POS[1] - py, GOAL_POS[0] - px)
                
                # 1. Teleport robot in MuJoCo
                data.qpos[:2] = [px, py]
                data.qpos[3:7] = yaw2quat(yaw)
                mj.mj_forward(model, data)
                
                # 2. Render Camera Image
                mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
                mj.mjr_render(read_vport, scene, context)
                rgb_arr = np.zeros((128, 128, 3), dtype=np.uint8)
                depth_arr = np.zeros((128, 128, 1), dtype=np.float32)
                mj.mjr_readPixels(rgb_arr, depth_arr, read_vport, context)
                
                rgb_resized = cv2.resize(np.flipud(rgb_arr), (64, 64))
                depth_norm = np.clip(cv2.resize(np.flipud(depth_arr), (64, 64)), 0.0, 3.0) / 3.0
                rgbd = np.concatenate([rgb_resized, np.expand_dims(depth_norm, axis=-1)], axis=-1)
                img_torch = torch.tensor(np.transpose(rgbd, (2, 0, 1)) / 255.0, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # 3. Process State and Predict
                state_5d = np.array([px, py, yaw, 0.15, 0.0])
                x_norm = (state_5d - x_mean) / x_std
                x_torch = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                Z_learned[i, j] = B_net(img_torch, x_torch).item()

            if i % 10 == 0:
                print(f"Processed row {i}/{NY}")

    glfw.terminate()

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
    cbar.set_label('Learned CBF Value B(img, x)')
    ax.set_title('Vision-Language-Action Safety Map (LV-CBF)')
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left')

    plt.tight_layout()
    plot_path = THIS_DIR / "balanced_0.15_m_s.png"
    # plt.show()
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"Success! Saved IEEE-ready plot to {plot_path}")

if __name__ == "__main__":
    main()