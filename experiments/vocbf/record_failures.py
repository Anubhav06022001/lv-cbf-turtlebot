import os
import sys
import numpy as np
import torch
import mujoco as mj
import imageio
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

# --- Paths & Setup ---
THIS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocbf.vocbf_network import BarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"

MEAN_PATH = PROJECT_ROOT / "models" / "tb3_state_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "tb3_state_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "tb3_vocbf.pt"

simend = 60.0  # 40 seconds is plenty to capture the failures
WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
ROBOT_RADIUS = 0.10
GOAL_POS = np.array([4.0, 0.0])
CYLINDER_COORDS = [(1.0, 0.5), (1.0, -0.5), (2.0, 0.0), (2.0, 1.0), (2.0, -1.0), (3.0, 0.5), (3.0, -0.5)]

ALPHA = 4.0
SAFETY_OFFSET = 0.10

# --- Load Network ---
x_mean = np.load(MEAN_PATH)
x_std = np.load(STD_PATH)
B_net = BarrierNet(state_dim=5).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
B_net.eval()

# --- Initialize MuJoCo Headless ---
model = mj.MjModel.from_xml_path(str(xml_path))
renderer = mj.Renderer(model, height=480, width=640)

track_cam = mj.MjvCamera()
mj.mjv_defaultCamera(track_cam)
track_cam.distance = 3.5
track_cam.elevation = -35.0
track_cam.azimuth = 135.0

output_dir_vids = PROJECT_ROOT / "experiments" / "vocbf" / "videos"
output_dir_vids.mkdir(parents=True, exist_ok=True)

failure_cases = ["1_stiction", "2_hysteresis", "3_semicircle"]

for case_name in failure_cases:
    print(f"\n--- Recording Case: {case_name} ---")
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, 0.02] 
    data.qpos[3:7] = [1, 0, 0, 0] 
    mj.mj_forward(model, data)

    video_path = output_dir_vids / f"{case_name}_failure.mp4"
    video_writer = imageio.get_writer(str(video_path), fps=60)

    is_recovering = False
    recovery_dir = 1.0

    while data.time < simend:
        pos_2d = data.qpos[:2]
        w_q, x_q, y_q, z_q = data.qpos[3:7]
        yaw = np.arctan2(2 * (w_q * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q))
        
        v_forward, w_yaw = data.qvel[0], data.qvel[5] 
        state_5d = np.array([pos_2d[0], pos_2d[1], yaw, v_forward, w_yaw])

        # Base Controller
        direction = GOAL_POS - pos_2d
        target_yaw = np.arctan2(direction[1], direction[0])
        yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
        v_des = 0.15 * max(0.0, np.cos(yaw_err))
        w_des = 1.5 * yaw_err
        u_des = np.array([v_des, w_des, 0.0]) 

        # CBF Filter
        x_norm = (state_5d - x_mean) / x_std
        x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)
        B_val = B_net(x_torch)
        grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
        grad_B_pos = (grad_B_norm / x_std) 
        L_g_B_v = (grad_B_pos[0] * np.cos(yaw)) + (grad_B_pos[1] * np.sin(yaw))
        L_g_B_w = grad_B_pos[2]
        grad_B_ctrl = np.array([L_g_B_v, L_g_B_w, 0.0])

        B_raw = B_val.item()
        B_calibrated = B_raw - SAFETY_OFFSET 
        
        u_safe = cbf_qp_multi(u_des, [grad_B_ctrl], [-ALPHA * B_calibrated])
        v_safe, w_safe = u_safe[0], u_safe[1]

        # --- FAILURE CASE LOGIC ---
        if case_name == "1_stiction":
            pass # No recovery, rely solely on pure QP solver

        elif case_name == "2_hysteresis":
            if not is_recovering:
                if v_safe < 0.05 and B_calibrated < 0.00:
                    is_recovering = True
                    recovery_dir = -1.0 * np.sign(yaw_err + 0.01)
            else:
                if B_calibrated > 0.15:
                    is_recovering = False
            if is_recovering:
                v_safe, w_safe = 0.0, recovery_dir

        elif case_name == "3_semicircle":
            if not is_recovering:
                if v_safe < 0.05 and B_calibrated < 0.00:
                    is_recovering = True
                    recovery_dir = np.sign(L_g_B_w) if abs(L_g_B_w) > 0.01 else np.sign(yaw_err + 0.01)
            else:
                if B_calibrated > 0.10:
                    is_recovering = False
            if is_recovering:
                v_safe, w_safe = 0.15, recovery_dir * 1.0

        # --- Hardware Math ---
        w_l = (v_safe - w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        w_r = (v_safe + w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        data.ctrl[0] = np.clip(w_l, -6.67, 6.67)
        data.ctrl[1] = np.clip(w_r, -6.67, 6.67)

        # Step physics to match 60 fps
        step_start = data.time
        while data.time - step_start < (1.0 / 60.0):
            mj.mj_step(model, data)

        # Stop early if goal reached or crashed
        if np.linalg.norm(GOAL_POS - pos_2d) < 0.15:
            break
        # Detect crash (Semi-circle failure jump)
        if data.time > 1.0 and abs(data.qpos[2] - 0.02) > 0.1: 
            print("Collision detected, ending recording early.")
            break

        # Dynamic tracking camera
        track_cam.lookat[0] = pos_2d[0]
        track_cam.lookat[1] = pos_2d[1]
        track_cam.lookat[2] = 0.0
        renderer.update_scene(data, camera=track_cam)
        video_writer.append_data(renderer.render())

    video_writer.close()
    print(f"Saved: {video_path}")

print("\nAll recordings finished successfully!")