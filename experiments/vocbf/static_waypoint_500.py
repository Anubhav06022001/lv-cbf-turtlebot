#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import torch
import mujoco as mj
from pathlib import Path

# --- Paths & Setup ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent 
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocbf.vocbf_network import BarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 
from src.utils.tb3_helpers import (
     WHEEL_RADIUS, TRACK_WIDTH, ROBOT_RADIUS, 
    GOAL_POS, CYLINDER_COORDS, ALPHA, SAFETY_OFFSET, LOOKAHEAD_DIST,
    quat2yaw
)
simend = 900.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"

MEAN_PATH = PROJECT_ROOT / "models" / "tb3_state_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "tb3_state_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "tb3_vocbf.pt"

# --- Collision Helper for Metrics ---
def check_physical_collision(pos_2d):
    """
    Checks if the robot's physical shell overlaps with a cylinder's physical shell.
    Cylinder radius is 0.1m, Robot radius is 0.1m. Collision occurs if hypot < 0.20m.
    """
    for cx, cy in CYLINDER_COORDS:
        if np.hypot(pos_2d[0] - cx, pos_2d[1] - cy) < (0.1 + ROBOT_RADIUS):
            return True
    return False

def main():
    # --- Load Benchmark Dataset ---
    csv_path = PROJECT_ROOT / "data" / "intial_pos" / "start_positions_500.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run your generation script first.")
    
    df = pd.read_csv(csv_path)
    EPISODES = len(df)
    
    # --- Initialize MuJoCo Headless ---
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    # --- Load Baseline State Network ---
    x_mean = np.load(MEAN_PATH)
    x_std = np.load(STD_PATH)
    B_net = BarrierNet(state_dim=5).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    B_net.eval()

    safe_count = 0
    reach_count = 0
    
    print(f"\n========================================")
    print(f"Starting VOCBF Benchmark: {EPISODES} Episodes")
    print(f"========================================")

    for ep in range(EPISODES):
        # 1. Reset Environment for New Episode
        start_x, start_y = df['start_x'][ep], df['start_y'][ep]
        start_pos = np.array([start_x, start_y])
        
        mj.mj_resetData(model, data)
        data.qpos[:3] = [start_x, start_y, 0.02] 
        data.qpos[3:7] = [1, 0, 0, 0] # Face forward initially
        mj.mj_forward(model, data)

        # 2. Generate Waypoints for this Start Position
        dist = np.linalg.norm(GOAL_POS - start_pos)
        n_points = max(2, int(dist / 0.05))
        waypoints = np.linspace(start_pos, GOAL_POS, n_points, endpoint=True)

        reached = False
        collided = False
        locked_w_sign = 0.0

        # 3. Simulation Loop
        while data.time < simend:
            # Step physics 10 times (50Hz control loop)
            for _ in range(10):
                mj.mj_step(model, data)

            pos_2d = data.qpos[:2]
            yaw = quat2yaw(data.qpos[3:7])
            v_forward, w_yaw = data.qvel[0], data.qvel[5] 
            state_5d = np.array([pos_2d[0], pos_2d[1], yaw, v_forward, w_yaw])

            # Check Safety Metric
            if check_physical_collision(pos_2d):
                collided = True
                break

            # Check Reachability Metric
            if np.linalg.norm(GOAL_POS - pos_2d) < 0.15:
                reached = True
                break

            # DYNAMIC LOOKAHEAD (Waypoint Selection)
            distances = np.linalg.norm(waypoints - pos_2d, axis=1)
            target_idx = closest_idx = np.argmin(distances)
            for i in range(closest_idx, len(waypoints)):
                if np.linalg.norm(waypoints[i] - pos_2d) >= LOOKAHEAD_DIST:
                    target_idx = i
                    break
            target_pos = waypoints[target_idx]

            # Base Controller
            direction = target_pos - pos_2d
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
            B_calibrated = B_val.item() - SAFETY_OFFSET 
            
            u_safe = cbf_qp_multi(u_des, [grad_B_ctrl], [-ALPHA * B_calibrated])
            v_safe, w_safe = u_safe[0], u_safe[1]

            # ANTI-CHATTER SYMMETRY BREAKING
            if B_calibrated < 0.0:
                if locked_w_sign == 0.0:
                    locked_w_sign = np.sign(w_safe) if abs(w_safe) > 0.01 else np.sign(yaw_err + 0.01)
                w_safe = locked_w_sign * max(abs(w_safe), 0.8)
            else:
                locked_w_sign = 0.0

            # Hardware limits
            w_l = (v_safe - w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
            w_r = (v_safe + w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
            data.ctrl[0] = np.clip(w_l, -6.67, 6.67)
            data.ctrl[1] = np.clip(w_r, -6.67, 6.67)

        # --- Tally Metrics ---
        if not collided:
            safe_count += 1
        if reached:
            reach_count += 1
            
        if (ep + 1) % 25 == 0:
            current_safe = (safe_count / (ep + 1)) * 100
            current_reach = (reach_count / (ep + 1)) * 100
            print(f"[{ep + 1:03d}/{EPISODES}] Safety Rate: {current_safe:.1f}% | Reach Rate: {current_reach:.1f}%")

    # --- Final Results ---
    print("\n" + "="*40)
    print("FINAL VOCBF BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Episodes Run : {EPISODES}")
    print(f"Overall Safety Rate: {(safe_count / EPISODES) * 100:.2f}%")
    print(f"Overall Reach Rate : {(reach_count / EPISODES) * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()