# import os
# import sys
# import numpy as np
# import torch
# import mujoco as mj
# import imageio
# from pathlib import Path

# os.environ["MUJOCO_GL"] = "egl"

# # --- Paths & Setup ---
# THIS_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = THIS_DIR.parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.vocbf.vocbf_network import BarrierNet
# from src.control.cbf_qp_multi import cbf_qp_multi 
# from src.utils.tb3_helpers import ( WHEEL_RADIUS, TRACK_WIDTH, GOAL_POS, 
#     ALPHA, SAFETY_OFFSET, LOOKAHEAD_DIST,
#     quat2yaw, get_min_dist
# )

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"

# MEAN_PATH = PROJECT_ROOT / "models" / "tb3_state_mean.npy"
# STD_PATH = PROJECT_ROOT / "models" / "tb3_state_std.npy"
# MODEL_PATH = PROJECT_ROOT / "models" / "tb3_vocbf.pt"

# simend = 900.0 


# # --- Load Network ---
# x_mean = np.load(MEAN_PATH)
# x_std = np.load(STD_PATH)
# B_net = BarrierNet(state_dim=5).to(DEVICE)
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
# B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
# B_net.eval()

# # --- Initialize MuJoCo Headless ---
# model = mj.MjModel.from_xml_path(str(xml_path))
# renderer = mj.Renderer(model, height=480, width=640)

# track_cam = mj.MjvCamera()
# mj.mjv_defaultCamera(track_cam)
# track_cam.distance = 3.5
# track_cam.elevation = -35.0
# track_cam.azimuth = 135.0

# output_dir_vids = PROJECT_ROOT / "experiments" / "vocbf" / "videos"
# output_dir_vids.mkdir(parents=True, exist_ok=True)

# print("\n--- Recording Case: 4_success_anti_chatter ---")
# data = mj.MjData(model)
# mj.mj_resetData(model, data)
# data.qpos[:3] = [0.0, 0.0, 0.02] 
# data.qpos[3:7] = [1, 0, 0, 0] 
# mj.mj_forward(model, data)

# video_path = output_dir_vids / "4_success_anti_chatter.mp4"
# video_writer = imageio.get_writer(str(video_path), fps=60)

# locked_w_sign = 0.0

# while data.time < simend:
#     pos_2d = data.qpos[:2]
#     w_q, x_q, y_q, z_q = data.qpos[3:7]
#     yaw = np.arctan2(2 * (w_q * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q))
#     v_forward, w_yaw = data.qvel[0], data.qvel[5] 
#     state_5d = np.array([pos_2d[0], pos_2d[1], yaw, v_forward, w_yaw])

#     # --- 2. Base Controller ---
#     direction = GOAL_POS - pos_2d
#     target_yaw = np.arctan2(direction[1], direction[0])
#     yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    
#     v_des = 0.15 * max(0.0, np.cos(yaw_err))
#     w_des = 1.5 * yaw_err
#     u_des = np.array([v_des, w_des, 0.0]) 

#     # --- 3. Safety Filter (State-Only) ---
#     x_norm = (state_5d - x_mean) / x_std
#     x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)

#     B_val = B_net(x_torch)
#     grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
#     grad_B_pos = (grad_B_norm / x_std) 
    
#     L_g_B_v = (grad_B_pos[0] * np.cos(yaw)) + (grad_B_pos[1] * np.sin(yaw))
#     L_g_B_w = grad_B_pos[2]
#     grad_B_ctrl = np.array([L_g_B_v, L_g_B_w, 0.0])

#     B_raw = B_val.item()
#     B_calibrated = B_raw - SAFETY_OFFSET 
    
#     A_list, b_list = [grad_B_ctrl], [-ALPHA * B_calibrated]
    
#     u_safe = cbf_qp_multi(u_des, A_list, b_list)
#     v_safe, w_safe = u_safe[0], u_safe[1]

#     # --- 4. YOUR ANTI-CHATTER FIX ---
#     if B_calibrated < 0.0:
#         if locked_w_sign == 0.0:
#             if abs(w_safe) > 0.01:
#                 locked_w_sign = np.sign(w_safe)
#             else:
#                 locked_w_sign = np.sign(yaw_err + 0.01)
#         w_safe = locked_w_sign * max(abs(w_safe), 0.8)
#     else:
#         locked_w_sign = 0.0

#     # --- Apply limits directly to hardware ---
#     w_l = (v_safe - w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
#     w_r = (v_safe + w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
#     data.ctrl[0] = np.clip(w_l, -6.67, 6.67)
#     data.ctrl[1] = np.clip(w_r, -6.67, 6.67)

#     # Step physics
#     step_start = data.time
#     while data.time - step_start < (1.0 / 60.0):
#         mj.mj_step(model, data)

#     # Rendering & Exit Logic
#     track_cam.lookat[0] = pos_2d[0]
#     track_cam.lookat[1] = pos_2d[1]
#     track_cam.lookat[2] = 0.0
#     renderer.update_scene(data, camera=track_cam)
#     video_writer.append_data(renderer.render())

#     if np.linalg.norm(GOAL_POS - pos_2d) < 0.15:
#         print("Goal Reached!")
#         break

# video_writer.close()
# renderer.close()
# print(f"Saved: {video_path}")










import os
import sys
import numpy as np
import torch
import mujoco as mj
import imageio
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

# --- Paths & Setup ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocbf.vocbf_network import BarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 
from src.utils.tb3_helpers import (
    WHEEL_RADIUS, TRACK_WIDTH, GOAL_POS, 
    ALPHA, SAFETY_OFFSET, LOOKAHEAD_DIST,
    quat2yaw, get_min_dist
)

simend = 900.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"

MEAN_PATH = PROJECT_ROOT / "models" / "tb3_state_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "tb3_state_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "tb3_vocbf.pt"

# --- Setup Waypoints ---
start_pos = np.array([0.0, 0.0])
goal_pos = np.array([4.0, 0.0])
spacing = 0.05
n_points = max(2, int(np.linalg.norm(goal_pos - start_pos) / spacing))
waypoints = np.linspace(start_pos, goal_pos, n_points, endpoint=True)

# --- Load Network & MuJoCo ---
x_mean = np.load(MEAN_PATH)
x_std = np.load(STD_PATH)
B_net = BarrierNet(state_dim=5).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
B_net.eval()

model = mj.MjModel.from_xml_path(str(xml_path))
renderer = mj.Renderer(model, height=480, width=640)
track_cam = mj.MjvCamera()
mj.mjv_defaultCamera(track_cam)
track_cam.distance, track_cam.elevation, track_cam.azimuth = 3.5, -35.0, 135.0

output_dir_vids = PROJECT_ROOT / "experiments" / "vocbf" / "videos"
output_dir_vids.mkdir(parents=True, exist_ok=True)
video_path = output_dir_vids / "4_success_waypoint_tracking.mp4"
# Using 50fps to perfectly match the 0.02s (10-step) control loop
video_writer = imageio.get_writer(str(video_path), fps=50) 

print(f"\n--- Recording Success Case to {video_path.name} ---")
data = mj.MjData(model)
mj.mj_resetData(model, data)
data.qpos[:3] = [0.0, 0.0, 0.02] 
data.qpos[3:7] = [1, 0, 0, 0] 
mj.mj_forward(model, data)

locked_w_sign = 0.0

while data.time < simend:
    # 1. Step physics EXACTLY 10 times (Matches static_test.py perfectly)
    for _ in range(10):
        mj.mj_step(model, data)

    pos_2d = data.qpos[:2]
    yaw = quat2yaw(data.qpos[3:7])
    v_forward, w_yaw = data.qvel[0], data.qvel[5] 
    state_5d = np.array([pos_2d[0], pos_2d[1], yaw, v_forward, w_yaw])

    if np.linalg.norm(GOAL_POS - pos_2d) < 0.15:
        print("Goal Reached!")
        break

    # Lookahead Waypoint Selection
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
    u_des = np.array([0.15 * max(0.0, np.cos(yaw_err)), 1.5 * yaw_err, 0.0]) 

    # CBF Filter
    x_norm = (state_5d - x_mean) / x_std
    x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)
    B_val = B_net(x_torch)
    grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
    grad_B_pos = (grad_B_norm / x_std) 
    
    L_g_B_v = (grad_B_pos[0] * np.cos(yaw)) + (grad_B_pos[1] * np.sin(yaw))
    grad_B_ctrl = np.array([L_g_B_v, grad_B_pos[2], 0.0])
    B_calibrated = B_val.item() - SAFETY_OFFSET 
    
    u_safe = cbf_qp_multi(u_des, [grad_B_ctrl], [-ALPHA * B_calibrated])
    v_safe, w_safe = u_safe[0], u_safe[1]

    # Anti-Chatter Override
    if B_calibrated < 0.0:
        if locked_w_sign == 0.0:
            locked_w_sign = np.sign(w_safe) if abs(w_safe) > 0.01 else np.sign(yaw_err + 0.01)
        w_safe = locked_w_sign * max(abs(w_safe), 0.8)
    else:
        locked_w_sign = 0.0

    # Hardware Math
    data.ctrl[0] = np.clip((v_safe - w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS, -6.67, 6.67)
    data.ctrl[1] = np.clip((v_safe + w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS, -6.67, 6.67)

    # Rendering (Notice the bottom time-loop is GONE)
    track_cam.lookat[0], track_cam.lookat[1], track_cam.lookat[2] = pos_2d[0], pos_2d[1], 0.0
    renderer.update_scene(data, camera=track_cam)
    video_writer.append_data(renderer.render())

video_writer.close()
renderer.close()
print(f"Saved: {video_path}")