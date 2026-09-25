#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import mujoco as mj
from mujoco.glfw import glfw
import cv2
from pathlib import Path

# --- Paths & Setup ---
THIS_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = THIS_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvcbf.vision_cbf_network import VisionBarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 
from src.utils.mouse_keyboard import MouseKeyboard
from src.utils.tb3_helpers import (
     WHEEL_RADIUS, TRACK_WIDTH, ROBOT_RADIUS, 
    GOAL_POS, CYLINDER_COORDS, ALPHA, SAFETY_OFFSET, LOOKAHEAD_DIST, quat2yaw, get_min_dist)
from src.utils.plot_utils import plot_2d_trajectory
simend = 900.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"

# Update paths to the NEW Vision model
MEAN_PATH = PROJECT_ROOT / "models" / "tb3_state_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "tb3_state_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "lvcbf" / "tb3_lvcbf_balanced_forest_final.pt"

# --- Generate Waypoints ---
start_pos = np.array([0.0, 0.0])
spacing = 0.05
n_points = max(2, int(np.linalg.norm(GOAL_POS - start_pos) / spacing))
waypoints = np.linspace(start_pos, GOAL_POS, n_points, endpoint=True)

# --- Initialize MuJoCo ---
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

mj.mj_resetData(model, data)
data.qpos[:3] = [0.0, 0.0, 0.02] 
data.qpos[3:7] = [1, 0, 0, 0] 
mj.mj_forward(model, data)

# --- Load Vision Network ---
x_mean = np.load(MEAN_PATH)
x_std = np.load(STD_PATH)

print(f"Loading Vision model from {MODEL_PATH}...")
B_net = VisionBarrierNet(state_dim=5).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
B_net.eval()

# --- Initialize GLFW ---
if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1200, 900, "TurtleBot Vision LV-CBF", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, int(mj.mjtFontScale.mjFONTSCALE_150.value))

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
cam.distance = 4.0

mousekbd = MouseKeyboard(model, data, scene, cam)
glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

# Offscreen camera for DINOv2
cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
offscreen_cam = mj.MjvCamera()
offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
offscreen_cam.fixedcamid = cam_id

locked_w_sign = 0.0
trajectory_log = []

while not glfw.window_should_close(window):
    for _ in range(10):
        mj.mj_step(model, data)

    pos_2d = data.qpos[:2]
    trajectory_log.append([pos_2d[0], pos_2d[1]])

    if data.time >= simend or np.linalg.norm(GOAL_POS - data.qpos[:2]) < 0.5:
        print("Simulation ended.")
        break
        
    # --- 1. Render Offscreen Camera for Vision Model ---
    read_vport = mj.MjrRect(0, 0, 128, 128)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(read_vport, scene, context)    

    rgb_arr = np.zeros((128, 128, 3), dtype=np.uint8)
    depth_arr = np.zeros((128, 128, 1), dtype=np.float32)
    mj.mjr_readPixels(rgb_arr, depth_arr, read_vport, context)

    rgb_arr = np.flipud(rgb_arr)
    depth_arr = np.flipud(depth_arr)

    rgb_resized = cv2.resize(rgb_arr, (64, 64))
    depth_resized = cv2.resize(depth_arr, (64, 64))
    depth_norm = np.clip(depth_resized, 0.0, 3.0) / 3.0
    rgbd = np.concatenate([rgb_resized, np.expand_dims(depth_norm, axis=-1)], axis=-1)
    img_t = np.transpose(rgbd, (2, 0, 1)) / 255.0
    img_torch = torch.tensor(img_t, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # --- 2. Extract State ---
    pos_2d = data.qpos[:2]
    yaw = quat2yaw(data.qpos[3:7])
    state_5d = np.array([pos_2d[0], pos_2d[1], yaw, data.qvel[0], data.qvel[5]])
    l_val = get_min_dist(pos_2d)

    # --- 3. DYNAMIC LOOKAHEAD (Waypoint Selection) ---
    distances = np.linalg.norm(waypoints - pos_2d, axis=1)
    closest_idx = np.argmin(distances)
    target_idx = closest_idx
    for i in range(closest_idx, len(waypoints)):
        if np.linalg.norm(waypoints[i] - pos_2d) >= LOOKAHEAD_DIST:
            target_idx = i
            break
    target_pos = waypoints[target_idx]

    # --- 3. Base Controller ---
    direction = target_pos - pos_2d
    target_yaw = np.arctan2(direction[1], direction[0])
    yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    
    v_des = 0.21 * max(0.0, np.cos(yaw_err))
    w_des = 1.5 * yaw_err
    u_des = np.array([v_des, w_des, 0.0]) 

    # --- 4. Vision Safety Filter ---
    x_norm = (state_5d - x_mean) / x_std
    x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)

    B_val = B_net(img_torch, x_torch)
    grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
    grad_B_pos = (grad_B_norm / x_std) 
    
    L_g_B_v = (grad_B_pos[0] * np.cos(yaw)) + (grad_B_pos[1] * np.sin(yaw))
    L_g_B_w = grad_B_pos[2]
    grad_B_ctrl = np.array([L_g_B_v, L_g_B_w, 0.0])

    B_raw = B_val.item()
    B_calibrated = B_raw - SAFETY_OFFSET 
    
    A_list, b_list = [grad_B_ctrl], [-ALPHA * B_calibrated]
    u_safe = cbf_qp_multi(u_des, A_list, b_list)
    v_safe, w_safe = u_safe[0], u_safe[1]

    # --- 6. ANTI-CHATTER SYMMETRY BREAKING ---
    if B_calibrated < 0.0:
        if locked_w_sign == 0.0:
            locked_w_sign = np.sign(w_safe) if abs(w_safe) > 0.01 else np.sign(yaw_err + 0.01)
        w_safe = locked_w_sign * max(abs(w_safe), 0.8)
    else:
        locked_w_sign = 0.0

    # --- Apply limits ---
    data.ctrl[0] = np.clip((v_safe - w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS, -6.67, 6.67)
    data.ctrl[1] = np.clip((v_safe + w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS, -6.67, 6.67)

    if int(data.time * 100) % 50 == 0:
        print(f"B_raw: {B_raw:.4f} | B_cal: {B_calibrated:.4f} | l: {l_val:.4f} | Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}] | u_des: [{u_des[0]:.2f}, {u_des[1]:.2f}] | u_safe: [{v_safe:.2f}, {w_safe:.2f}]")

    # --- Render Main Window ---
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, int(mj.mjtCatBit.mjCAT_ALL.value), scene)
    mj.mjr_render(viewport, scene, context)

    # Render Inset Wrist View (Visual only, 320x240)
    inset_w, inset_h = 320, 240
    inset_vport = mj.MjrRect(viewport_width - inset_w, viewport_height - inset_h, inset_w, inset_h)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(inset_vport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
print("Generating trajectory plot...")
plot_2d_trajectory(trajectory_log)