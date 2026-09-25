#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import mujoco as mj
from mujoco.glfw import glfw
from pathlib import Path

# --- Paths & Setup ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent 
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocbf.vocbf_network import BarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 
from src.utils.mouse_keyboard import MouseKeyboard 
from src.utils.tb3_helpers import (
    simend, WHEEL_RADIUS, TRACK_WIDTH, ROBOT_RADIUS, 
    GOAL_POS, CYLINDER_COORDS, ALPHA, SAFETY_OFFSET,
    quat2yaw, get_min_dist
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"

MEAN_PATH = PROJECT_ROOT / "models" / "tb3_state_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "tb3_state_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "tb3_vocbf.pt"


# --- Initialize MuJoCo ---
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

mj.mj_resetData(model, data)
data.qpos[:3] = [0.0, 0.0, 0.02] 
data.qpos[3:7] = [1, 0, 0, 0] 
mj.mj_forward(model, data)

# --- Load Baseline State Network ---
x_mean = np.load(MEAN_PATH)
x_std = np.load(STD_PATH)

B_net = BarrierNet(state_dim=5).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
B_net.eval()

# --- Initialize GLFW ---
if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1200, 900, "TurtleBot Baseline VOCBF", None, None)
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

is_recovering = False
# recovery_spin = 0.0
recovery_dir = 1.0

locked_w_sign = 0.0

while not glfw.window_should_close(window):
    # Step physics 
    for _ in range(10):
        mj.mj_step(model, data)

    if data.time >= simend or np.linalg.norm(GOAL_POS - data.qpos[:2]) < 0.15:
        print("Simulation ended.")
        break

     # --- 1. Extract State ---
    pos_2d = data.qpos[:2]
    yaw = quat2yaw(data.qpos[3:7])
    v_forward = data.qvel[0]
    w_yaw = data.qvel[5] 
        
    state_5d = np.array([pos_2d[0], pos_2d[1], yaw, v_forward, w_yaw])
    l_val = get_min_dist(pos_2d)

   # --- 2. Base Controller ---
    direction = GOAL_POS - pos_2d
    target_yaw = np.arctan2(direction[1], direction[0])
    yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    
    v_des = 0.15 * max(0.0, np.cos(yaw_err))
    w_des = 1.5 * yaw_err
    u_des = np.array([v_des, w_des, 0.0]) 

    # --- 3. Safety Filter (State-Only) ---
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
    
    A_list, b_list = [grad_B_ctrl], [-ALPHA * B_calibrated]
    
    u_safe = cbf_qp_multi(u_des, A_list, b_list)
    v_safe, w_safe = u_safe[0], u_safe[1]

    # --- 4. YOUR ANTI-CHATTER FIX ---
    if B_calibrated < 0.0:
        # We are in the unsafe region. Lock the direction!
        if locked_w_sign == 0.0:
            if abs(w_safe) > 0.01:
                locked_w_sign = np.sign(w_safe)
            else:
                locked_w_sign = np.sign(yaw_err + 0.01)
        
        # Take the mod (abs) of the QP's chaotic output and force the locked sign.
        # Ensure it spins at least 0.8 rad/s to break static friction!
        w_safe = locked_w_sign * max(abs(w_safe), 0.8)
    else:
        # Back in the safe region, release the lock.
        locked_w_sign = 0.0

    # --- Apply limits directly to hardware ---
    data.ctrl[0] = np.clip((v_safe - w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS, -6.67, 6.67)
    data.ctrl[1] = np.clip((v_safe + w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS, -6.67, 6.67)


    if int(data.time * 100) % 50 == 0:
        print(f"B_raw: {B_raw:.4f} | l: {l_val:.4f} | u_des: [{u_des[0]:.2f}, {u_des[1]:.2f}] | u_safe: [{v_safe:.2f}, {w_safe:.2f}]")


    # --- Render ---
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, int(mj.mjtCatBit.mjCAT_ALL.value), scene)
    mj.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()