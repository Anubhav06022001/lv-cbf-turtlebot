#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import torch
import mujoco as mj
import glfw
import cv2 
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent # Adjust based on experiments/lv_cbf depth
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvcbf.vision_cbf_network import VisionBarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 
from src.utils.mouse_keyboard import MouseKeyboard 
from src.utils.logger import ControlLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"

# Corrected Model & Normalization Paths using PROJECT_ROOT
MEAN_PATH = PROJECT_ROOT / "src" / "lvcbf" / "models" / "tb3_x_mean.npy"
STD_PATH = PROJECT_ROOT / "src" / "lvcbf" / "models" / "tb3_x_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "tb3_lvcbf_waypoint_ghost_final.pt"

simend = 60.0
WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
ROBOT_RADIUS = 0.10
GOAL_POS = np.array([4.0, 0.0])
CYLINDER_COORDS = [(1.0, 0.5), (1.0, -0.5), (2.0, 0.0), (2.0, 1.0), (2.0, -1.0), (3.0, 0.5), (3.0, -0.5)]
SAFETY_MARGIN = -0.0

def quat2yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def get_min_dist(pos_2d):
    min_dist = float('inf')
    for cx, cy in CYLINDER_COORDS:
        dist = np.hypot(pos_2d[0] - cx, pos_2d[1] - cy) - 0.1 - ROBOT_RADIUS
        if dist < min_dist: min_dist = dist
    return min_dist

model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

mj.mj_resetData(model, data)
data.qpos[:3] = [0.0, 0.0, 0.02] 
data.qpos[3:7] = [1, 0, 0, 0] 
mj.mj_forward(model, data)

x_mean = np.load(MEAN_PATH)
x_std = np.load(STD_PATH)

B_net = VisionBarrierNet(state_dim=5).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
state_dict = checkpoint.get('model_state_dict', checkpoint)
B_net.load_state_dict(state_dict)
B_net.eval()

if not glfw.init():
    raise Exception("Could not initialize GLFW")

window = glfw.create_window(1800, 1200, "TurtleBot LV-CBF", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")

glfw.make_context_current(window)
glfw.swap_interval(1)

cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, int(mj.mjtFontScale.mjFONTSCALE_150.value))

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
cam.azimuth = 135.0
cam.elevation = -30.0
cam.distance = 4.0
cam.lookat = np.array([2.0, 0.0, 0.0])

mousekbd = MouseKeyboard(model, data, scene, cam)
glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
offscreen_cam = mj.MjvCamera()
offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
offscreen_cam.fixedcamid = cam_id

alpha = 400000

logger = ControlLogger()
while not glfw.window_should_close(window):
    time_prev = data.time

    while data.time - time_prev < 3.0/60.0:
        mj.mj_step(model, data)

    # Step physics 10 times per control/render loop to speed up simulation
    # for _ in range(10):
    #     mj.mj_step(model, data)

    if data.time >= simend:
        break

    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    read_vport = mj.MjrRect(0, 0, 128, 128)
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

    pos_2d = data.qpos[:2]
    yaw = quat2yaw(data.qpos[3:7]) # position 3:7 contains quaternion (w, x, y, z)
    v_forward = data.qvel[0]
    w_yaw = data.qvel[5]  # angluar vel omega_z at index 5 of qvel, 1st three linear velocities, next three angular velocities
    state_5d = np.array([pos_2d[0], pos_2d[1], yaw, v_forward, w_yaw])
    l_val = get_min_dist(pos_2d)

    direction = GOAL_POS - pos_2d
    target_yaw = np.arctan2(direction[1], direction[0])
    yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    v_des = 0.21 * np.cos(yaw_err)
    w_des = 1.5 * yaw_err
    u_des = np.array([max(0.0, v_des), w_des, 0.0]) 

    x_norm = (state_5d - x_mean) / x_std
    x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)

    B_val = B_net(img_torch, x_torch)
    grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
    grad_B_pos = (grad_B_norm / x_std) 
    # grad_B_ctrl = np.array([grad_B_pos[3], grad_B_pos[4], 0.0]) 

    # grad_B_pos contains 5 elements: [dB/dx, dB/dy, dB/dyaw, dB/dv, dB/dw]
    
    # 1. How commanding 'v' changes the barrier (affects x, y)
    L_g_B_v = (grad_B_pos[0] * np.cos(yaw)) + (grad_B_pos[1] * np.sin(yaw))
    
    # 2. How commanding 'w' changes the barrier (affects yaw)
    L_g_B_w = grad_B_pos[2]
    
    # Pass to the QP
    grad_B_ctrl = np.array([L_g_B_v, L_g_B_w, 0.0])

    B_raw = B_val.item()
    B_calibrated = B_raw - SAFETY_MARGIN

    # A_list, b_list = [grad_B_ctrl], [-alpha * B_val.item()]
    A_list, b_list = [grad_B_ctrl], [-alpha * B_calibrated]
    u_safe = cbf_qp_multi(u_des, A_list, b_list)

    v_safe, w_safe = u_safe[0], u_safe[1]
    # logger.log(data.time, v_safe, w_safe, w_l, w_r)

    data.ctrl[0] = (v_safe - w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    data.ctrl[1] = (v_safe + w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS

    # print(f"B(x): {B_val.item():.4f} | l: {l_val:.4f} | ||grad_B||: {np.linalg.norm(grad_B_ctrl):.4f} | u_des: [{u_des[0]:.4f}, {u_des[1]:.4f}] | u_safe: [{u_safe[0]:.4f}, {u_safe[1]:.4f}]")
    print(f"B_raw: {B_raw:.4f} | B_cal: {B_calibrated:.4f} | l: {l_val:.4f} | u_des: [{u_des[0]:.2f}, {u_des[1]:.2f}] | u_safe: [{u_safe[0]:.2f}, {u_safe[1]:.2f}]")

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, int(mj.mjtCatBit.mjCAT_ALL.value), scene)
    mj.mjr_render(viewport, scene, context)

    inset_w, inset_h = 320, 240
    inset_vport = mj.MjrRect(viewport_width - inset_w, viewport_height - inset_h, inset_w, inset_h)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(inset_vport, scene, context)

    # Render Inset Wrist View (Visual only, 320x240)
    inset_w, inset_h = 320, 240
    inset_vport = mj.MjrRect(viewport_width - inset_w, viewport_height - inset_h, inset_w, inset_h)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(inset_vport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()


    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
# logger.close()

