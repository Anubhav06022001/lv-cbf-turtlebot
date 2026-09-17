import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sys
import cv2
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.mouse_keyboard import MouseKeyboard 
from src.lvcbf.vision_cbf_network import VisionBarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi

# safety fn ka use karke but sirf velcotyi de rahe wheel pe

# --- Configuration & Paths ---
xml_path = str(PROJECT_ROOT / 'robotis_tb3' / 'scene_turtlebot3_burger.xml')
simend = 2200

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.5
SAFETY_OFFSET = 0.22

# --- Load Network & Normalization Data ---
x_mean = np.load(str(PROJECT_ROOT / "src/lvcbf/models/tb3_x_mean.npy"))
x_std = np.load(str(PROJECT_ROOT / "src/lvcbf/models/tb3_x_std.npy"))

B_net = VisionBarrierNet(state_dim=5).to(DEVICE)
model_path = str(PROJECT_ROOT / "models" / "tb3_lvcbf_apf_final.pt")
checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
B_net.eval()

# Global variable to hold the latest image for the physics controller
global_img_torch = None

# def controller(model, data):
#     global global_img_torch

#     # 1. Base Controller: Just drive blindly straight
#     v_des = 0.22
#     w_des = 0.0
#     u_des = np.array([v_des, w_des, 0.0])

#     x, y = data.qpos[0], data.qpos[1]
#     w_quat, x_quat, y_quat, z_quat = data.qpos[3:7]
#     yaw = np.arctan2(2 * (w_quat * z_quat + x_quat * y_quat), 1 - 2 * (y_quat * y_quat + z_quat * z_quat))
#     pos_2d = np.array([x, y])

#     # Default to desired if vision isn't ready
#     v, w = v_des, w_des

#     # 2. Apply LV-CBF Safety Filter
#     if global_img_torch is not None:
#         state_5d = np.array([pos_2d[0], pos_2d[1], yaw, data.qvel[0], data.qvel[5]])
#         x_norm = (state_5d - x_mean) / x_std
#         x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)

#         B_val = B_net(global_img_torch, x_torch)
#         grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
#         grad_B_pos = (grad_B_norm / x_std) 

#         # Calculate Lie derivatives
#         L_g_B_v = (grad_B_pos[0] * np.cos(yaw)) + (grad_B_pos[1] * np.sin(yaw))
#         L_g_B_w = grad_B_pos[2]
#         grad_B_ctrl = np.array([L_g_B_v, L_g_B_w, 0.0])

#         # QP constraint setup
#         B_calibrated = B_val.item() - SAFETY_OFFSET
#         A_list, b_list = [grad_B_ctrl], [-ALPHA * B_calibrated]
        
#         # Override the blind straight command with safe velocities
#         u_safe = cbf_qp_multi(u_des, A_list, b_list)
#         v, w = u_safe[0], u_safe[1]

#     # 3. Convert v and w to wheel velocities
#     w_l = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
#     w_r = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS

#     # 4. Apply to actuators (left is 0, right is 1)
#     data.ctrl[0] = np.clip(w_l, -6.67, 6.67)
#     data.ctrl[1] = np.clip(w_r, -6.67, 6.67)

#     if int(data.time * 100) % 50 == 0:  
#         if global_img_torch is not None:
#             print(f"Time: {data.time:.2f} | B: {B_calibrated:.3f} | u_des: [{v_des:.2f}, {w_des:.2f}] -> u_safe: [{v:.2f}, {w:.2f}]")
#         else:
#             print(f"Time: {data.time:.2f} | Waiting for camera...")

def controller(model, data):
    global global_img_torch

    # 1. Base Controller: BREAK SYMMETRY
    v_des = 0.22
    # By giving it a tiny nudge (0.2), we prevent the math from getting stuck dead-center!
    w_des = 0.2 
    u_des = np.array([v_des, w_des, 0.0])

    x, y = data.qpos[0], data.qpos[1]
    w_quat, x_quat, y_quat, z_quat = data.qpos[3:7]
    yaw = np.arctan2(2 * (w_quat * z_quat + x_quat * y_quat), 1 - 2 * (y_quat * y_quat + z_quat * z_quat))
    pos_2d = np.array([x, y])

    v, w = v_des, w_des

    # 2. Apply LV-CBF Safety Filter
    if global_img_torch is not None:
        state_5d = np.array([pos_2d[0], pos_2d[1], yaw, data.qvel[0], data.qvel[5]])
        x_norm = (state_5d - x_mean) / x_std
        x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)

        B_val = B_net(global_img_torch, x_torch)
        grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
        grad_B_pos = (grad_B_norm / x_std) 

        # Calculate Lie derivatives
        L_g_B_v = (grad_B_pos[0] * np.cos(yaw)) + (grad_B_pos[1] * np.sin(yaw))
        L_g_B_w = grad_B_pos[2]
        grad_B_ctrl = np.array([L_g_B_v, L_g_B_w, 0.0])

        # QP constraint setup
        B_calibrated = B_val.item() - SAFETY_OFFSET
        A_list, b_list = [grad_B_ctrl], [-ALPHA * B_calibrated]
        
        # Override the command with safe velocities
        u_safe = cbf_qp_multi(u_des, A_list, b_list)
        v, w = u_safe[0], u_safe[1]

    # 3. Convert v and w to wheel velocities
    w_l = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    w_r = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS

    # 4. Apply to actuators (left is 0, right is 1)
    data.ctrl[0] = np.clip(w_l, -6.67, 6.67)
    data.ctrl[1] = np.clip(w_r, -6.67, 6.67)

    if int(data.time * 100) % 50 == 0:  
        if global_img_torch is not None:
            print(f"Time: {data.time:.2f} | B: {B_calibrated:.3f} | u_des: [{v_des:.2f}, {w_des:.2f}] -> u_safe: [{v:.2f}, {w:.2f}]")

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# Initial Robot State
mj.mj_resetData(model, data)
data.qpos[:3] = [0.0, 0.0, 0.02] 
data.qpos[3:7] = [1, 0, 0, 0] 
mj.mj_forward(model, data)

# Init GLFW
glfw.init()
window = glfw.create_window(1200, 900, "LV-CBF Static Safety Test", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

mousekbd = MouseKeyboard(model, data, scene, cam)
glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

# --- Robot Inset Camera Setup ---
cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
offscreen_cam = mj.MjvCamera()
offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
offscreen_cam.fixedcamid = cam_id

# Set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    
    # --- Capture and process vision for LV-CBF before stepping physics ---
    read_vport = mj.MjrRect(0, 0, 128, 128)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(read_vport, scene, context)
    
    rgb_arr = np.zeros((128, 128, 3), dtype=np.uint8)
    depth_arr = np.zeros((128, 128, 1), dtype=np.float32)
    mj.mjr_readPixels(rgb_arr, depth_arr, read_vport, context)
    
    rgb_resized = cv2.resize(np.flipud(rgb_arr), (64, 64))
    depth_norm = np.clip(cv2.resize(np.flipud(depth_arr), (64, 64)), 0.0, 3.0) / 3.0
    rgbd = np.concatenate([rgb_resized, np.expand_dims(depth_norm, axis=-1)], axis=-1)
    
    global_img_torch = torch.tensor(np.transpose(rgbd, (2, 0, 1)) / 255.0, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # --- Step Physics ---
    simstart = data.time
    # while (data.time - simstart < 1.0/60.0):
    #     mj.mj_step(model, data)

    while (data.time - simstart < 3.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size( window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render (Main Third Person)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # Top-right wrist camera inset render (320x240)
    inset_w, inset_h = 320, 240
    inset_vport = mj.MjrRect(viewport_width - inset_w, viewport_height - inset_h, inset_w, inset_h)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(inset_vport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()