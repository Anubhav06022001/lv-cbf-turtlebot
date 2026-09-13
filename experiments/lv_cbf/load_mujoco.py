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
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvcbf.vision_cbf_network import VisionBarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 
from src.utils.mouse_keyboard import MouseKeyboard 



# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
print("================================" ,PROJECT_ROOT)
xml_path = "/home/anubhav/Documents/lv-cbf-turtle_bot/robotis_tb3/scene_turtlebot3_burger.xml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
models_dir = BASE_DIR / "models"
simend= 20
# TB3 Kinematics
WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
GOAL_POS = np.array([4.0, 0.0])

def quat2yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

# ================= MUJOCO INIT =================
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

# 1. Reset physics
mj.mj_resetData(model, data)

# 5. Update kinematics for the new pose
mj.mj_forward(model, data)
# ================= GLFW INIT & RENDERING =================
if not glfw.init():
    raise Exception("Could not initialize GLFW")

window = glfw.create_window(1800, 1200, "Demo", None, None)
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

cam.azimuth = 180.86
cam.elevation = -15.95
cam.distance = 3.22
cam.lookat = np.array([0.0, 0.0, 0.0])

mousekbd = MouseKeyboard(model, data, scene, cam)

glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')

# Offscreen camera setup for reading pixels
offscreen_cam = mj.MjvCamera()
offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
offscreen_cam.fixedcamid = cam_id

viewport_width, viewport_height = glfw.get_framebuffer_size(window)

# ================= SIM LOOP =================
while not glfw.window_should_close(window):
    time_prev = data.time

    while data.time - time_prev < 1.0/60.0:
        mj.mj_step(model, data)

    if data.time >= simend:
        break

     # 1. Render wrist camera to a hidden 128x128 buffer
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    read_vport = mj.MjrRect(0, 0, 128, 128)
    mj.mjr_render(read_vport, scene, context)    

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(
        model, data, opt, None, cam,
        int(mj.mjtCatBit.mjCAT_ALL.value), scene
    )
    mj.mjr_render(viewport, scene, context)

    # Render Inset Wrist View (Visual only, 320x240)
    inset_w, inset_h = 320, 240
    inset_vport = mj.MjrRect(viewport_width - inset_w, viewport_height - inset_h, inset_w, inset_h)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(inset_vport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
