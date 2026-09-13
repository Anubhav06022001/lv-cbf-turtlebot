import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.mouse_keyboard import MouseKeyboard 
from src.utils.logger import ControlLogger
logger = ControlLogger(PROJECT_ROOT / 'results' / 'csv' / 'baseline_waypoint_controls.csv')

#  planner se ja rahe hain obstalce dodge karte hue goal tak

xml_path = PROJECT_ROOT / 'robotis_tb3' / 'scene_turtlebot3_burger.xml'
simend = 2200

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
ROBOT_RADIUS = 0.10

waypoints = [
    np.array([1.5, 0.0]), 
    np.array([2.0, 0.35]),  
    np.array([2.5, 0.0]),  
    np.array([4.0, 0.0])   
]
current_wp = 0  # Start at the first waypoint

def controller(model, data):
    global current_wp
    
    x, y = data.qpos[0], data.qpos[1]
    w_quat, x_quat, y_quat, z_quat = data.qpos[3:7]
    yaw = np.arctan2(2 * (w_quat * z_quat + x_quat * y_quat), 1 - 2 * (y_quat * y_quat + z_quat * z_quat))
    pos_2d = np.array([x, y])

    if current_wp >= len(waypoints):
        v, w = 0.0, 0.0  # Stop at goal
    else:
        target = waypoints[current_wp]
        dist = np.linalg.norm(target - pos_2d)
        
        if dist < 0.15:  # Accepted radius
            current_wp += 1
            v, w = 0.0, 0.0 
        else:
            target_yaw = np.arctan2(target[1] - y, target[0] - x)
            yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
            
            # BULLETPROOF LOGIC: Align then drive
            if abs(yaw_err) > 0.5: # If off by more than ~30 degrees
                v = 0.0 # Stop moving forward
                w = 2.0 * np.sign(yaw_err) # Rotate in place
            else:
                v = 0.22 # Max speed forward
                w = 2.5 * yaw_err # Minor steering corrections

    # Convert v and w to wheel velocities
    w_l = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    w_r = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    
    # Apply to actuators with strict safety clipping
    data.ctrl[0] = np.clip(w_l, -6.67, 6.67)
    data.ctrl[1] = np.clip(w_r, -6.67, 6.67)

    logger.log(data.time, v, w, data.ctrl[0], data.ctrl[1])

    if int(data.time * 100) % 50 == 0:  
        print(f"Time: {data.time:.2f} | WP: {current_wp} | v: {v:.2f} | w: {w:.2f} | Left: {w_l:.2f} | Right: {w_r:.2f}")

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options


# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

mousekbd = MouseKeyboard(model, data, scene, cam)
# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

# --- Robot Inset Camera Setup ---
cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
offscreen_cam = mj.MjvCamera()
offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
offscreen_cam.fixedcamid = cam_id

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size( window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # 2. Top-right wrist camera inset render (320x240)
    inset_w, inset_h = 320, 240
    inset_vport = mj.MjrRect(viewport_width - inset_w, viewport_height - inset_h, inset_w, inset_h)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(inset_vport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
logger.close()