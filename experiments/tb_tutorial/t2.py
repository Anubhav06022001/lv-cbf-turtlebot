import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.mouse_keyboard import MouseKeyboard 


#  v ,w se command karte hain aka linear and angular velocities, respectively.

xml_path = '/home/anubhav/Documents/lv-cbf-turtle_bot/robotis_tb3/scene_turtlebot3_burger.xml'
simend = 3000

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
ROBOT_RADIUS = 0.10

def controller(model, data):
    v = 0.22  
    w = 0.0   
    
    w_l = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS    # w_l = (v-0.08*w)/0.033
    w_r = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS    # w_r = (v+0.08*w)/0.033
    
    data.ctrl[0] = w_r
    data.ctrl[1] = w_l
    
    data.ctrl[0] = np.clip(data.ctrl[0], -6.67, 6.67)
    data.ctrl[1] = np.clip(data.ctrl[1], -6.67, 6.67)
    
    if int(data.time * 100) % 50 == 0:  
        print(f"Time: {data.time:.2f} | v: {v:.2f} | w: {w:.2f} | Left: {w_l:.2f} | Right: {w_r:.2f}")



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

#set the controller
mj.set_mjcb_control(controller)

#print(os.name)

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

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()