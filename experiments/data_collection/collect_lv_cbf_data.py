import os
import sys
from pathlib import Path
import os
import mujoco
import numpy as np
import h5py
from pathlib import Path
import torch

# This dynamically finds the root folder, no matter what it is named
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve() 
sys.path.append(str(PROJECT_ROOT))  # <--- ADD THIS LINE

XML_PATH = str(PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml")
H5_PATH = str(PROJECT_ROOT / "experiments" / "data_collection" / "dataset_apf.h5")

from src.lvcbf.vision_cbf_network import VisionBarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 
from src.utils.mouse_keyboard import MouseKeyboard

# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



GOAL_POS = np.array([4.0, 0.0])
EPISODES = 500
MAX_STEPS = 600

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
ROBOT_RADIUS = 0.10  

def quat2yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def get_mujoco_min_distance(model, data):
    min_dist = float('inf')
    robot_pos = data.qpos[:2]
    for i in range(1, 8):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{i}")
        if geom_id != -1:
            dist = np.linalg.norm(robot_pos - model.geom_pos[geom_id][:2]) - model.geom_size[geom_id][0] - ROBOT_RADIUS
            if dist < min_dist: min_dist = dist
    return min_dist

def compute_apf_control(pos, yaw, is_reckless):
    f_att = 1.0 * (GOAL_POS - pos)
    k_rep = 0.0 if is_reckless else 0.15 # 0.0 turns off avoidance
    f_rep = np.zeros(2)
    
    coords = [(1.0, 0.5), (1.0, -0.5), (2.0, 0.0), (2.0, 1.0), (2.0, -1.0), (3.0, 0.5), (3.0, -0.5)]
    for cx, cy in coords:
        dist = np.hypot(pos[0] - cx, pos[1] - cy)
        edge_dist = dist - 0.1 - ROBOT_RADIUS
        if 0.01 < edge_dist < 0.6:
            rep_mag = k_rep * (1.0 / edge_dist - 1.0 / 0.6) * (1.0 / (edge_dist**2))
            f_rep += rep_mag * np.array([pos[0] - cx, pos[1] - cy]) / dist
            
    target_yaw = np.arctan2((f_att + f_rep)[1], (f_att + f_rep)[0])
    yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    return max(0.0, 0.25 * np.cos(yaw_err)), 1.5 * yaw_err

def main():
    # model = mujoco.MjModel.from_xml_path(XML_PATH)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)

    os.makedirs(Path(H5_PATH).parent, exist_ok=True)
    file_exists = os.path.exists(H5_PATH)
    # f = h5py.File(H5_PATH, 'a') 
    f = h5py.File(str(H5_PATH), 'a')
    
    if not file_exists:
        d_ep = f.create_dataset('episode', (0,), maxshape=(None,), dtype='i')
        d_state = f.create_dataset('state', (0, 5), maxshape=(None, 5), dtype='f4')
        d_action = f.create_dataset('action', (0, 2), maxshape=(None, 2), dtype='f4')
        d_rgb = f.create_dataset('rgb', (0, 3, 64, 64), maxshape=(None, 3, 64, 64), dtype='u1')
        d_dep = f.create_dataset('depth', (0, 1, 64, 64), maxshape=(None, 1, 64, 64), dtype='f4')
        d_dist = f.create_dataset('distance', (0, 1), maxshape=(None, 1), dtype='f4')
        start_ep, total_transitions = 0, 0
    else:
        d_ep, d_state, d_action = f['episode'], f['state'], f['action']
        d_rgb, d_dep, d_dist = f['rgb'], f['depth'], f['distance']
        total_transitions = d_ep.shape[0]
        start_ep = d_ep[-1] + 1 if total_transitions > 0 else 0

    print(f"Starting APF Data Collection...")
    for ep in range(start_ep, EPISODES):
        is_reckless = (np.random.rand() < 0.40) # 40% chance to crash
        
        mujoco.mj_resetData(model, data)
        start_x, start_y = np.random.uniform(-1.5, 0.0), np.random.uniform(-2.5, 2.5)
        data.qpos[:3] = [start_x, start_y, 0.05]
        init_yaw = np.arctan2(GOAL_POS[1] - start_y, GOAL_POS[0] - start_x)
        data.qpos[3:7] = [np.cos(init_yaw/2), 0, 0, np.sin(init_yaw/2)] 
        mujoco.mj_forward(model, data)
        
        for step in range(MAX_STEPS):
            pos_2d, yaw = data.qpos[:2], quat2yaw(data.qpos[3:7])
            state = np.array([pos_2d[0], pos_2d[1], yaw, data.qvel[0], data.qvel[5]], dtype=np.float32)
            
            v_des, w_des = compute_apf_control(pos_2d, yaw, is_reckless)
            min_dist = get_mujoco_min_distance(model, data)
            
            action = np.array([(v_des - w_des * TRACK_WIDTH / 2.0) / WHEEL_RADIUS, 
                               (v_des + w_des * TRACK_WIDTH / 2.0) / WHEEL_RADIUS], dtype=np.float32)
            data.ctrl[0], data.ctrl[1] = action[0], action[1]
            
            renderer.update_scene(data, camera="wrist_camera")
            rgb = np.transpose(renderer.render(), (2, 0, 1))
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="wrist_camera")
            depth_t = np.expand_dims(np.clip(renderer.render(), 0.0, 3.0) / 3.0, axis=0)
            renderer.disable_depth_rendering()
            
            for _ in range(10): mujoco.mj_step(model, data)
                
            idx = total_transitions
            for dset in (d_ep, d_state, d_action, d_rgb, d_dep, d_dist): dset.resize(idx + 1, axis=0)
            d_ep[idx], d_state[idx], d_action[idx] = ep, state, action
            d_rgb[idx], d_dep[idx], d_dist[idx] = rgb, depth_t, [min_dist]
            total_transitions += 1
            
            if np.linalg.norm(GOAL_POS - pos_2d) < 0.2: break 

        if (ep + 1) % 10 == 0: print(f"Episode {ep + 1}/{EPISODES} | Frames: {total_transitions}")

    f.close()

if __name__ == "__main__":
    main()