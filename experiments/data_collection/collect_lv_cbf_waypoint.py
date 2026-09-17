import os
import sys
import numpy as np
import h5py
from pathlib import Path

# Crucial: EGL headless backend MUST be set before mujoco imports
os.environ["MUJOCO_GL"] = "egl"
import mujoco

# Dynamically resolve root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

XML_PATH = str(PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml")
H5_PATH = str(PROJECT_ROOT / "experiments" / "data_collection" / "dataset_waypoint_ghosts.h5")

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
    # Check all 7 obstacles based on your XML
    for i in range(1, 8):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{i}")
        if geom_id != -1:
            # If robot goes inside the ghost cylinder, this value naturally becomes negative!
            dist = np.linalg.norm(robot_pos - model.geom_pos[geom_id][:2]) - model.geom_size[geom_id][0] - ROBOT_RADIUS
            if dist < min_dist:
                min_dist = dist
    return min_dist

def compute_waypoint_control(pos, yaw):
    """Directly drives toward the goal using your specified controller."""
    direction = GOAL_POS - pos
    target_yaw = np.arctan2(direction[1], direction[0])
    yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    
    # Matched exactly to your physical limits
    v = 0.21 * max(0.0, np.cos(yaw_err))
    w = 1.5 * yaw_err
    return v, w

def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)

    os.makedirs(Path(H5_PATH).parent, exist_ok=True)
    file_exists = os.path.exists(H5_PATH)
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

    print("Starting Negative-Mining Ghost Data Collection...")
    for ep in range(start_ep, EPISODES):
        mujoco.mj_resetData(model, data)
        # Randomize start positions to get broad coverage of the unsafe regions
        start_x, start_y = np.random.uniform(0.0, 3.5), np.random.uniform(-1.5, 1.5)
        data.qpos[:3] = [start_x, start_y, 0.05]
        init_yaw = np.arctan2(GOAL_POS[1] - start_y, GOAL_POS[0] - start_x)
        data.qpos[3:7] = [np.cos(init_yaw / 2), 0, 0, np.sin(init_yaw / 2)] 
        mujoco.mj_forward(model, data)
        
        for step in range(MAX_STEPS):
            pos_2d, yaw = data.qpos[:2], quat2yaw(data.qpos[3:7])
            state = np.array([pos_2d[0], pos_2d[1], yaw, data.qvel[0], data.qvel[5]], dtype=np.float32)
            
            v_des, w_des = compute_waypoint_control(pos_2d, yaw)
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
            
            for _ in range(10):
                mujoco.mj_step(model, data)
                
            idx = total_transitions
            for dset in (d_ep, d_state, d_action, d_rgb, d_dep, d_dist):
                dset.resize(idx + 1, axis=0)
            d_ep[idx], d_state[idx], d_action[idx] = ep, state, action
            d_rgb[idx], d_dep[idx], d_dist[idx] = rgb, depth_t, [min_dist]
            total_transitions += 1
            
            if np.linalg.norm(GOAL_POS - pos_2d) < 0.15:
                break 

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{EPISODES} | Frames: {total_transitions}")

    f.close()

if __name__ == "__main__":
    main()