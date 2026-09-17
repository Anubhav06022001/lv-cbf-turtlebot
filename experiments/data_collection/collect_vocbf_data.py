import os
import sys
import numpy as np
import csv
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

XML_PATH = str(PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml")
CSV_PATH = str(PROJECT_ROOT / "experiments" / "data_collection" / "dataset_vocbf.csv")

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
            if dist < min_dist:
                min_dist = dist
    return min_dist

def compute_waypoint_control(pos, yaw):
    direction = GOAL_POS - pos
    target_yaw = np.arctan2(direction[1], direction[0])
    yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    v = 0.21 * max(0.0, np.cos(yaw_err))
    w = 1.5 * yaw_err
    return v, w

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    os.makedirs(Path(CSV_PATH).parent, exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)
    
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['episode', 'step', 'x', 'y', 'yaw', 'v', 'w', 'w_left', 'w_right', 'min_dist'])

        print("Starting Baseline VOCBF Data Collection...")
        for ep in range(EPISODES):
            mujoco.mj_resetData(model, data)
            start_x, start_y = np.random.uniform(0.0, 3.5), np.random.uniform(-1.5, 1.5)
            data.qpos[:3] = [start_x, start_y, 0.05]
            init_yaw = np.arctan2(GOAL_POS[1] - start_y, GOAL_POS[0] - start_x)
            data.qpos[3:7] = [np.cos(init_yaw / 2), 0, 0, np.sin(init_yaw / 2)] 
            mujoco.mj_forward(model, data)
            
            for step in range(MAX_STEPS):
                pos_2d, yaw = data.qpos[:2], quat2yaw(data.qpos[3:7])
                v_forward, w_yaw = data.qvel[0], data.qvel[5]
                
                v_des, w_des = compute_waypoint_control(pos_2d, yaw)
                min_dist = get_mujoco_min_distance(model, data)
                
                a_left = (v_des - w_des * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
                a_right = (v_des + w_des * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
                
                data.ctrl[0], data.ctrl[1] = a_left, a_right
                
                for _ in range(10):
                    mujoco.mj_step(model, data)
                    
                writer.writerow([ep, step, pos_2d[0], pos_2d[1], yaw, v_forward, w_yaw, a_left, a_right, min_dist])
                
                if np.linalg.norm(GOAL_POS - pos_2d) < 0.15:
                    break 

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}/{EPISODES} completed.")

if __name__ == "__main__":
    main()