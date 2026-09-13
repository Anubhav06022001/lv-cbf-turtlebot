import os
import sys
import numpy as np
import mujoco as mj
import imageio
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
simend = 60.0

xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

# Match the default 640x480 offscreen buffer
renderer = mj.Renderer(model, height=480, width=640)

mj.mj_resetData(model, data)
data.qpos[:3] = [0.0, 0.0, 0.02]
data.qpos[3:7] = [1, 0, 0, 0]
mj.mj_forward(model, data)

# Camera Setup
track_cam = mj.MjvCamera()
mj.mjv_defaultCamera(track_cam)
track_cam.distance = 3.5
track_cam.elevation = -35.0
track_cam.azimuth = 135.0

# Output Path: lv-cbf-turtle_bot/results/video
output_dir_vids = PROJECT_ROOT / "results" / "video"
output_dir_vids.mkdir(parents=True, exist_ok=True)

video_path = output_dir_vids / "baseline_waypoint_tracking.mp4"
video_writer = imageio.get_writer(str(video_path), fps=60)

waypoints = [
    np.array([1.5, 0.0]),
    np.array([2.0, 0.5]),
    np.array([2.5, 0.0]),
    np.array([4.0, 0.0])
]
current_wp = 0

print("Rendering baseline demonstration...")

while data.time < simend:
    x, y = data.qpos[0], data.qpos[1]
    w_q, x_q, y_q, z_q = data.qpos[3:7]
    yaw = np.arctan2(2 * (w_q * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q))
    pos_2d = np.array([x, y])

    if current_wp >= len(waypoints):
        v, w = 0.0, 0.0
    else:
        target = waypoints[current_wp]
        dist = np.linalg.norm(target - pos_2d)
        if dist < 0.20:
            current_wp += 1
            v, w = 0.0, 0.0
        else:
            target_yaw = np.arctan2(target[1] - y, target[0] - x)
            yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
            if abs(yaw_err) > 0.5:
                v = 0.0
                w = 2.0 * np.sign(yaw_err)
            else:
                v = 0.22
                w = 2.5 * yaw_err

    w_l = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    w_r = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS

    data.ctrl[0] = np.clip(w_l, -6.67, 6.67)
    data.ctrl[1] = np.clip(w_r, -6.67, 6.67)

    # Step physics to match 60 fps
    step_start = data.time
    while data.time - step_start < (1.0 / 60.0):
        mj.mj_step(model, data)

    # Dynamic tracking camera
    track_cam.lookat[0] = pos_2d[0]
    track_cam.lookat[1] = pos_2d[1]
    track_cam.lookat[2] = 0.0

    renderer.update_scene(data, camera=track_cam)
    video_writer.append_data(renderer.render())

    if current_wp >= len(waypoints) and np.linalg.norm(waypoints[-1] - pos_2d) < 0.05:
        break

video_writer.close()
print(f"Video saved to: {video_path}")