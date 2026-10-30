#!/usr/bin/env python
import os
os.environ["MUJOCO_GL"] = "egl"   # same as the data collector
import sys
import numpy as np
import pandas as pd
import torch
import mujoco as mj
from mujoco.glfw import glfw
import cv2
from pathlib import Path

# --- Paths & Setup ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvcbf.vision_cbf_network import VisionBarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi
from src.utils.tb3_helpers import (
    WHEEL_RADIUS, TRACK_WIDTH, ROBOT_RADIUS,
    GOAL_POS, CYLINDER_COORDS, ALPHA, SAFETY_OFFSET, LOOKAHEAD_DIST,
    quat2yaw
)

# ==========================================
# EXPERIMENT SETTINGS (change these per run)
# ==========================================
RUN_DIR = PROJECT_ROOT / "src" / "lvcbf" / "models" / "yawrand_x-3.5to0_v1"
MODEL_PATH = RUN_DIR / "tb3_lvcbf_yawrand_x-3.5to0_v1_final.pt"
MEAN_PATH = RUN_DIR / "tb3_x_mean.npy"
STD_PATH = RUN_DIR / "tb3_x_std.npy"
CSV_PATH = PROJECT_ROOT / "data" / "intial_pos" / "start_positions_500.csv"

# allow sweeping from the command line without editing the file
SAFETY_OFFSET = float(os.environ.get("SAFETY_OFFSET", SAFETY_OFFSET))
ALPHA = float(os.environ.get("ALPHA", ALPHA))
N_EPISODES = int(os.environ.get("N_EPISODES", 100))
USE_CHATTER_BREAKER = os.environ.get("CHATTER", "0") == "1"

SHOW_CAM = False              # True = show camera window (slower, needs display)
INIT_YAW = 0.0                # keep identical across runs you compare
simend = 150.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xml_path = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_burger.xml"


# --- Collision helper (metrics only, NOT used by the controller) ---
def check_physical_collision(pos_2d):
    for cx, cy in CYLINDER_COORDS:
        if np.hypot(pos_2d[0] - cx, pos_2d[1] - cy) < (0.1 + ROBOT_RADIUS):
            return True
    return False


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Start positions not found at {CSV_PATH}.")
    df = pd.read_csv(CSV_PATH)
    EPISODES = min(len(df), N_EPISODES)

    # --- MuJoCo + offscreen GLFW renderer (matches data collection) ---
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    glfw.window_hint(glfw.VISIBLE, 0)
    window = glfw.create_window(64, 64, "Invisible Benchmark Window", None, None)
    glfw.make_context_current(window)

    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, int(mj.mjtFontScale.mjFONTSCALE_150.value))
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
    vport = mj.MjrRect(0, 0, 64, 64)

    # --- Vision network ---
    x_mean = np.load(MEAN_PATH)
    x_std = np.load(STD_PATH)
    print(f"Loading Vision model from {MODEL_PATH}...")
    B_net = VisionBarrierNet(state_dim=5).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    B_net.eval()

    safe_count = reach_count = timeout_count = 0

    print("\n========================================")
    print(f"LV-CBF Benchmark: {EPISODES} episodes | chatter_breaker={USE_CHATTER_BREAKER}")
    print(f"Model: {MODEL_PATH.name}")
    print("========================================")

    with torch.no_grad():
        for ep in range(EPISODES):
            start_x, start_y = df['start_x'][ep], df['start_y'][ep]
            start_pos = np.array([start_x, start_y])

            mj.mj_resetData(model, data)
            data.qpos[:3] = [start_x, start_y, 0.02]
            data.qpos[3:7] = [np.cos(INIT_YAW / 2), 0, 0, np.sin(INIT_YAW / 2)]
            mj.mj_forward(model, data)

            dist = np.linalg.norm(GOAL_POS - start_pos)
            n_points = max(2, int(dist / 0.05))
            waypoints = np.linspace(start_pos, GOAL_POS, n_points, endpoint=True)

            reached = False
            collided = False
            locked_w_sign = 0.0
            b_hist = []

            while data.time < simend:
                for _ in range(10):
                    mj.mj_step(model, data)

                pos_2d = data.qpos[:2]
                yaw = quat2yaw(data.qpos[3:7])
                v_forward, w_yaw = data.qvel[0], data.qvel[5]
                state_5d = np.array([pos_2d[0], pos_2d[1], yaw, v_forward, w_yaw])

                if check_physical_collision(pos_2d):
                    collided = True
                    if b_hist:
                        print(f"ep {ep}: COLLISION at {pos_2d.round(2)} | "
                              f"B_last={b_hist[-1]:.2f} | B_min(last 1s)={min(b_hist[-50:]):.2f}")
                    break
                if np.linalg.norm(GOAL_POS - pos_2d) < 0.15:
                    reached = True
                    break

                # --- Render offscreen camera (same path as data collection) ---
                mj.mjv_updateScene(model, data, opt, None, cam,
                                   mj.mjtCatBit.mjCAT_ALL.value, scene)
                mj.mjr_render(vport, scene, context)
                rgb_arr = np.zeros((64, 64, 3), dtype=np.uint8)
                depth_arr = np.zeros((64, 64, 1), dtype=np.float32)
                mj.mjr_readPixels(rgb_arr, depth_arr, vport, context)
                rgb_arr = np.flipud(rgb_arr)

                if SHOW_CAM:
                    cv2.imshow("wrist_camera",
                               cv2.resize(rgb_arr[:, :, ::-1].copy(), (384, 384)))
                    cv2.waitKey(1)

                # network only reads RGB; 4th channel kept zero for shape
                rgbd = np.concatenate(
                    [rgb_arr, np.zeros((64, 64, 1), dtype=np.uint8)], axis=-1)
                img_t = np.transpose(rgbd, (2, 0, 1)) / 255.0
                img_torch = torch.tensor(img_t, dtype=torch.float32,
                                         device=DEVICE).unsqueeze(0)

                # --- Waypoint lookahead ---
                distances = np.linalg.norm(waypoints - pos_2d, axis=1)
                target_idx = closest_idx = np.argmin(distances)
                for i in range(closest_idx, len(waypoints)):
                    if np.linalg.norm(waypoints[i] - pos_2d) >= LOOKAHEAD_DIST:
                        target_idx = i
                        break
                target_pos = waypoints[target_idx]

                # --- Base controller ---
                direction = target_pos - pos_2d
                target_yaw = np.arctan2(direction[1], direction[0])
                yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
                v_des = 0.21 * max(0.0, np.cos(yaw_err))
                w_des = 1.5 * yaw_err
                u_des = np.array([v_des, w_des, 0.0])

                # --- LV-CBF filter ---
                with torch.enable_grad():
                    x_norm = (state_5d - x_mean) / x_std
                    x_torch = torch.tensor(x_norm, dtype=torch.float32,
                                           requires_grad=True, device=DEVICE).unsqueeze(0)
                    B_val = B_net(img_torch, x_torch)
                    grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]

                grad_B_pos = grad_B_norm / x_std
                L_g_B_v = grad_B_pos[0] * np.cos(yaw) + grad_B_pos[1] * np.sin(yaw)
                L_g_B_w = grad_B_pos[2]
                grad_B_ctrl = np.array([L_g_B_v, L_g_B_w, 0.0])
                B_calibrated = B_val.item() - SAFETY_OFFSET
                b_hist.append(B_calibrated)

                u_safe = cbf_qp_multi(u_des, [grad_B_ctrl], [-ALPHA * B_calibrated])
                v_safe, w_safe = u_safe[0], u_safe[1]

                # --- Optional anti-chatter symmetry breaking (disclosed variant) ---
                if USE_CHATTER_BREAKER:
                    if B_calibrated < 0.0:
                        if locked_w_sign == 0.0:
                            locked_w_sign = (np.sign(w_safe) if abs(w_safe) > 0.01
                                             else np.sign(yaw_err + 0.01))
                        w_safe = locked_w_sign * max(abs(w_safe), 0.8)
                    else:
                        locked_w_sign = 0.0

                # --- Hardware limits ---
                w_l = (v_safe - w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
                w_r = (v_safe + w_safe * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
                data.ctrl[0] = np.clip(w_l, -6.67, 6.67)
                data.ctrl[1] = np.clip(w_r, -6.67, 6.67)

            # --- Tally ---
            if not collided:
                safe_count += 1
            if reached:
                reach_count += 1
            if not collided and not reached:
                timeout_count += 1

            if (ep + 1) % 25 == 0:
                print(f"[{ep + 1:03d}/{EPISODES}] Safety: {safe_count / (ep + 1) * 100:.1f}% | "
                      f"Reach: {reach_count / (ep + 1) * 100:.1f}% | Timeouts: {timeout_count}")

    print("\n" + "=" * 40)
    print("FINAL LV-CBF BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Model              : {MODEL_PATH.name}")
    print(f"Chatter breaker    : {USE_CHATTER_BREAKER}")
    print(f"Episodes           : {EPISODES}")
    print(f"Safety Rate        : {safe_count / EPISODES * 100:.2f}%")
    print(f"Reach Rate         : {reach_count / EPISODES * 100:.2f}%")
    print(f"Timeouts (stalled) : {timeout_count}")
    print(f"Collisions         : {EPISODES - safe_count}")
    print("=" * 40)


    print(f"SAFETY_OFFSET={SAFETY_OFFSET} | ALPHA={ALPHA}")
    glfw.terminate()


if __name__ == "__main__":
    main()