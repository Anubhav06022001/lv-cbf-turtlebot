"""
Single-episode dynamic-obstacle run for the TurtleBot3 Waffle Pi, with live
MuJoCo viewer + trajectory/time-series plots at the end. Analogue of your
drone single-case dynamic debug script -- useful for Fig. 3-style qualitative
figures of one interception scenario before running the full 100-episode
dynamic benchmark.
"""
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.env_setup import init_drone_env, XML_PATH
init_drone_env()

from src.utils.turtlebot_mujoco_env import TurtleBotMuJoCoEnv
from src.utils.env_utils import get_2d_min_distance, CYLINDER_COORDS, place_goal_marker
from src.utils.process_image import process_image
from src.utils.plot_utils import plot_2d_dynamic_trajectory, plot_time_series
from src.vocbf.vision_cbf_network import VisionBarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi
from src.control.unicycle_controller import lookahead_point, unicycle_from_planar_velocity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = PROJECT_ROOT / "models" / "vision_vocbf"

x_mean = np.load(models_dir / "vis_x_mean.npy")
x_std = np.load(models_dir / "vis_x_std.npy")

B_net = VisionBarrierNet().to(DEVICE)
checkpoint = torch.load(models_dir / "vision_vocbf_turtlebot.pt", map_location=DEVICE)
B_net.load_state_dict(checkpoint["model_state_dict"])
B_net.eval()

MIDDLE_CYL_IDX = 2

# --- Edit these to pick your single test case ---
START_POS = np.array([0.0, 1.0])
START_THETA = 0.0
GOAL_POS = np.array([4.0, 0.0])
CYL_START_Y = -2.25
CYL_SPEED = 0.002       # [m/step], matches your drone single-case script
USE_GUI = True
MAX_STEPS = 7000
LOOKAHEAD_DIST = 1.2
BASE_ALPHA = 0.5
ALPHA_FREEZE = 0.20
BUFFER = 0.20
V_THRESH = 0.03
D_THRESH = 0.50
V_SWERVE = 0.15


def main():
    env = TurtleBotMuJoCoEnv(XML_PATH, render_width=64, render_height=64, gui=USE_GUI)
    place_goal_marker(env, GOAL_POS)
    static_coords = [c for i, c in enumerate(CYLINDER_COORDS) if i != MIDDLE_CYL_IDX]

    viewer = None
    if USE_GUI:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    dist = np.linalg.norm(GOAL_POS - START_POS)
    n = max(2, int(dist / 0.1))
    waypoints = np.linspace(START_POS, GOAL_POS, n, endpoint=True)

    obs = env.reset((START_POS[0], START_POS[1], START_THETA))
    env.set_obstacle_positions(static_coords + [(2.0, CYL_START_Y)])

    trajectory, obstacle_trajectory = [], []
    dist_log, speed_log = [], []

    for step in range(MAX_STEPS):
        new_y = CYL_START_Y + step * CYL_SPEED
        if new_y > 2.25:
            new_y = 2.25
        env.set_obstacle_positions(static_coords + [(2.0, new_y)])
        dynamic_coords = static_coords + [(2.0, new_y)]
        obstacle_trajectory.append((2.0, new_y))

        state6 = obs["0"]["state6"]
        x, y, cth, sth, v, omega = state6
        theta = np.arctan2(sth, cth)
        pos = np.array([x, y])
        trajectory.append((x, y))

        img_torch = process_image(obs["0"], device=DEVICE)

        lp = lookahead_point(x, y, theta)
        distances = np.linalg.norm(waypoints - lp, axis=1)
        target_idx = np.argmin(distances)
        for i in range(target_idx, len(waypoints)):
            if np.linalg.norm(waypoints[i] - lp) >= LOOKAHEAD_DIST:
                target_idx = i
                break

        target_pos = waypoints[target_idx]
        direction = target_pos - lp
        dist_to_target = np.linalg.norm(direction)
        u_des = direction / dist_to_target if dist_to_target > 0.01 else np.zeros(2)

        current_alpha = BASE_ALPHA
        if abs(v) < V_THRESH and dist_to_target > D_THRESH:
            current_alpha = ALPHA_FREEZE
            perp_vec = np.array([-direction[1], direction[0]])
            if np.linalg.norm(perp_vec) > 0:
                perp_vec = perp_vec / np.linalg.norm(perp_vec)
                u_des = u_des + V_SWERVE * perp_vec

        x_norm = (state6 - x_mean) / x_std
        x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)
        B_val = B_net(img_torch, x_torch)

        grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
        grad_B_pos = (grad_B_norm / x_std)[0:2]
        grad_B_pos = np.clip(grad_B_pos, -5.0, 5.0)

        conservative_B = B_val.item() - BUFFER
        A_list, b_list = [grad_B_pos], [-current_alpha * conservative_B]
        u_safe = cbf_qp_multi(u_des, A_list, b_list)

        v_cmd, omega_cmd = unicycle_from_planar_velocity(u_safe, theta)
        obs = env.step((v_cmd, omega_cmd))

        min_dist = get_2d_min_distance(pos, dynamic_coords)
        dist_log.append(min_dist)
        speed_log.append(abs(v))

        if step % 25 == 0:
            print(f"Step {step:04d} | B(x): {B_val.item():.4f} | dist: {min_dist:.3f} | "
                  f"u_des: {np.round(u_des,3)} | u_safe: {np.round(u_safe,3)}")

        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break

        if np.linalg.norm(pos - GOAL_POS) < 0.15:
            print(f"Reached goal at step {step}.")
            break
        if min_dist <= -0.05:
            print(f"Collision at step {step}.")
            break

    if viewer is not None:
        viewer.close()

    print("Generating trajectory plot...")
    plot_2d_dynamic_trajectory(
        trajectory=trajectory, obstacle_trajectory=obstacle_trajectory,
        start_pos=START_POS, goal_pos=GOAL_POS, static_cylinder_coords=static_coords,
        save_path=PROJECT_ROOT / "results" / "single_dynamic_trajectory.png",
    )
    print("Generating time-series plot...")
    plot_time_series(dist_log, speed_log,
                      save_path=PROJECT_ROOT / "results" / "single_dynamic_timeseries.png")

    env.close()


if __name__ == "__main__":
    main()