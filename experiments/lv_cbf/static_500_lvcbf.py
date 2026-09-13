"""
Quantitative static-obstacle benchmark for LV-CBF on the TurtleBot3 Waffle Pi,
mirroring your drone test_static.py (Table I in the paper).

Differences from the drone version:
  - u_des / u_safe are planar velocities of the look-ahead point; they are
    inverted to (v, omega) via unicycle_from_planar_velocity() before being
    sent to env.step().
  - grad_B_pos is evaluated w.r.t. the position channels of the 6D state
    (x, y == indices 0, 1) instead of the drone's 12D state's first 3.
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.env_setup import init_drone_env, XML_PATH
init_drone_env()

from src.utils.turtlebot_mujoco_env import TurtleBotMuJoCoEnv
from src.utils.env_utils import get_2d_min_distance, CYLINDER_COORDS, place_static_obstacles, place_goal_marker
from src.utils.eval_metrics import check_safety, check_reachability
from src.utils.process_image import process_image
from src.vocbf.vision_cbf_network import VisionBarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi
from src.control.unicycle_controller import (
    lookahead_point, unicycle_from_planar_velocity, LOOKAHEAD_L,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dir = PROJECT_ROOT / "models" / "vision_vocbf"
x_mean = np.load(models_dir / "vis_x_mean.npy")
x_std = np.load(models_dir / "vis_x_std.npy")

B_net = VisionBarrierNet().to(DEVICE)
checkpoint = torch.load(models_dir / "vision_vocbf_turtlebot.pt", map_location=DEVICE)
B_net.load_state_dict(checkpoint["model_state_dict"])
B_net.eval()


def main():
    csv_path = PROJECT_ROOT / "data" / "intial_pos" / "start_positions_turtlebot_500.csv"
    df = pd.read_csv(csv_path)  # expects columns: start_x, start_y, start_theta
    EPISODES = len(df)

    env = TurtleBotMuJoCoEnv(XML_PATH, render_width=64, render_height=64, gui=False)
    place_static_obstacles(env, CYLINDER_COORDS)

    goal_pos = np.array([4.0, 0.0])
    place_goal_marker(env, goal_pos)
    LOOKAHEAD_DIST = 1.2
    alpha = 0.5
    DT = 1.0 / env.CONTROL_FREQ_HZ

    safe_count, reach_count = 0, 0
    print(f"Starting Quantitative Evaluation on Benchmark Dataset: {EPISODES} Episodes...")

    for ep in range(EPISODES):
        start_x, start_y = df["start_x"][ep], df["start_y"][ep]
        start_theta = df["start_theta"][ep] if "start_theta" in df.columns else 0.0
        start_pos = np.array([start_x, start_y])

        dist = np.linalg.norm(goal_pos - start_pos)
        n = max(2, int(dist / 0.05))
        waypoints = np.linspace(start_pos, goal_pos, n, endpoint=True)

        obs = env.reset((start_x, start_y, start_theta))

        min_episode_dist = float("inf")
        reached = False

        for step in range(10000):
            state6 = obs["0"]["state6"]
            x, y, cth, sth, v, omega = state6
            theta = np.arctan2(sth, cth)
            pos = np.array([x, y])

            current_dist = get_2d_min_distance(pos, CYLINDER_COORDS)
            min_episode_dist = min(min_episode_dist, current_dist)

            if check_reachability(pos, goal_pos):
                reached = True
                break

            lp = lookahead_point(x, y, theta)
            distances = np.linalg.norm(waypoints - lp, axis=1)
            closest_idx = np.argmin(distances)
            target_idx = closest_idx
            for i in range(closest_idx, len(waypoints)):
                if np.linalg.norm(waypoints[i] - lp) >= LOOKAHEAD_DIST:
                    target_idx = i
                    break

            target_pos = waypoints[target_idx]
            direction = target_pos - lp
            dist_to_target = np.linalg.norm(direction)

            if dist_to_target > 0.01:
                speed_scale = min(dist_to_target * 2.0, 1.0)
                u_des = (direction / dist_to_target) * speed_scale
            else:
                u_des = np.zeros(2)

            img_torch = process_image(obs["0"], device=DEVICE)

            x_norm = (state6 - x_mean) / x_std
            x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)

            B_val = B_net(img_torch, x_torch)
            grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
            grad_B_pos = (grad_B_norm / x_std)[0:2]   # d/dx, d/dy (indices 0,1 of the 6D state)
            grad_B_pos = np.clip(grad_B_pos, -5.0, 5.0)

            A_list, b_list = [grad_B_pos], [-alpha * B_val.item()]
            u_safe = cbf_qp_multi(u_des, A_list, b_list)

            v_cmd, omega_cmd = unicycle_from_planar_velocity(u_safe, theta)

            if not check_safety(min_episode_dist):
                break

            obs = env.step((v_cmd, omega_cmd))

        is_safe = check_safety(min_episode_dist)
        if is_safe:
            safe_count += 1
        if reached:
            reach_count += 1

        if (ep + 1) % 25 == 0:
            print(f"[{ep + 1}/{EPISODES}] Safety Rate: {(safe_count/(ep+1))*100:.1f}% | "
                  f"Reachability Rate: {(reach_count/(ep+1))*100:.1f}%")

    print("\n" + "=" * 40)
    print("FINAL BENCHMARK RESULTS (TurtleBot3 Waffle Pi, Static)")
    print("=" * 40)
    print(f"Safety Rate: {(safe_count/EPISODES)*100:.2f}%")
    print(f"Reachability Rate: {(reach_count/EPISODES)*100:.2f}%")
    print("=" * 40)
    env.close()


if __name__ == "__main__":
    main()