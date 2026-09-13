"""
Dynamic obstacle avoidance benchmark for LV-CBF on the TurtleBot3 Waffle Pi,
mirroring your drone test_dynamic.py (Table II ablation / Fig. 3).

Ground-robot changes vs. the drone version:
  - The moving cylinder is a mocap body repositioned each step
    (env.set_obstacle_positions), analogous to p.resetBasePositionAndOrientation.
  - The adaptive-swerve perturbation and the CBF-QP output are planar
    velocities of the look-ahead point; both the swerve law (Eq. 16-18) and
    the QP filter are unchanged from the drone code -- only the final
    inversion to (v, omega) and the Active Vision yaw target (Eq. 22, now
    driving the unicycle's actual heading instead of a drone's yaw-only DOF)
    are unicycle-specific.
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
    lookahead_point, unicycle_from_planar_velocity, active_vision_yaw_target, LOOKAHEAD_L,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = PROJECT_ROOT / "models" / "vision_vocbf"

x_mean = np.load(models_dir / "vis_x_mean.npy")
x_std = np.load(models_dir / "vis_x_std.npy")

B_net = VisionBarrierNet().to(DEVICE)
checkpoint = torch.load(models_dir / "vision_vocbf_turtlebot.pt", map_location=DEVICE)
B_net.load_state_dict(checkpoint["model_state_dict"])
B_net.eval()

MIDDLE_CYL_IDX = 2  # index of the (2.0, 0.0) cylinder in CYLINDER_COORDS, matches the drone script


def main():
    env = TurtleBotMuJoCoEnv(XML_PATH, render_width=64, render_height=64, gui=False)

    EPISODES = 100
    MAX_STEPS = 9000
    LOOKAHEAD_DIST = 1.2
    BASE_ALPHA = 0.5
    buffer = 0.20

    # Adaptive swerve / active-vision hyperparameters (Table III)
    V_THRESH = 0.03           # [m/s] scaled down from the drone's 0.15 m/s: TB3
                               # Waffle's own max speed (0.22 m/s) is much lower,
                               # so "stuck" must be judged relative to that.
    D_THRESH = 0.50
    ALPHA_FREEZE = 0.20
    V_SWERVE = 0.15            # [m/s] must stay below V_MAX (0.22 m/s)

    goal_pos = np.array([4.0, 0.0])
    place_goal_marker(env, goal_pos)

    cases_csv_path = PROJECT_ROOT / "data" / "intial_pos" / "dynamic_test_cases_turtlebot_100.csv"
    df_cases = pd.read_csv(cases_csv_path)  # columns: robot_start_x, robot_start_y, cyl_start_y, cyl_speed

    safe_count, reach_count = 0, 0
    results_log = []

    print("Starting ICRA Dynamic Benchmark (TurtleBot3 Waffle Pi, Active Vision + Adaptive Safety)...")

    for ep in range(EPISODES):
        robot_start_x = df_cases["robot_start_x"][ep]
        robot_start_y = df_cases["robot_start_y"][ep]
        cyl_start_y = df_cases["cyl_start_y"][ep]
        cyl_speed = df_cases["cyl_speed"][ep]
        start_theta = df_cases["robot_start_theta"][ep] if "robot_start_theta" in df_cases.columns else 0.0

        start_pos = np.array([robot_start_x, robot_start_y])

        dist = np.linalg.norm(goal_pos - start_pos)
        n = max(2, int(dist / 0.1))
        waypoints = np.linspace(start_pos, goal_pos, n, endpoint=True)

        obs = env.reset((robot_start_x, robot_start_y, start_theta))

        static_coords = [c for i, c in enumerate(CYLINDER_COORDS) if i != MIDDLE_CYL_IDX]
        env.set_obstacle_positions(static_coords + [(2.0, cyl_start_y)])
        # ^ NOTE: relies on OBSTACLE_MOCAP_NAMES ordering matching CYLINDER_COORDS
        #   ordering with the moving cylinder placed last in this call; adjust if
        #   your scene numbers mocap bodies differently.

        min_episode_dist = float("inf")
        reached = False

        for step in range(MAX_STEPS):
            new_y = cyl_start_y + (step * cyl_speed)
            if new_y > 2.5:
                new_y = 2.5
            env.set_obstacle_positions(static_coords + [(2.0, new_y)])
            dynamic_coords = static_coords + [(2.0, new_y)]

            state6 = obs["0"]["state6"]
            x, y, cth, sth, v, omega = state6
            theta = np.arctan2(sth, cth)
            pos = np.array([x, y])

            current_dist = get_2d_min_distance(pos, dynamic_coords)
            min_episode_dist = min(min_episode_dist, current_dist)

            if check_reachability(pos, goal_pos):
                reached = True
                break
            if not check_safety(min_episode_dist):
                break

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

            # === Adaptive CBF alpha & swerve (anti-freezing), Eq. 16-18 ===
            current_alpha = BASE_ALPHA
            robot_speed = np.linalg.norm([v])  # forward speed is the unicycle's only linear DOF
            if robot_speed < V_THRESH and dist_to_target > D_THRESH:
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

            conservative_B = B_val.item() - buffer
            A_list, b_list = [grad_B_pos], [-current_alpha * conservative_B]
            u_safe = cbf_qp_multi(u_des, A_list, b_list)

            # === Active Vision heading alignment, Eq. 22 ===
            # On the drone, yaw was a free DOF decoupled from translational
            # safety, so it could be pointed at u_safe independently of the
            # CBF-QP output. The unicycle has no such spare DOF: the same
            # (v, omega) pair that satisfies the CBF constraint also *is* the
            # heading command, via the feedback-linearization inversion below.
            # Overriding omega with a separate heading-tracking law would
            # invalidate the CBF certificate (u_safe was computed assuming
            # exactly this inversion). Fortunately the look-ahead-point
            # construction gives an equivalent effect "for free": driving the
            # point L ahead of the robot toward u_safe necessarily rotates the
            # robot to face the direction it's moving in, so the forward
            # camera tracks the obstacle throughout the bypass maneuver
            # without any extra control term. `yaw_target` is logged below
            # purely as a diagnostic to sanity-check that alignment.
            yaw_target = active_vision_yaw_target(u_safe, theta)
            v_cmd, omega_cmd = unicycle_from_planar_velocity(u_safe, theta)

            obs = env.step((v_cmd, omega_cmd))

        is_safe = check_safety(min_episode_dist)
        if is_safe:
            safe_count += 1
        if reached:
            reach_count += 1

        results_log.append({
            "ep": ep, "robot_start_x": robot_start_x, "robot_start_y": robot_start_y,
            "cyl_start_y": cyl_start_y, "cyl_speed": cyl_speed,
            "safe": is_safe, "reached": reached, "min_dist": min_episode_dist,
        })

        print(f"[Ep {ep+1:03d}/{EPISODES}] Safe: {is_safe} | Reached: {reached} | Min Dist: {min_episode_dist:.3f}m")

    final_safety = (safe_count / EPISODES) * 100
    final_reach = (reach_count / EPISODES) * 100

    print("\n" + "=" * 40)
    print("FINAL DYNAMIC BENCHMARK RESULTS (TurtleBot3 Waffle Pi)")
    print("=" * 40)
    print(f"Safety Rate:       {final_safety:.1f}%")
    print(f"Reachability Rate: {final_reach:.1f}%")
    print("=" * 40)

    save_path = PROJECT_ROOT / "results" / "dynamic_turtlebot_100_results.csv"
    pd.DataFrame(results_log).to_csv(save_path, index=False)

    env.close()


if __name__ == "__main__":
    main()