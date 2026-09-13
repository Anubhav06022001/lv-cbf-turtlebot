"""Same semantics as the drone code's check_safety / check_reachability."""
import numpy as np

D_SAFE = 0.05          # [m] matches d_safe used in training script
GOAL_RADIUS = 0.15     # [m] terminal acceptance radius, matches the paper (Sec. V-A)


def check_safety(min_episode_dist, d_safe=D_SAFE):
    """True if clearance never dropped below the safety threshold."""
    return min_episode_dist >= d_safe


def check_reachability(pos_xy, goal_xy, goal_radius=GOAL_RADIUS):
    pos_xy = np.asarray(pos_xy, dtype=np.float32)[:2]
    goal_xy = np.asarray(goal_xy, dtype=np.float32)[:2]
    return np.linalg.norm(pos_xy - goal_xy) <= goal_radius
