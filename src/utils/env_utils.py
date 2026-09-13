"""
Ground-robot analogues of the PyBullet drone helpers
`get_2d_min_distance`, `inject_fixed_obstacles`, `inject_goal_marker`.

MuJoCo scenes are defined statically in XML, so instead of *injecting* new
bodies at runtime (as PyBullet's `p.createMultiBody` did), the 7 cylinders
already exist in scene_turtlebot3_waffle_pi.xml as static geoms
("obstacle1".."obstacle7"); these helpers just (re)position them via
TurtleBotMuJoCoEnv.set_obstacle_positions().
"""
import numpy as np

CYLINDER_RADIUS = 0.1  # [m] matches `size="0.1 0.5"` in scene_turtlebot3_waffle_pi.xml

# Same layout as the drone's CYLINDER_COORDS, reused for the static benchmark.
CYLINDER_COORDS = [
    (1.0, 0.5), (1.0, -0.5), (2.0, 0.0), (2.0, 1.0),
    (2.0, -1.0), (3.0, 0.5), (3.0, -0.5),
]


def get_2d_min_distance(pos_xy, cylinder_coords, cylinder_radius=CYLINDER_RADIUS):
    """Minimum clearance (surface distance, can go negative on penetration)
    from `pos_xy` to the nearest cylinder in `cylinder_coords`."""
    pos_xy = np.asarray(pos_xy, dtype=np.float32)
    min_d = float("inf")
    for cx, cy in cylinder_coords:
        d = np.linalg.norm(pos_xy - np.array([cx, cy])) - cylinder_radius
        min_d = min(min_d, d)
    return min_d


def place_static_obstacles(env, cylinder_coords=CYLINDER_COORDS):
    """Analogue of inject_fixed_obstacles for the static benchmark: moves the
    scene's pre-existing obstacle mocap bodies to the benchmark layout."""
    env.set_obstacle_positions(cylinder_coords)
    return list(range(len(cylinder_coords)))


def place_goal_marker(env, goal_xy):
    env.set_goal_marker(goal_xy)
