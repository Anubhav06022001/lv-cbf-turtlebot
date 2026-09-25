import numpy as np

# --- Simulation & Hardware Constants ---
simend = 150.0
WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
ROBOT_RADIUS = 0.10

# --- CBF & Controller Constants ---
ALPHA = 40.0
SAFETY_OFFSET = 0.05
LOOKAHEAD_DIST = 0.45

# --- Environment Constants ---
GOAL_POS = np.array([4.0, 0.0])
CYLINDER_COORDS = [
    (1.0, 0.5), (1.0, -0.5), (2.0, 0.0), 
    (2.0, 1.0), (2.0, -1.0), (3.0, 0.5), (3.0, -0.5)
]

# --- Helper Functions ---
def quat2yaw(q):
    """Converts a quaternion [w, x, y, z] to a 2D yaw angle."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def get_min_dist(pos_2d):
    """Calculates the physical distance to the nearest cylinder obstacle."""
    min_dist = float('inf')
    for cx, cy in CYLINDER_COORDS:
        dist = np.hypot(pos_2d[0] - cx, pos_2d[1] - cy) - 0.1 - ROBOT_RADIUS
        if dist < min_dist: 
            min_dist = dist
    return min_dist