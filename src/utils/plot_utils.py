import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

DEFAULT_CYLINDER_COORDS = [
    (1.0, 0.5), (1.0, -0.5),
    (2.0, 1.0), (2.0, -1.0), (2.0, 0.0), 
    (3.0, 0.5), (3.0, -0.5)
]

def plot_2d_trajectory(trajectory, start_pos=(0.0, 0.0), goal_pos=(4.0, 0.0), cylinder_coords=DEFAULT_CYLINDER_COORDS):
    traj = np.array(trajectory)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    # 1. Obstacles (Slate Gray to match video)
    for i, (cx, cy) in enumerate(cylinder_coords):
        circle = plt.Circle((cx, cy), 0.12, facecolor='lightslategray', edgecolor='black', 
                            linewidth=1.5, zorder=2, label='Obstacle' if i == 0 else None)
        ax.add_patch(circle)

    # 2. Start Position (Gold Star)
    ax.scatter(*start_pos, marker='*', s=300, color='gold', edgecolor='black', linewidth=1.2, zorder=5, label='Start')
    
    # 3. Goal Position (Green Dot)
    ax.scatter(*goal_pos, marker='o', s=120, color='tab:green', edgecolor='black', linewidth=1.2, zorder=5, label='Goal')

    # 4. Trajectory (Thick Crimson Line to match video)
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], color='crimson', linewidth=3.0, label='Trajectory', zorder=4)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.4)

    # Cleaner axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend = ax.legend(frameon=True, fontsize=11, loc='upper right')

    # Bold ticks
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    # plt.savefig("trajectory_icra_style.pdf", bbox_inches='tight', dpi=300)
    plt.show()
