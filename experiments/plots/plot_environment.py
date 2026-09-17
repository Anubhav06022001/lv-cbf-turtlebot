import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

plot_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plot_dir, exist_ok=True)

# Configuration matches your XML and Robot
ROBOT_RADIUS = 0.10
CYLINDER_RADIUS = 0.10
SAFE_MARGIN = 0.0  # Boundary where signed distance = 0

# Cylinder Centers from XML
cylinders = [(1.0, 0.5), (1.0, -0.5),(2.0, 0.0), (2.0, 1.0), (2.0, -1.0),(3.0, 0.5), (3.0, -0.5)]

# Generate grid
x = np.linspace(0, 4.5, 500)
y = np.linspace(-1.5, 1.5, 500)
X, Y = np.meshgrid(x, y)

# Calculate minimum signed distance across the entire grid
Z = np.full_like(X, np.inf)
for cx, cy in cylinders:
    # Distance to center minus cylinder radius minus robot radius
    dist = np.hypot(X - cx, Y - cy) - CYLINDER_RADIUS - ROBOT_RADIUS
    Z = np.minimum(Z, dist)

# --- Plotting ---
# IEEE style formatting
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(8, 5))

# Create contour map (Red = Negative/Unsafe, Green = Positive/Safe)
# vmin and vmax lock the color scaling so 0 is exactly in the middle (white/yellow transition)
contour = ax.contourf(X, Y, Z, levels=50, cmap='RdYlGn', vmin=-0.5, vmax=0.5, alpha=0.8)

# Add a distinct solid black line at the exact safety boundary (distance = 0)
ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2, linestyles='dashed')

# Draw the physical cylinders as solid gray circles for reference
for cx, cy in cylinders:
    circle = patches.Circle((cx, cy), CYLINDER_RADIUS, facecolor='dimgray', edgecolor='black', zorder=3)
    ax.add_patch(circle)

# Draw the Start and Goal
ax.plot(0.0, 0.0, 'bo', markersize=10, label='Start Region (Approx)', zorder=4)
ax.plot(4.0, 0.0, 'b*', markersize=15, label='Goal', zorder=4)

# Formatting
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Signed Distance Field [m]')
ax.set_title('Environment Safety Map (LV-CBF Ground Truth)')
ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.set_xlim(0, 4.5)
ax.set_ylim(-1.5, 1.5)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower left')

plt.tight_layout()
# plt.savefig('environment_safety_map.png', dpi=300)
plt.savefig(os.path.join(plot_dir, "environment_safety_map.png"), dpi=300, bbox_inches='tight')
print("Saved IEEE-ready plot to 'environment_safety_map.png'")
plt.show()