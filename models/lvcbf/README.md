# TurtleBot3 CBF Models & Data Tracking

### Datasets
* **`dataset_waypoint_ghosts.h5`**: Core training dataset. Collected by driving the robot through pass-through "ghost" cylinders using waypoint tracking to capture safe/unsafe visual transitions without physical collision terminations.

### State-Based Models (Baseline)
* **`tb3_vocbf.pt`**: State-only (non-vision) baseline Control Barrier Function model. Relies strictly on `[x, y, yaw, v, w]` state vectors and known obstacle coordinates.

### Vision-Language-Action Models (LV-CBF)
* **`tb3_lvcbf_apf_final.pt`**: Early vision model trained purely on Artificial Potential Field (APF) avoidance data.

* **`tb3_lvcbf_waypoint_ghost_final.pt`**: Initial vision model trained on ghost data. Suffered from conservative boundaries due to dense forest constraints.

* **`tb3_lvcbf_balanced_forest_final.pt`**: Tuned model with balanced safe/unsafe anchor weights (0.05 unsafe / 0.08 safe) for a 0.4m safe threshold. Enabled smooth navigation between dense cylinders.

* **`tb3_lvcbf_balanced_forest_3.pt`**: Latest model. Trained identically to the balanced forest model, but with the green `goal_marker` removed from the MuJoCo XML to eliminate Out-of-Distribution (OOD) visual artifacts at the target destination.