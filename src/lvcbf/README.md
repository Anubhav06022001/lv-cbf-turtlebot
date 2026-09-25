dataset_waypoint_ghosts.h5: Training data capturing safe/unsafe visual transitions by driving the robot through phantom (ghost) cylinders using pure waypoint tracking.

tb3_lvcbf_apf_final.pt: Legacy Vision-CBF model trained purely on Artificial Potential Field (APF) collision-avoidance data.

tb3_lvcbf_balanced_forest.pt: Latest Vision-CBF model trained on ghost data with balanced anchor weights and a 0.4m safe-zone threshold optimized for dense environments.