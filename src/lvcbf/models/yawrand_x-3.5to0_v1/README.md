# Run: yawrand_x-3.5to0_v1

- Dataset: dataset_waypoint_ghosts_yaw_random.h5
- Data collection: 500 episodes, MAX_STEPS=2000, start x in [-3.5, 0], y in [-2.5, 2.5], random yaw, waypoint controller, ghost cylinders, GLFW 64x64 render
- Training: gamma=0.5, tau=0.8, lr=0.0003, epochs=30, d_safe=0.05, batch_size=128
- Anchors: unsafe (d<0.05) weight 0.05, safe (d>0.4) weight 0.1
- Files: ckpt_epochNNN.pt (every 5 epochs), ckpt_latest.pt (resume), tb3_x_mean.npy / tb3_x_std.npy (state normalization), tb3_lvcbf_yawrand_x-3.5to0_v1_final.pt (final weights)
