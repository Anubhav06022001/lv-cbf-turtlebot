# VOCBF



### 1. `static_test.py` (Post-QP Anti-Chatter Sign-Locking)
* **The Problem:** The nominal base controller constantly pulls the robot toward the goal, while the QP solver pushes it away from the perfectly circular safety boundary. This tug-of-war causes the QP solver to rapidly oscillate (chatter) between positive and negative angular velocities.
* **The Fix:** This script implements a post-QP sign-lock. Upon entering the unsafe boundary ($B < 0.0$), it records the initial sign of the commanded rotation and takes the absolute value of all subsequent QP angular outputs. It mandates a minimum rotation speed of $0.8$ rad/s to shatter static friction, forcing the robot to aggressively commit to dodging left or right.

### 2. `static_test_goal_amnesia.py` (Pre-QP Dynamic Base Controller)
* **The Problem:** State-based CBFs suffer from the "Moth to a Flame" local minimum, where the robot clears the obstacle, immediately turns back toward the goal, and gets trapped in an infinite loop.
* **The Fix:** This script fixes the problem *before* the QP solver. When approaching an obstacle ($B < 0.15$), the base controller temporarily ignores the goal position ("Goal Amnesia"). The intent $u_{des}$ is dynamically overwritten to force a smooth, forward-moving arc. By aligning the base controller's intent with the required evasion maneuver, the QP solver acts as a smooth safety filter rather than fighting the base controller.

## Automated Recording Scripts
These scripts run the simulation headless (without a live viewer) to generate 60 FPS MP4 video demonstrations of the various mathematical behaviors.

* **`record_failures.py`:** Generates three videos demonstrating classic CBF failures:
  1. `1_stiction_failure.mp4`: The QP drops $v \to 0$, but the allowed $\omega$ torque fails to break MuJoCo static friction, resulting in permanent deadlock.
  2. `2_hysteresis_failure.mp4`: The strict threshold ($B(x) < 0.00$) traps the robot hovering permanently at $B(x) \approx 0.006$ without triggering recovery.
  3. `3_semicircle_failure.mp4`: Relying on the pure neural network angular gradient ($L_{g}B_{\omega}$) near a circular boundary causes numerical noise, commanding an arc directly into the obstacle.

* **`record_success.py`:** Generates `4_success_anti_chatter.mp4`. This headless script mirrors `static_test.py` to record a successful traversal of the maze using the Anti-Chatter sign-locking logic ($\alpha = 40.0$, safety offset $= 0.05$). While successful, the evasion maneuvers highlight the inefficiency of state-only models compared to vision-guided trajectories.