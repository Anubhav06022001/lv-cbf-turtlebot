"""
MuJoCo environment wrapper for the TurtleBot3 Waffle Pi, playing the same
role that `VisionAviary` (gym-pybullet-drones) played for the Crazyflie --
now written against your actual scene_turtlebot3_waffle_pi.xml.

Observation is deliberately just (x, y, yaw), per your spec -- no velocity
terms in the state fed to the barrier network. Wheel actuators are velocity
actuators (also per your spec): step() takes (left_wheel_vel, right_wheel_vel)
directly in rad/s, no unicycle inversion needed inside the env itself (the
planar-velocity -> wheel-velocity inversion for CBF tracking lives in
src/control/unicycle_controller.py and is applied by the calling script).

*** Run scripts/utils/inspect_xml.py first and check the auto-detected
names below against its output. ***
"""
import numpy as np
import mujoco

# ============================================================================
# CONFIG
# ============================================================================
# Your scene includes `turtlebot3_waffle_pi.xml`, whose internal joint/
# actuator/camera names aren't visible from here. The free joint is
# auto-detected (there should be exactly one in the whole model: the robot
# base -- the obstacle cylinders in your XML are plain static geoms with no
# joint at all). Wheel actuators and the camera are located by name
# substring match; if that fails, set the *_OVERRIDE constants below to the
# exact names printed by scripts/utils/inspect_xml.py.
WHEEL_LEFT_ACT_OVERRIDE = "wheel_left"    # exact name from turtlebot3_waffle_pi.xml
WHEEL_RIGHT_ACT_OVERRIDE = "wheel_right"  # exact name from turtlebot3_waffle_pi.xml
CAMERA_NAME_OVERRIDE = "wrist_camera"     # only camera in the model

# NOTE: wrist_camera is mounted at pos="0.08 0 0.15" euler="0 30 29.9" -- i.e.
# pitched down ~30 deg AND yawed ~30 deg off the body's forward (+x) axis, not
# a plain forward-facing camera. The controller still drives the robot's BODY
# heading toward the direction of travel (see unicycle_controller.py); the
# camera's boresight will therefore be offset from that direction by the
# yaw component (~30 deg) at all times. If obstacle-centering in the image
# matters for your results, either fix this in the XML (zero the yaw
# component) or compensate by biasing the heading target by
# CAMERA_YAW_OFFSET_RAD -- see the note in unicycle_controller.py.
CAMERA_YAW_OFFSET_RAD = np.radians(29.9)

CONTROL_FREQ_HZ = 50   # outer-loop safety-filter frequency (matches the paper)
SIM_TIMESTEP = None    # None -> use the XML's <option timestep=".."/>

# Obstacle geoms as they literally appear in your XML.
OBSTACLE_GEOM_NAMES = [f"obstacle{i}" for i in range(1, 8)]  # obstacle1 .. obstacle7
# ============================================================================


def _find_actuator_by_substrings(model, must_contain_all):
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        low = nm.lower()
        if all(s in low for s in must_contain_all):
            return i, nm
    return -1, None


class TurtleBotMuJoCoEnv:
    """
    obs["0"] = {
        "pose":  np.ndarray (3,)   # [x, y, yaw]   <- fed to the barrier network
        "rgb":   np.ndarray (H, W, 3) uint8
        "dep":   np.ndarray (H, W)  float32, metres
    }
    """

    def __init__(self, xml_path, render_width=64, render_height=64, gui=False):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        if SIM_TIMESTEP is not None:
            self.model.opt.timestep = SIM_TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.gui = gui
        self.render_width = render_width
        self.render_height = render_height

        self.TIMESTEP = self.model.opt.timestep
        self.CONTROL_FREQ_HZ = CONTROL_FREQ_HZ
        self._substeps = max(1, int(round(1.0 / (self.CONTROL_FREQ_HZ * self.TIMESTEP))))

        # --- Free joint (robot base) ---
        free_joints = [i for i in range(self.model.njnt) if int(self.model.jnt_type[i]) == 0]
        if len(free_joints) != 1:
            raise ValueError(
                f"Expected exactly 1 free joint (the robot base), found {len(free_joints)}. "
                "Run scripts/utils/inspect_xml.py and check your model -- if the base uses "
                "slide+hinge joints instead of a free joint, this wrapper needs a small edit."
            )
        self._free_jnt_id = free_joints[0]
        self._qpos_adr = self.model.jnt_qposadr[self._free_jnt_id]
        self._qvel_adr = self.model.jnt_dofadr[self._free_jnt_id]

        # --- Wheel actuators ---
        if WHEEL_LEFT_ACT_OVERRIDE is not None:
            self._act_left = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEEL_LEFT_ACT_OVERRIDE)
        else:
            self._act_left, found_left = _find_actuator_by_substrings(self.model, ["left", "wheel"])
        if WHEEL_RIGHT_ACT_OVERRIDE is not None:
            self._act_right = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEEL_RIGHT_ACT_OVERRIDE)
        else:
            self._act_right, found_right = _find_actuator_by_substrings(self.model, ["right", "wheel"])

        if -1 in (self._act_left, self._act_right):
            raise ValueError(
                "Could not auto-detect left/right wheel actuators (looked for names "
                "containing 'left'+'wheel' / 'right'+'wheel'). Run "
                "scripts/utils/inspect_xml.py and set WHEEL_LEFT_ACT_OVERRIDE / "
                "WHEEL_RIGHT_ACT_OVERRIDE at the top of this file."
            )

        # --- Camera ---
        if CAMERA_NAME_OVERRIDE is not None:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME_OVERRIDE)
        elif self.model.ncam == 1:
            cam_id = 0
        else:
            cam_id = -1
        if cam_id == -1:
            raise ValueError(
                f"Model has {self.model.ncam} cameras -- can't auto-pick one. Run "
                "scripts/utils/inspect_xml.py and set CAMERA_NAME_OVERRIDE at the top "
                "of this file."
            )
        self._camera_id = cam_id

        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)

        # --- Obstacle geoms (static, no joint -- repositioned via model.geom_pos) ---
        self._obstacle_geom_ids = []
        for name in OBSTACLE_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self._obstacle_geom_ids.append(gid)
        if not self._obstacle_geom_ids:
            raise ValueError(
                f"None of {OBSTACLE_GEOM_NAMES} found as geoms in the XML. "
                "Check OBSTACLE_GEOM_NAMES matches your scene."
            )

    # ------------------------------------------------------------------ #
    # Core lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, start_xytheta):
        """start_xytheta: (x, y, theta) in world frame."""
        mujoco.mj_resetData(self.model, self.data)
        x, y, theta = start_xytheta
        quat = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])  # yaw-only quat (w,x,y,z)
        self.data.qpos[self._qpos_adr:self._qpos_adr + 3] = [x, y, 0.0]
        self.data.qpos[self._qpos_adr + 3:self._qpos_adr + 7] = quat
        self.data.qvel[self._qvel_adr:self._qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, wheel_vel_left_right):
        """wheel_vel_left_right: (omega_left, omega_right) in rad/s, sent
        directly to the velocity actuators."""
        wl, wr = wheel_vel_left_right
        for _ in range(self._substeps):
            self.data.ctrl[self._act_left] = wl
            self.data.ctrl[self._act_right] = wr
            mujoco.mj_step(self.model, self.data)
        return self._get_obs()

    def close(self):
        self.renderer.close()

    # ------------------------------------------------------------------ #
    # Obstacle placement -- direct geom_pos edit (these geoms have no joint,
    # so they're rigidly fixed to the world; editing model.geom_pos before
    # mj_forward/mj_step relocates them exactly like a mocap body would).
    # ------------------------------------------------------------------ #
    def set_obstacle_positions(self, xy_list, z=0.0):
        for (x, y), gid in zip(xy_list, self._obstacle_geom_ids):
            self.model.geom_pos[gid] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)

    @property
    def num_obstacles(self):
        return len(self._obstacle_geom_ids)

    def set_goal_marker(self, xy, z=0.05):
        pass  # no goal-marker geom in this scene; purely cosmetic, safe to no-op

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _get_pose(self):
        pos = self.data.qpos[self._qpos_adr:self._qpos_adr + 3]
        quat = self.data.qpos[self._qpos_adr + 3:self._qpos_adr + 7]  # w,x,y,z
        w, xq, yq, zq = quat
        theta = np.arctan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))
        return np.array([pos[0], pos[1], theta], dtype=np.float32)

    def get_planar_velocity(self):
        """Forward speed and yaw rate -- NOT part of the barrier network's
        state, only used by the controller (e.g. the adaptive-swerve
        stagnation check)."""
        theta = self._get_pose()[2]
        lin_vel_world = self.data.qvel[self._qvel_adr:self._qvel_adr + 3]
        omega = self.data.qvel[self._qvel_adr + 5]
        v = np.cos(theta) * lin_vel_world[0] + np.sin(theta) * lin_vel_world[1]
        return float(v), float(omega)

    def _get_obs(self):
        pose = self._get_pose()

        self.renderer.update_scene(self.data, camera=self._camera_id)
        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self._camera_id)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        return {"0": {"pose": pose, "rgb": rgb, "dep": depth}}
