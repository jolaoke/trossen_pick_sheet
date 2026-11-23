"""Paper-picking environment for reinforcement learning with the compliant-gripper WidowX AI.

Control stack (each ``step``):

    policy action (25 Hz) -> end-effector target pose + grip command
        -> OSC torques + grip-force feedback (500 Hz, every physics step)

The API follows Gymnasium (``reset`` -> ``(obs, info)``, ``step`` -> ``(obs, reward, terminated,
truncated, info)``) but doesn't require it; ``observation_space``/``action_space`` are Gymnasium
``Box`` spaces when Gymnasium is installed and ``(low, high)`` arrays otherwise.

Action (5 values in [-1, 1]):
    0-2  end-effector target displacement in x, y, z (scaled by ``max_step_pos`` meters)
    3    rotation of the target about world z (scaled by ``max_step_yaw`` radians)
    4    gripper: > 0 close (grip-force feedback), <= 0 open

Usage:
    from trossen_arm_mujoco.paper_env import PaperPickEnv
    env = PaperPickEnv()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.sample_action())
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from trossen_arm_mujoco.src.osc import (
    DOWNWARD_QUAT,
    GripController,
    OSCController,
    use_torque_actuators,
)

try:
    import gymnasium
except ImportError:
    gymnasium = None

SCENE_XML_PATH = Path(__file__).parent / "assets" / "wxai" / "scene_wxai_paper.xml"

# Arm posture at reset: bent elbow, gripper ~10 cm above the paper (the straight-up home pose is
# a kinematic singularity, which OSC can't start from).
START_QPOS = np.array([0.0, 1.0, 0.8, -0.2, 0.0, 0.0])

# End-effector target limits [x, y, z] (m). z can go slightly below the table so the policy can
# press the fingertips onto it.
WORKSPACE_LOW = np.array([0.15, -0.25, -0.005])
WORKSPACE_HIGH = np.array([0.50, 0.25, 0.30])

SUCCESS_HEIGHT = 0.05  # paper mean height above the table (m) that counts as picked up


class PaperPickEnv:
    """Pick a sheet of paper off the table.

    :param scene_xml: Scene to load
    :param control_hz: Policy rate; the OSC and physics run at the model's timestep
    :param max_steps: Episode length in policy steps
    :param max_step_pos: Max target displacement per policy step (m)
    :param max_step_yaw: Max target rotation per policy step (rad)
    :param paper_pos_noise: Paper reset position noise, uniform +- this in x and y (m)
    :param paper_yaw_noise: Paper reset yaw noise, uniform +- this (rad)
    :param grip_force: Target squeeze force for the grip-force feedback (N)
    """

    def __init__(
        self,
        scene_xml: str | Path = SCENE_XML_PATH,
        control_hz: float = 25.0,
        max_steps: int = 250,
        max_step_pos: float = 0.01,
        max_step_yaw: float = 0.1,
        paper_pos_noise: float = 0.02,
        paper_yaw_noise: float = np.pi / 12,
        grip_force: float = 2.0,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        use_torque_actuators(self.model)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self.max_steps = max_steps
        self.max_step_pos = max_step_pos
        self.max_step_yaw = max_step_yaw
        self.paper_pos_noise = paper_pos_noise
        self.paper_yaw_noise = paper_yaw_noise

        self.osc = OSCController(self.model, self.data, q_null=START_QPOS)
        self.grip = GripController(self.model, self.data, target_force=grip_force)
        self.arm_qadr = self.osc.qadr

        # Paper vertices are bodies with 3 slide joints each; qpos is the offset from body_pos.
        vert_bodies = self.model.flex_vertbodyid
        self.paper_qadr = np.array(
            [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in vert_bodies]
        )
        self.paper_rest = self.model.body_pos[vert_bodies].copy()
        self.table_z = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.paper_z0 = self.data.flexvert_xpos[:, 2].mean()

        self.rng = np.random.default_rng()
        self.steps = 0
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0

        n_obs = len(self._observation())
        if gymnasium is not None:
            spaces = gymnasium.spaces
            self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        else:
            self.observation_space = (np.full(n_obs, -np.inf), np.full(n_obs, np.inf))
            self.action_space = (-np.ones(5), np.ones(5))

    # ----------------------------------------------------------------------------------------
    # Gym API

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = START_QPOS

        # Random paper placement: rotate about its center, then shift.
        center = self.paper_rest.mean(axis=0)
        yaw = self.rng.uniform(-self.paper_yaw_noise, self.paper_yaw_noise)
        shift = np.r_[self.rng.uniform(-self.paper_pos_noise, self.paper_pos_noise, 2), 0.0]
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        placed = (self.paper_rest - center) @ rot.T + center + shift
        for k, adr in enumerate(self.paper_qadr):
            d.qpos[adr : adr + 3] = placed[k] - self.paper_rest[k]

        mujoco.mj_forward(m, d)
        self.osc.reset()
        self.grip.open()
        self.target_pos, _ = self.osc.ee_pose()
        self.target_yaw = 0.0
        self.steps = 0
        self.paper_z0 = d.flexvert_xpos[:, 2].mean()
        return self._observation(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.target_pos = np.clip(
            self.target_pos + action[:3] * self.max_step_pos, WORKSPACE_LOW, WORKSPACE_HIGH
        )
        self.target_yaw = float(
            np.clip(self.target_yaw + action[3] * self.max_step_yaw, -np.pi / 2, np.pi / 2)
        )
        if action[4] > 0:
            self.grip.close()
        else:
            self.grip.open()

        target_quat = self._target_quat()
        m, d = self.model, self.data
        unstable = False
        for _ in range(self.n_substeps):
            self.osc.apply(self.target_pos, target_quat)
            self.grip.update(m.opt.timestep)
            mujoco.mj_step(m, d)
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.isfinite(
                d.qpos
            ).all():
                unstable = True
                break
        self.steps += 1

        info = self._info()
        info["unstable"] = unstable
        reward = self._reward(info, action)
        terminated = info["success"] or unstable
        truncated = self.steps >= self.max_steps
        return self._observation(), reward, terminated, truncated, info

    def sample_action(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, 5)

    # ----------------------------------------------------------------------------------------
    # Feedback signals

    def _target_quat(self) -> np.ndarray:
        """Fingers down, rotated by target_yaw about world z."""
        yaw_q = np.array([np.cos(self.target_yaw / 2), 0.0, 0.0, np.sin(self.target_yaw / 2)])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, yaw_q, DOWNWARD_QUAT)
        return q

    def paper_state(self) -> dict[str, np.ndarray]:
        v = self.data.flexvert_xpos
        nx = ny = int(round(np.sqrt(len(v))))
        corners = v[[0, nx - 1, len(v) - nx, len(v) - 1]] if nx * ny == len(v) else v[:4]
        height = v[:, 2].mean() - self.paper_z0
        return {"center": v.mean(axis=0), "corners": corners, "height": height}

    def _observation(self) -> np.ndarray:
        ee_pos, ee_quat = self.osc.ee_pose()
        ee_vel = self.osc.ee_velocity()
        squeeze = self.grip.squeeze_forces()
        paper = self.paper_state()
        d = self.data
        return np.concatenate(
            [
                ee_pos,
                ee_quat,
                ee_vel,
                self.target_pos - ee_pos,
                [d.qpos[self.grip.qadr], squeeze["left"], squeeze["right"]],
                [d.qpos[self.model.jnt_qposadr[self.model.joint("right_finger_joint").id]]],
                paper["center"] - ee_pos,
                (paper["corners"] - ee_pos).ravel(),
                [paper["height"]],
            ]
        ).astype(np.float32)

    def _info(self) -> dict:
        paper = self.paper_state()
        squeeze = self.grip.squeeze_forces()
        return {
            "paper_height": float(paper["height"]),
            "grip_force": float(min(squeeze.values())),
            "success": bool(paper["height"] > SUCCESS_HEIGHT and min(squeeze.values()) > 0.05),
            "ee_pos": self.osc.ee_pose()[0],
        }

    def _reward(self, info: dict, action: np.ndarray) -> float:
        """Shaped reward: reach the paper, grip it, lift it.

        Kept simple on purpose; tune or replace it for the actual training run.
        """
        ee_pos = info["ee_pos"]
        paper = self.paper_state()
        reach = -np.linalg.norm(paper["center"][:2] - ee_pos[:2]) - abs(ee_pos[2] - self.table_z)
        grip = 1.0 if info["grip_force"] > 0.1 else 0.0
        lift = 10.0 * min(max(info["paper_height"], 0.0), 0.1)
        reward = 0.5 * reach + grip + lift - 0.01 * float(action[:4] @ action[:4])
        if info["success"]:
            reward += 50.0
        if info.get("unstable"):
            reward -= 50.0
        return float(reward)
