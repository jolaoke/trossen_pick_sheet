"""Operational space control (OSC) and grip-force feedback for the WidowX AI arm.

The OSC runs at the physics rate and turns a Cartesian end-effector target into joint torques:

    tau = J^T Lambda (Kp e - Kd v) + N^T tau_null + c(q, qdot)

(plus an optional integral term Ki * integral(e) that removes the steady-state error the
joints' friction loss would otherwise leave), where Lambda = (J M^-1 J^T)^-1 is the task-space
inertia, e the pose error, v the end-effector twist, N the dynamically consistent null-space
projector and c the Coriolis/centrifugal forces not covered by the model's gravity compensation.
An RL policy (or a scripted policy) only sets the target pose and the gripper command at a lower
rate; see ``trossen_arm_mujoco/paper_env.py``.

The arm actuators in the MJCF are position servos. ``use_torque_actuators`` rewrites them in the
compiled model into direct torque motors (same force limits), so the XML stays unchanged for the
IK controller and the viewer.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINT_NAMES = [f"joint_{i}" for i in range(6)]
GRIPPER_ACTUATOR_NAME = "left_gripper"
GRIPPER_JOINT_NAME = "left_carriage_joint"

# Gripper carriage position in meters (0 = fully closed, 0.044 = fully open)
GRIPPER_OPEN = 0.044
GRIPPER_CLOSED = 0.0

# Fingers pointing straight down, closing along world y. Quaternion [w, x, y, z].
DOWNWARD_QUAT = np.array([0.70710678, 0.0, 0.70710678, 0.0])


def use_torque_actuators(model: mujoco.MjModel, joint_names: list[str] = ARM_JOINT_NAMES) -> None:
    """Turn the position servos of the given joints into torque motors, in place.

    ctrl then is joint torque in N m, clamped to the actuator's original force range.

    :param model: Compiled model (modified in place)
    :param joint_names: Joints whose actuator (named like the joint) becomes a torque motor
    """
    for name in joint_names:
        act = model.actuator(name).id
        model.actuator_gaintype[act] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[act, :] = 0
        model.actuator_gainprm[act, 0] = 1
        model.actuator_biasprm[act, :] = 0
        model.actuator_ctrlrange[act] = model.actuator_forcerange[act]
        model.actuator_ctrllimited[act] = 1


def quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, world frame) taking ``current`` to ``target``.

    :param target: Target quaternion [w, x, y, z]
    :param current: Current quaternion [w, x, y, z]
    :return: 3D rotation error
    """
    err = np.zeros(3)
    # mju_subQuat gives the error in the frame of `current`; rotate it to world.
    mujoco.mju_subQuat(err, target, current)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, current)
    return rot.reshape(3, 3) @ err


class OSCController:
    """Operational space controller for the end-effector site.

    Call ``compute`` (or ``apply``) every physics step with the current target.

    :param model: MuJoCo model, with the arm actuators converted by ``use_torque_actuators``
    :param data: MuJoCo data
    :param ee_site_name: Controlled site
    :param joint_names: Arm joints driven by the controller (actuators named like the joints)
    :param kp_pos: Position stiffness (1/s^2, task-space acceleration per meter of error)
    :param kp_rot: Orientation stiffness (1/s^2 per radian)
    :param damping_ratio: 1.0 = critically damped
    :param ki_pos: Position integral gain (1/s^3); 0 disables integral action
    :param ki_rot: Orientation integral gain (1/s^3)
    :param integral_limit: Clamp on each integrated error component (anti-windup), [m, rad] * s
    :param kp_null: Stiffness pulling joints toward ``q_null`` inside the null space
    :param q_null: Null-space posture; defaults to the posture at construction
    :param singular_damping: Damping added when inverting J M^-1 J^T near singularities
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_name: str = "ee_site",
        joint_names: list[str] = ARM_JOINT_NAMES,
        kp_pos: float = 400.0,
        kp_rot: float = 400.0,
        damping_ratio: float = 1.0,
        ki_pos: float = 1500.0,
        ki_rot: float = 1500.0,
        integral_limit: float = 0.01,
        kp_null: float = 10.0,
        q_null: np.ndarray | None = None,
        singular_damping: float = 1e-3,
    ):
        self.model = model
        self.data = data
        self.site_id = model.site(ee_site_name).id
        joints = [model.joint(n) for n in joint_names]
        self.dof = np.array([model.jnt_dofadr[j.id] for j in joints])
        self.qadr = np.array([model.jnt_qposadr[j.id] for j in joints])
        self.act = np.array([model.actuator(n).id for n in joint_names])
        self.tau_limit = model.actuator_forcerange[self.act].copy()

        self.kp = np.r_[np.full(3, kp_pos), np.full(3, kp_rot)]
        self.kd = 2.0 * damping_ratio * np.sqrt(self.kp)
        self.ki = np.r_[np.full(3, ki_pos), np.full(3, ki_rot)]
        self.integral_limit = integral_limit
        self.integral = np.zeros(6)
        self.kp_null = kp_null
        self.kd_null = 2.0 * np.sqrt(kp_null)
        self.q_null = data.qpos[self.qadr].copy() if q_null is None else np.asarray(q_null)
        self.singular_damping = singular_damping

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._M = np.zeros((model.nv, model.nv))

    def reset(self) -> None:
        """Clear the integral term (call on episode reset or when the target jumps)."""
        self.integral[:] = 0

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current end-effector position and quaternion [w, x, y, z]."""
        pos = self.data.site_xpos[self.site_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return pos, quat

    def ee_velocity(self) -> np.ndarray:
        """Current end-effector twist [linear, angular] in the world frame."""
        J = self._jacobian()
        return J @ self.data.qvel[self.dof]

    def _jacobian(self) -> np.ndarray:
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        return np.vstack([self._jacp[:, self.dof], self._jacr[:, self.dof]])

    def compute(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        target_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Joint torques that drive the end effector to the target pose.

        :param target_pos: Target position [x, y, z]
        :param target_quat: Target quaternion [w, x, y, z]; None controls position only and
            leaves the orientation to the null-space posture
        :param target_vel: Optional target twist (6,) for trajectory tracking
        :return: Arm joint torques, clipped to the actuator force limits
        """
        m, d = self.model, self.data
        J = self._jacobian()
        mujoco.mj_fullM(m, self._M, d.qM)
        M = self._M[np.ix_(self.dof, self.dof)]
        M_inv = np.linalg.inv(M)
        qvel = d.qvel[self.dof]

        pos, quat = self.ee_pose()
        err = np.zeros(6)
        err[:3] = target_pos - pos
        if target_quat is not None:
            err[3:] = quat_error(np.asarray(target_quat), quat)
        vel_err = -(J @ qvel)
        if target_vel is not None:
            vel_err += target_vel

        self.integral = np.clip(
            self.integral + err * m.opt.timestep, -self.integral_limit, self.integral_limit
        )

        rows = slice(0, 6) if target_quat is not None else slice(0, 3)
        Jt = J[rows]
        lam_inv = Jt @ M_inv @ Jt.T
        Lambda = np.linalg.inv(lam_inv + self.singular_damping * np.eye(lam_inv.shape[0]))
        F = Lambda @ (
            self.kp[rows] * err[rows]
            + self.kd[rows] * vel_err[rows]
            + self.ki[rows] * self.integral[rows]
        )
        tau = Jt.T @ F

        # Null-space posture, projected so it doesn't disturb the task (only acts when the
        # task uses fewer than 6 DOF, e.g. position-only control).
        J_bar = M_inv @ Jt.T @ Lambda
        N = np.eye(len(self.dof)) - Jt.T @ J_bar.T
        tau_null = M @ (
            self.kp_null * (self.q_null - d.qpos[self.qadr]) - self.kd_null * qvel
        )
        tau += N @ tau_null

        # Coriolis/centrifugal plus any gravity the bodies' gravcomp doesn't cancel
        # (e.g. the gravcomp=0 compliant finger links).
        tau += d.qfrc_bias[self.dof] - d.qfrc_gravcomp[self.dof]
        return np.clip(tau, self.tau_limit[:, 0], self.tau_limit[:, 1])

    def apply(self, target_pos, target_quat=None, target_vel=None) -> np.ndarray:
        """``compute`` and write the torques to ``data.ctrl``. Returns the torques."""
        tau = self.compute(target_pos, target_quat, target_vel)
        self.data.ctrl[self.act] = tau
        return tau


class GripController:
    """Grip-force feedback on top of the gripper's position servo.

    ``close`` drives the carriages inward until the object squeezes back on *both* fingers with
    ``target_force``, then keeps adjusting the carriage command to hold that force. Only force
    along each finger's closing direction counts, so pressing down on the object (e.g. a finger
    resting on top of the paper) isn't mistaken for a grasp. ``open`` releases.

    :param model: MuJoCo model
    :param data: MuJoCo data
    :param object_filter: ``f(model, contact) -> bool``, True for contacts with the grasped
        object (default: contacts with any flex, i.e. the paper)
    :param target_force: Desired squeeze force on each finger (N)
    :param close_speed: Carriage closing speed while not yet squeezing (m/s)
    :param force_gain: Carriage adjustment per newton of force error per second (m/(N s))
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        object_filter=None,
        target_force: float = 2.0,
        close_speed: float = 0.05,
        force_gain: float = 0.002,
    ):
        self.model = model
        self.data = data
        self.act = model.actuator(GRIPPER_ACTUATOR_NAME).id
        self.qadr = model.jnt_qposadr[model.joint(GRIPPER_JOINT_NAME).id]
        self.target_force = target_force
        self.close_speed = close_speed
        self.force_gain = force_gain
        self.object_filter = object_filter or (lambda m, c: c.geom[0] < 0 or c.geom[1] < 0)

        # Each finger's geoms and the carriage joint that opens it (+q = outward).
        sides = {
            "left": (["carriage_left"], "left_carriage_joint"),
            "right": (["finger_proximal_right", "finger_distal_right"], "right_carriage_joint"),
        }
        self.finger_side = {}
        self.side_joint = {}
        for side, (bodies, joint) in sides.items():
            body_ids = {model.body(b).id for b in bodies}
            for g in range(model.ngeom):
                if model.geom_bodyid[g] in body_ids:
                    self.finger_side[g] = side
            self.side_joint[side] = model.joint(joint).id
        self.finger_geoms = set(self.finger_side)
        self.command = data.qpos[self.qadr]
        self.closing = False
        self._f = np.zeros(6)

    def squeeze_forces(self) -> dict[str, float]:
        """Force (N) with which the object pushes each finger outward, i.e. grip force per side."""
        m, d = self.model, self.data
        out = {side: 0.0 for side in self.side_joint}
        axis = {}
        for side, j in self.side_joint.items():
            body = m.jnt_bodyid[j]
            axis[side] = d.xmat[body].reshape(3, 3) @ m.jnt_axis[j]
        for i in range(d.ncon):
            c = d.contact[i]
            if not (c.geom[0] in self.finger_geoms or c.geom[1] in self.finger_geoms):
                continue
            if not self.object_filter(m, c):
                continue
            mujoco.mj_contactForce(m, d, i, self._f)
            # Contact force acts on geom[1] along the frame; the finger gets the opposite if it
            # is geom[0].
            f_world = c.frame.reshape(3, 3).T @ self._f[:3]
            g, sign = (c.geom[0], -1.0) if c.geom[0] in self.finger_geoms else (c.geom[1], 1.0)
            side = self.finger_side[g]
            out[side] += max(0.0, sign * float(f_world @ axis[side]))
        return out

    def grip_force(self) -> float:
        """Grip force (N): the smaller of the two fingers' squeeze forces."""
        return min(self.squeeze_forces().values())

    def open(self, width: float = GRIPPER_OPEN) -> None:
        self.closing = False
        self.command = width

    def close(self) -> None:
        if not self.closing:
            self.command = self.data.qpos[self.qadr]
        self.closing = True

    def update(self, dt: float) -> float:
        """Advance the feedback loop by ``dt`` seconds and write the gripper ctrl.

        :return: Current grip force (N)
        """
        force = self.grip_force()
        if self.closing:
            if force < 0.2 * self.target_force:
                self.command -= self.close_speed * dt
            else:
                self.command += self.force_gain * (force - self.target_force) * dt
            self.command = float(np.clip(self.command, GRIPPER_CLOSED, GRIPPER_OPEN))
        self.data.ctrl[self.act] = self.command
        return force
