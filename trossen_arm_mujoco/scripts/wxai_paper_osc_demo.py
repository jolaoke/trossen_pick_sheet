"""Scripted paper pick with the OSC + grip-force feedback stack, through the RL environment.

Drives ``PaperPickEnv`` with the same 5-value actions an RL policy would output, so it doubles as
a check that the environment can reach success. Strategy (pinch-slide): put the fingertips on
the table with the rigid finger on the sheet and the compliant finger just past its edge, close
(the rigid finger drags the sheet so it buckles up into the compliant finger), then lift.

Usage:
    uv run python trossen_arm_mujoco/scripts/wxai_paper_osc_demo.py            # viewer
    uv run python trossen_arm_mujoco/scripts/wxai_paper_osc_demo.py --headless
"""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from trossen_arm_mujoco.paper_env import PaperPickEnv

# Fingertip target relative to the paper center (m), and how far below the table surface to
# press: 1 mm compliance keeps both fingertips touching the table without crushing the sheet.
GRASP_OFFSET = np.array([0.0, -0.02])
PRESS_DEPTH = 0.001

# (phase, target z, gripper closed?, policy steps)
PHASES = [
    ("approach", 0.08, False, 40),
    ("descend", -PRESS_DEPTH, False, 40),
    ("close", -PRESS_DEPTH, True, 40),
    ("lift", 0.12, True, 60),
    ("hold", 0.12, True, 25),
]


def scripted_action(env: PaperPickEnv, goal: np.ndarray, close: bool) -> np.ndarray:
    """Proportional feedback on the target: step toward ``goal``, saturating at the limit."""
    delta = (goal - env.target_pos) / env.max_step_pos
    return np.r_[np.clip(delta, -1.0, 1.0), 0.0, 1.0 if close else -1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--headless", action="store_true", help="Run without the viewer")
    parser.add_argument("--seed", type=int, default=None, help="Randomize paper placement")
    args = parser.parse_args()

    env = PaperPickEnv(
        paper_pos_noise=0.0 if args.seed is None else 0.02,
        paper_yaw_noise=0.0 if args.seed is None else np.pi / 12,
        max_steps=sum(p[3] for p in PHASES),
    )
    env.reset(seed=args.seed)
    xy = env.paper_state()["center"][:2] + GRASP_OFFSET
    dt = env.n_substeps * env.model.opt.timestep

    viewer = None if args.headless else mujoco.viewer.launch_passive(env.model, env.data)
    total = 0.0
    info = {}
    try:
        for name, z, close, n in PHASES:
            for _ in range(n):
                t0 = time.perf_counter()
                action = scripted_action(env, np.r_[xy, z], close)
                _, reward, terminated, truncated, info = env.step(action)
                total += reward
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                if info["unstable"]:
                    break
            print(
                f"{name:9s} paper height {info['paper_height'] * 100:5.1f} cm  "
                f"grip force {info['grip_force']:.2f} N"
            )
        print(f"success: {info['success']}  return: {total:.1f}")
        while viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
