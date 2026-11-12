from pathlib import Path
import sys
from time import perf_counter

import mujoco


# Pass a scene path to benchmark it, e.g.
#   uv run python trossen_arm_mujoco/scripts/benchmark_collision.py trossen_arm_mujoco/assets/wxai/scene_wxai_paper.xml
# With no argument it benchmarks the pick-and-place scene, as before.
XML = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trossen_arm_mujoco/assets/wxai/scene_wxai_pick_place.xml")
WARMUP_STEPS = 1_000
MEASURE_STEPS = 10_000


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML)) # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model) # pyright: ignore[reportAttributeAccessIssue]

    for _ in range(WARMUP_STEPS):
        mujoco.mj_step(model, data) # pyright: ignore[reportAttributeAccessIssue]

    start = perf_counter()

    total_contacts = 0
    maximum_contacts = 0

    for _ in range(MEASURE_STEPS):
        mujoco.mj_step(model, data) # pyright: ignore[reportAttributeAccessIssue]
        total_contacts += data.ncon
        maximum_contacts = max(maximum_contacts, data.ncon)

    elapsed = perf_counter() - start
    step_time = elapsed / MEASURE_STEPS
    steps_per_second = MEASURE_STEPS / elapsed
    simulated_seconds = MEASURE_STEPS * model.opt.timestep
    real_time_factor = simulated_seconds / elapsed

    collision_geom_count = sum(
        model.geom_group[i] == 3 for i in range(model.ngeom)
    )

    print(f"XML: {XML}")
    print(f"Total geoms: {model.ngeom}")
    print(f"Collision geoms: {collision_geom_count}")
    print(f"qpos: {model.nq}, qvel: {model.nv}")
    print(f"Average contacts: {total_contacts / MEASURE_STEPS:.2f}")
    print(f"Maximum contacts: {maximum_contacts}")
    print(f"Step time: {step_time * 1e6:.1f} microseconds")
    print(f"Simulation rate: {steps_per_second:.1f} steps/second")
    print(f"Real-time factor: {real_time_factor:.2f}x")


if __name__ == "__main__":
    main()