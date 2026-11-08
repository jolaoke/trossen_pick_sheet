from pathlib import Path
import time

import mujoco
import mujoco.viewer


XML = Path("trossen_arm_mujoco/assets/wxai/scene_wxai_pick_place.xml")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)

    print(f"Loaded: {XML}")
    print(f"nq={model.nq}, nv={model.nv}, ngeom={model.ngeom}")

    print("\nCollision geometries:")
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] != 3:
            continue

        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ) or f"unnamed_geom_{geom_id}"

        print(
            f"  id={geom_id:2d} "
            f"name={name:40s} "
            f"type={model.geom_type[geom_id]} "
            f"pos={model.geom_pos[geom_id]} "
            f"size={model.geom_size[geom_id]} "
            f"quat={model.geom_quat[geom_id]}"
        )

    mujoco.mj_forward(model, data)

    # Show both visual meshes (group 2) and collision geometry (group 3).
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[2] = 1
        viewer.opt.geomgroup[3] = 1

        # Make collision geometry visibly red and semi-transparent.
        for geom_id in range(model.ngeom):
            if model.geom_group[geom_id] == 3:
                model.geom_matid[geom_id] = -1
                model.geom_rgba[geom_id] = [1.0, 0.0, 0.0, 0.35]

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()