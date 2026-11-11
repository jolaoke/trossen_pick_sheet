from pathlib import Path
import mujoco
import mujoco.viewer

xml = Path("trossen_arm_mujoco/assets/wxai/scene_wxai_paper.xml")
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()