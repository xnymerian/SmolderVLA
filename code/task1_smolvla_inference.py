import numpy as np
import torch
import mujoco
import mujoco.viewer
import imageio.v2 as imageio
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action

BASE_DIR = Path(__file__).resolve().parent

# Paths and runtime settings
MODEL_DIR = str(BASE_DIR / "model")
XML_PATH = str(BASE_DIR / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml")
OUTPUT_VIDEO = str(BASE_DIR / "outputs" / "task1_smolvla_panda.mp4")
VIDEO_FPS = 30
SAVE_VIDEO = True  # Set False for faster interactive testing
TOTAL_STEPS = 500
POLICY_EVERY_N_STEPS = 6  # Run policy every N sim steps
SIM_STEPS_PER_LOOP = 2  # Physics steps per loop
USE_AMP = torch.cuda.is_available()  # Enable AMP on CUDA

# Load policy
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = SmolVLAPolicy.from_pretrained(MODEL_DIR).to(device).eval()
print("Policy loaded:", type(policy).__name__, "| device:", device)
torch_device = torch.device(device)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=MODEL_DIR,
    preprocessor_overrides={"device_processor": {"device": device}},
)

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Renderer
renderer = mujoco.Renderer(model, height=256, width=256)
Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
video_writer = imageio.get_writer(OUTPUT_VIDEO, fps=VIDEO_FPS) if SAVE_VIDEO else None

# Language instruction
task_text = "Move the arm smoothly."


def get_observation():
    renderer.update_scene(data)
    rgb = renderer.render()  # HxWx3, uint8

    # Model config expects state dimension 8
    state = np.zeros(8, dtype=np.float32)
    n = min(8, data.qpos.shape[0])
    state[:n] = data.qpos[:n].astype(np.float32)

    return {
        "observation.images.image": rgb,
        "observation.images.image2": rgb,  # duplicate when second camera is absent
        "observation.state": state,
    }


try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        action = np.zeros(model.nu, dtype=np.float32)
        for step in range(TOTAL_STEPS):
            if step % POLICY_EVERY_N_STEPS == 0:
                obs = get_observation()
                pred = predict_action(
                    observation=obs,
                    policy=policy,
                    device=torch_device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=USE_AMP,
                    task=task_text,
                    robot_type="franka",
                )

                if hasattr(pred, "detach"):
                    pred = pred.detach().cpu().numpy()
                pred = np.asarray(pred).reshape(-1)

                # Clip prediction to actuator dimension
                nu = model.nu
                action[:] = 0
                k = min(nu, pred.shape[0])
                action[:k] = pred[:k]

            data.ctrl[:] = action
            for _ in range(SIM_STEPS_PER_LOOP):
                mujoco.mj_step(model, data)

            if SAVE_VIDEO and video_writer is not None:
                renderer.update_scene(data)
                frame = renderer.render()
                video_writer.append_data(frame)
            viewer.sync()
finally:
    if video_writer is not None:
        video_writer.close()
    renderer.close()

if SAVE_VIDEO:
    print(f"Done. Video saved to: {OUTPUT_VIDEO}")
else:
    print("Done. Ran in performance mode (video recording disabled).")
