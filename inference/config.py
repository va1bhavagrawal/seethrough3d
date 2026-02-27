import os
import os.path as osp

# ── Base Directories ──────────────────────────────────────────────────────────
# Base directory for the gradio_app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory where generated images, segmasks, and jsonl logs are stored
GRADIO_FILES_DIR = os.path.join(BASE_DIR, "gradio_files")

# Directory where scene pickle files are saved
SAVED_SCENES_DIR = os.path.join(BASE_DIR, "saved_scenes")

# ── Model & Weights Paths ─────────────────────────────────────────────────────
# The base FLUX model repository or local path
PRETRAINED_MODEL_NAME_OR_PATH = "black-forest-labs/FLUX.1-dev"

# The root directory where LoRA fine-tunes are stored
LORA_WEIGHTS_ROOT = "/archive/vaibhav.agrawal/a-bev-of-the-latents/easycontrol_cuboids"

# The directory containing cached inference embeddings (if used)
INFERENCE_EMBEDS_DIR = "/archive/vaibhav.agrawal/a-bev-of-the-latents/inference_embeds_flux2"

# Available Checkpoint weights (dropdown options)
CHECKPOINT_NAMES = [
    "seethrough3d_release/seethrough3d_release"
]

# ── Blender Backend Server URLs ───────────────────────────────────────────────
BLENDER_CV_SERVER_URL = "http://127.0.0.1:5001"
BLENDER_FINAL_SERVER_URL = "http://127.0.0.1:5002"
BLENDER_SEGMASK_SERVER_URL = "http://127.0.0.1:5003"
