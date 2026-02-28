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
LORA_WEIGHTS_ROOT = os.path.join(os.path.dirname(BASE_DIR), "checkpoints") 
if not os.path.exists(LORA_WEIGHTS_ROOT):
    raise FileNotFoundError(f"LoRA weights root directory not found: {LORA_WEIGHTS_ROOT}") 

# Available Checkpoint weights (dropdown options)
CHECKPOINT_NAMES = [
    "seethrough3d_release/seethrough3d_release"
]

# ── Blender Backend Server URLs ───────────────────────────────────────────────
BLENDER_CV_SERVER_URL = "http://127.0.0.1:5001"
BLENDER_FINAL_SERVER_URL = "http://127.0.0.1:5002"
BLENDER_SEGMASK_SERVER_URL = "http://127.0.0.1:5003"
