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
    "seethrough3d_release"
]

# ── Blender Backend Configuration ─────────────────────────────────────────────
# Number of parallel Blender worker instances (each gets 3 servers: cv, final, segmask)
NUM_BLENDER_WORKERS = 2

# Port scheme: Worker i uses ports BASE + i*STRIDE + {0,1,2} for cv/final/segmask
BLENDER_BASE_PORT = 5001
BLENDER_PORT_STRIDE = 10  # Worker 0: 5001/5002/5003, Worker 1: 5011/5012/5013, etc.

def get_blender_ports(worker_idx: int) -> tuple:
    """Return (cv_port, final_port, segmask_port) for the given worker index."""
    base = BLENDER_BASE_PORT + worker_idx * BLENDER_PORT_STRIDE
    return base, base + 1, base + 2

def get_blender_urls(worker_idx: int) -> tuple:
    """Return (cv_url, final_url, segmask_url) for the given worker index."""
    cv_port, final_port, segmask_port = get_blender_ports(worker_idx)
    return (
        f"http://127.0.0.1:{cv_port}",
        f"http://127.0.0.1:{final_port}",
        f"http://127.0.0.1:{segmask_port}",
    )
