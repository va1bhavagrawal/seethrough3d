from huggingface_hub import hf_hub_download
import os

# Download the checkpoint from Hugging Face
checkpoint_path = hf_hub_download(
    repo_id="va1bhavagrawa1/seethrough3d-flux.1-weights",
    filename="checkpoints/seethrough3d_release/lora.safetensors",
    repo_type="model",
    local_dir=".",
    local_dir_use_symlinks=False
)

print(f"Checkpoint downloaded to: {checkpoint_path}")
