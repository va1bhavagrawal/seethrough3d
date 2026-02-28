from huggingface_hub import hf_hub_download
import os

LOCAL_DIR = "."

repo_id = "va1bhavagrawa1/seethrough3d-data"
file_in_repo = "seethrough3d_data.tar"  # change this

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=file_in_repo,
    repo_type="dataset",
    local_dir=LOCAL_DIR,                  # current directory
    local_dir_use_symlinks=False    # makes a real copy (not symlink)
)

print("Downloaded to:", os.path.abspath(local_path))