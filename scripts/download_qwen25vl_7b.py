import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

repo_id = "Qwen/Qwen2.5-VL-7B-Instruct"
local_dir = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"

os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {repo_id} to {local_dir}")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("Download finished.")
