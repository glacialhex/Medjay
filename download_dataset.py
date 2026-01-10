from huggingface_hub import snapshot_download
import os

repo_id = "Arasoul/hieroglyph-yolo-dataset"
local_dir = "datasets/hieroglyph-yolo-dataset"

print(f"Downloading {repo_id} to {local_dir}...")
snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="dataset", ignore_patterns=[".git*"])
print("Download complete.")
