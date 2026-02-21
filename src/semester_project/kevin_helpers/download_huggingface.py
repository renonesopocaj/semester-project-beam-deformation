from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="kstnr/elastic-beam-3d", repo_type="dataset", local_dir="../../../dataset/raw")
print(local_dir)