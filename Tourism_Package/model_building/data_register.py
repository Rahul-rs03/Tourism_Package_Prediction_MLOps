
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from pathlib import Path
from huggingface_hub import login, whoami

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data directory not found at: {DATA_PATH}")

# -----------------------------
# Hugging Face Repo Info
# -----------------------------
repo_id = "rahulsuren/tourism-package-prediction"
repo_type = "dataset"

# -----------------------------
# Auth using ENV token
# -----------------------------
api = HfApi()

# -----------------------------
# Create repo if not exists
# -----------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# -----------------------------
# Upload dataset
# -----------------------------
api.upload_folder(
    folder_path=str(DATA_PATH),
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="raw"
)

print("Data successfully registered on Hugging Face Dataset Hub.")
