
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------
# Paths
# -----------------------------
#BASE_PATH = "/content/drive/MyDrive/Projects/Tourism_Package"
#DATA_PATH = os.path.join(BASE_PATH, "data")

BASE_PATH = "Tourism_Package"
DATA_PATH = os.path.join(BASE_PATH, "data")

# -----------------------------
# Hugging Face Repo Info
# -----------------------------
repo_id = "rahulsuren12/tourism-package-prediction"
repo_type = "dataset"

# -----------------------------
# Auth using ENV token
# -----------------------------
import os
os.environ["TOURISM_TOKEN"] = "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"

TOURISM_TOKEN = os.getenv("TOURISM_TOKEN")
if TOURISM_TOKEN is None:
    raise ValueError("TOURISM_TOKEN not found in environment variables")

api = HfApi(token=TOURISM_TOKEN)

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
    folder_path=DATA_PATH,
    repo_id=repo_id,
    repo_type=repo_type
)

print("Data successfully registered on Hugging Face Dataset Hub.")
