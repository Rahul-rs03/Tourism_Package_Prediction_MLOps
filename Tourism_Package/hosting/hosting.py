from pathlib import Path
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]   # Tourism_Package
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"

if not DEPLOYMENT_DIR.exists():
    raise FileNotFoundError(f"Deployment directory not found at: {DEPLOYMENT_DIR}")

# -------------------------------------------------
# Hugging Face Space upload
# -------------------------------------------------
api = HfApi()

SPACE_REPO = "rahulsuren/tourism-package-model"  
REPO_TYPE = "space"

api.upload_folder(
    folder_path=str(DEPLOYMENT_DIR),
    repo_id=SPACE_REPO,
    repo_type=REPO_TYPE,
    path_in_repo=""
)

print("Deployment files uploaded to Hugging Face Space successfully..")
