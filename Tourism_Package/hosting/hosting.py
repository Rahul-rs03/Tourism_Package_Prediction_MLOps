from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/content/drive/MyDrive/Projects/Tourism_Package/deployment",     # the local folder containing your files
    repo_id="rahulsuren/tourism-package-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

print("Deployment files uploaded to Hugging Face Space successfully.")
