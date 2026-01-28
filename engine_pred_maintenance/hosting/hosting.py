from huggingface_hub import HfApi
import os

proj_name = "engine-maintenance-prediction-proj"
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="engine_pred_maintenance/deployment",   # the local folder containing your files
    repo_id=f"jackfroooot/{proj_name}",                 # the target repo
    repo_type="space",                        # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
