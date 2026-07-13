#!/usr/bin/env python3
"""Deploy the FREE static HF Space (no PRO needed — static Spaces are free).

Usage:
  python deploy_static_space.py --user YOUR_HF_USERNAME --token hf_xxx
  # or: export HF_TOKEN=hf_xxx ; python deploy_static_space.py --user YOUR_HF_USERNAME
"""
import argparse, os
from huggingface_hub import HfApi, create_repo

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
ap.add_argument("--space-name", default="the-verifier")
a = ap.parse_args()
if not a.token:
    raise SystemExit("Need --token hf_xxx (or export HF_TOKEN).")

repo = f"{a.user}/{a.space_name}"
api = HfApi(token=a.token)
print(f"creating static Space {repo} ...")
create_repo(repo, token=a.token, repo_type="space", space_sdk="static", exist_ok=True)
print("uploading static_space/ ...")
api.upload_folder(folder_path="static_space", repo_id=repo, repo_type="space")
print(f"\n✅ DEPLOYED: https://huggingface.co/spaces/{repo}")
print("   (static Space builds in seconds; refresh if it shows 'Building')")
