#!/usr/bin/env python3
"""Deploy the Cited Newsroom Verifier to Hugging Face: model repo + Gradio Space.

Does three things (idempotent — safe to re-run):
  1. Push the v6 LoRA adapter to a Model repo, with a model card (results + usage).
  2. Create a Gradio Space and upload space/ (app + modules + requirements + README).
  3. Wire the Space to load the adapter (sets HF_ADAPTER_REPO) and reminds you to add
     the TAVILY_API_KEY secret in the Space UI.

PREREQS
  pip install huggingface_hub
  A HF account + a WRITE token from https://huggingface.co/settings/tokens

USAGE
  python push_to_hf.py --user YOUR_HF_USERNAME --token hf_xxx
  # or: export HF_TOKEN=hf_xxx ; python push_to_hf.py --user YOUR_HF_USERNAME
"""
from __future__ import annotations

import argparse
import os
import sys

ADAPTER_DIR = "artifacts/qwen3-verifier-lora-v6"
SPACE_DIR = "space"

MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
tags: [journalism, fact-checking, ap-style, qlora, newsroom]
---

# Cited Newsroom Verifier — Qwen3-1.7B (QLoRA adapter)

A LoRA adapter that turns **Qwen/Qwen3-1.7B** into a cited, evidence-bounded newsroom
verifier: given a passage + retrieved sources, it flags each claim/quote as
`supported` / `unsupported` / `misleading` and only says `supported` when it can quote
the **verbatim** backing sentence from a real source — it never vouches for a claim it
can't cite. (AP Style is handled deterministically in the app, not the model.)

## Results (held-out golden set)
| Model | spec_pass | verdict_acc | fabricated↓ |
|---|---|---|---|
| **This model (1.7B, tuned)** | **73.7–84%** | **74.9%** | **2.9–5.6%** |
| GPT-OSS-120B (zero-shot) | 30.3% | 46.3% | 6.1% |
| Llama-3.3-70B (zero-shot) | 14.3% | 25.7% | 2.4% |
| Base Qwen3-1.7B (zero-shot) | 9.1% | 19.4% | 61.4% |

Beats a prompted 120B by ~2.4×. Base vs tuned significance: McNemar p < 0.0001.
Thesis: **behavior from data, not scale.**

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(base)
m = AutoModelForCausalLM.from_pretrained(base)
m = PeftModel.from_pretrained(m, "{repo}")
```
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Push adapter + Space to Hugging Face.")
    ap.add_argument("--user", required=True, help="Your HF username.")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF WRITE token.")
    ap.add_argument("--model-name", default="qwen3-1.7b-newsroom-verifier")
    ap.add_argument("--space-name", default="newsroom-verifier")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-space", action="store_true")
    args = ap.parse_args(argv)

    if not args.token:
        raise SystemExit("Need a HF write token: --token hf_xxx  (or export HF_TOKEN).")

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=args.token)
    model_repo = f"{args.user}/{args.model_name}"
    space_repo = f"{args.user}/{args.space_name}"

    # 1) MODEL REPO -----------------------------------------------------------
    if not args.skip_model:
        print(f"[model] creating {model_repo} ...")
        create_repo(model_repo, token=args.token, repo_type="model", exist_ok=True)
        # write the model card into the adapter dir
        with open(os.path.join(ADAPTER_DIR, "README.md"), "w", encoding="utf-8") as f:
            f.write(MODEL_CARD.replace("{repo}", model_repo))
        print(f"[model] uploading {ADAPTER_DIR} ...")
        api.upload_folder(folder_path=ADAPTER_DIR, repo_id=model_repo, repo_type="model")
        print(f"[model] done -> https://huggingface.co/{model_repo}")

    # 2) SPACE ----------------------------------------------------------------
    if not args.skip_space:
        print(f"[space] creating {space_repo} (gradio) ...")
        create_repo(space_repo, token=args.token, repo_type="space",
                    space_sdk="gradio", exist_ok=True)
        # point the Space app at the model repo we just pushed
        app_path = os.path.join(SPACE_DIR, "app.py")
        src = open(app_path, encoding="utf-8").read()
        src = src.replace("REPLACE_WITH_YOUR_HF_USERNAME/qwen3-1.7b-newsroom-verifier", model_repo)
        open(app_path, "w", encoding="utf-8").write(src)
        print(f"[space] uploading {SPACE_DIR} ...")
        api.upload_folder(folder_path=SPACE_DIR, repo_id=space_repo, repo_type="space")
        print(f"[space] done -> https://huggingface.co/spaces/{space_repo}")

    print("\n=== NEXT (manual, 30 seconds) ===")
    print(f"1. Open https://huggingface.co/spaces/{space_repo}/settings")
    print("2. Under 'Variables and secrets' add a SECRET: TAVILY_API_KEY = <your tvly- key>")
    print("   (optional — without it the Space uses the free Wikipedia search fallback)")
    print("3. The Space will build & launch automatically. First model load is slow on CPU.")
    print(f"\nYour deployed site: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
