#!/usr/bin/env python3
"""Merge the v6 LoRA adapter into Qwen3-1.7B -> a standalone fp16 model.

This is the foundation for browser (WebGPU) conversion: converters expect a
single self-contained model, not base + adapter. Output goes to
artifacts/qwen3-verifier-merged/.
"""
from __future__ import annotations
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "artifacts/qwen3-verifier-lora-v6"
OUT = "artifacts/qwen3-verifier-merged"


def main():
    print(f"[1/4] loading base {BASE} (fp16, cpu) ...")
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
    print(f"[2/4] attaching adapter {ADAPTER} ...")
    m = PeftModel.from_pretrained(m, ADAPTER)
    print("[3/4] merge_and_unload ...")
    m = m.merge_and_unload()
    print(f"[4/4] saving merged model -> {OUT} ...")
    os.makedirs(OUT, exist_ok=True)
    m.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    print("done.")
    print("files:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
