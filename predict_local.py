#!/usr/bin/env python3
"""Generate base + tuned predictions locally (Apple Silicon / MPS), using the
downloaded LoRA adapter. Mirrors the Colab generate step but runs full-precision
on Mac (4-bit bitsandbytes isn't available here).

  python predict_local.py --testset data/golden.json \
      --adapter artifacts/qwen3-verifier-lora \
      --base-out base_preds.jsonl --tuned-out tuned_preds.jsonl
"""
from __future__ import annotations
import argparse, json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from eval import load_testset, build_messages

BASE = "Qwen/Qwen3-1.7B"  # full-precision base (adapter was trained on the 4-bit variant; compatible)


def gen(model, tok, scenarios, out_path, device, max_new_tokens=512):
    model.eval()
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, scn in enumerate(scenarios, 1):
            msgs = build_messages(scn)
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"id": scn.id, "output": text}, ensure_ascii=False) + "\n")
            if i % 5 == 0 or i == len(scenarios):
                el = time.time() - t0
                print(f"  {i}/{len(scenarios)}  ({el:.0f}s, {el/i:.1f}s/rec)", file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="data/golden.json")
    ap.add_argument("--adapter", default="artifacts/qwen3-verifier-lora")
    ap.add_argument("--base-out", default="base_preds.jsonl")
    ap.add_argument("--tuned-out", default="tuned_preds.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    scenarios = load_testset(args.testset)
    if args.limit:
        scenarios = scenarios[:args.limit]
    print(f"device={device} dtype={dtype} scenarios={len(scenarios)}", file=sys.stderr)

    print("loading base model ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype).to(device)

    print("=== BASE predictions (no adapter) ===", file=sys.stderr)
    gen(model, tok, scenarios, args.base_out, device)

    print("=== attaching LoRA adapter ===", file=sys.stderr)
    tuned = PeftModel.from_pretrained(model, args.adapter).to(device)
    print("=== TUNED predictions (with adapter) ===", file=sys.stderr)
    gen(tuned, tok, scenarios, args.tuned_out, device)
    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
