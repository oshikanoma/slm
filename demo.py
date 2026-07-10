#!/usr/bin/env python3
"""End-to-end demo (spec §8): paste a passage -> retrieve real sources -> the tuned
SLM emits cited verdicts over them. Optionally shows the BASE model on the same
bundle fabricating/leaking, for the base-vs-tuned "money shot".

    passage --> retriever.build_bundle --> source bundle in context
            --> Qwen3-1.7B (+ v3 LoRA adapter) --> cited JSON verdicts

Needs a search key (BRAVE_API_KEY etc.; see retriever.py) and the local model deps
(torch/transformers/peft) already installed for predict_local.py.

USAGE
  export BRAVE_API_KEY=BSA...
  python demo.py "Unemployment fell to 3.2% in March, the governor said."
  python demo.py --compare "..."      # also run the base model for contrast
  python demo.py --bundle-only "..."  # just show what the retriever found
"""
from __future__ import annotations

import argparse
import json
import sys

from eval import Scenario, build_messages, record_spec_pass, load_predictions  # noqa: F401
from retriever import build_bundle, _active_backend

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "artifacts/qwen3-verifier-lora-v3"


def _load_model(with_adapter: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype).to(device)
    if with_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER).to(device)
    model.eval()
    return model, tok, device


def _run(model, tok, device, scn: Scenario, max_new_tokens: int = 512) -> str:
    import torch
    prompt = tok.apply_chat_template(build_messages(scn), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _print_verdicts(raw: str) -> None:
    try:
        obj = json.loads(raw)
    except Exception:
        print("  (model did not return valid JSON)\n  " + raw[:400]); return
    for v in obj.get("verdicts", []):
        mark = {"supported": "✓ SUPPORTED", "unsupported": "✗ UNSUPPORTED",
                "misleading": "! MISLEADING"}.get(v.get("verdict"), v.get("verdict"))
        print(f"  [{mark}] {v.get('span','')[:90]}")
        if v.get("source_url"):
            print(f"      cited:   {v['source_url']}")
            print(f"      backing: \"{(v.get('evidence_span') or '')[:110]}\"")
        if v.get("checked_source_url"):
            print(f"      checked: {v['checked_source_url']}")
            print(f"      nearest: \"{(v.get('nearest_span') or '')[:110]}\"")
    if not obj.get("verdicts"):
        print("  (clean — nothing to flag)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cited-verifier demo: retrieve + verify.")
    ap.add_argument("passage")
    ap.add_argument("-k", type=int, default=4, help="Sources to retrieve.")
    ap.add_argument("--compare", action="store_true", help="Also run the base model.")
    ap.add_argument("--bundle-only", action="store_true", help="Only show retrieval.")
    args = ap.parse_args(argv)

    name, _ = _active_backend()
    print(f"\n=== RETRIEVE (search backend: {name}) ===")
    bundle = build_bundle(args.passage, k=args.k, verbose=True)
    if not bundle:
        print("No sources retrieved. Set a search key (e.g. BRAVE_API_KEY) — see retriever.py.",
              file=sys.stderr)
        sys.exit(1)
    for i, s in enumerate(bundle, 1):
        print(f"  [{i}] {s['url']}\n      {s['text'][:120]}...")
    if args.bundle_only:
        return

    scn = Scenario(id="demo", bucket="", passage=args.passage, sources=bundle,
                   gold_verdicts=[])

    print("\n=== TUNED model (v3 adapter) verdicts ===")
    tmodel, tok, device = _load_model(with_adapter=True)
    _print_verdicts(_run(tmodel, tok, device, scn))

    if args.compare:
        print("\n=== BASE model (no adapter) verdicts — for contrast ===")
        bmodel, tok2, device2 = _load_model(with_adapter=False)
        _print_verdicts(_run(bmodel, tok2, device2, scn))

    print()


if __name__ == "__main__":
    main()
