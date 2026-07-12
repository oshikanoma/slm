#!/usr/bin/env python3
"""Run frontier / larger models ZERO-SHOT on the golden set, for comparison.

The core thesis test: does a fine-tuned 1.7B beat a *prompted* frontier model on the
exact same task + golden set? This prompts each comparison model with the SAME
GEN_SYSTEM prompt the tuned model uses (no fine-tuning, no examples — zero-shot),
and writes eval-format predictions so `eval.py score` grades them identically.

Uses any OpenAI-compatible endpoint (defaults to the one in TEACHER_BASE_URL, e.g.
Groq, which hosts Llama-3.3-70B, GPT-OSS-120B, Qwen3-32B, Llama-4-Scout — all free).

USAGE
  source /tmp/.slm_teacher_env         # OPENAI_API_KEY + TEACHER_BASE_URL (Groq)
  python run_frontier.py --model llama-3.3-70b-versatile --out preds/frontier.llama70b.jsonl
  # or run the default suite:
  python run_frontier.py --suite
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from eval import load_testset, render_user_prompt, GEN_SYSTEM

SUITE = [
    ("llama-3.3-70b-versatile", "llama70b"),
    ("openai/gpt-oss-120b", "gptoss120b"),
    ("qwen/qwen3-32b", "qwen3-32b"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "llama4scout"),
]


def _client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY (+ TEACHER_BASE_URL for Groq).")
    kw = {"api_key": key}
    base = os.environ.get("TEACHER_BASE_URL")
    if base:
        kw["base_url"] = base
    return OpenAI(**kw)


def run_model(client, model: str, scenarios, out_path: str, delay: float = 0.5) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, scn in enumerate(scenarios, 1):
            user = render_user_prompt(scn)
            text = ""
            for attempt in range(4):
                try:
                    r = client.chat.completions.create(
                        model=model, temperature=0,
                        messages=[{"role": "system", "content": GEN_SYSTEM},
                                  {"role": "user", "content": user}])
                    text = r.choices[0].message.content or ""
                    n_ok += 1
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if any(x in msg for x in ("rate", "429", "timeout", "overloaded", "503")):
                        time.sleep(2 * (attempt + 1))
                        continue
                    print(f"  [{model}] {scn.id}: {str(e)[:100]}", file=sys.stderr)
                    break
            f.write(json.dumps({"id": scn.id, "output": text}, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"  {model}: {i}/{len(scenarios)}", file=sys.stderr)
            time.sleep(delay)
    return n_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="Zero-shot frontier-model predictions on the golden set.")
    ap.add_argument("--testset", default="data/golden.json")
    ap.add_argument("--model", help="Single model id (OpenAI-compatible).")
    ap.add_argument("--out", help="Output path for --model.")
    ap.add_argument("--suite", action="store_true", help="Run the default comparison suite.")
    ap.add_argument("--outdir", default="preds", help="Output dir for --suite.")
    args = ap.parse_args(argv)

    scenarios = load_testset(args.testset)
    client = _client()
    print(f"endpoint: {os.environ.get('TEACHER_BASE_URL','<openai>')} | {len(scenarios)} scenarios",
          file=sys.stderr)

    if args.suite:
        for model, tag in SUITE:
            out = os.path.join(args.outdir, f"frontier.{tag}.jsonl")
            print(f"=== {model} -> {out} ===", file=sys.stderr)
            ok = run_model(client, model, scenarios, out)
            print(f"  done: {ok}/{len(scenarios)} predictions", file=sys.stderr)
    else:
        if not (args.model and args.out):
            raise SystemExit("Provide --model and --out, or use --suite.")
        ok = run_model(client, args.model, scenarios, args.out)
        print(f"done: {ok}/{len(scenarios)} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
