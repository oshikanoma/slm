#!/usr/bin/env python3
"""Multi-model leaderboard on the golden set — the thesis test.

Scores any set of prediction files (base Qwen, tuned SLM, and zero-shot frontier
models) on the SAME golden set with the SAME metrics, and prints one leaderboard
sorted by spec_pass. Answers: does the fine-tuned 1.7B beat prompted frontier models?

USAGE
  python compare_models.py --testset data/golden.json \
      --model "Tuned Qwen3-1.7B (ours)=tuned_preds.v5.jsonl" \
      --model "Base Qwen3-1.7B=/tmp/base.v5.jsonl" \
      --model "Llama-3.3-70B (zero-shot)=preds/frontier.llama70b.jsonl" \
      ...
  # or auto-discover:
  python compare_models.py --auto
"""
from __future__ import annotations

import argparse
import glob
import os

from eval import load_testset, load_predictions, compute_metrics

KEY_METRICS = [
    ("spec_pass_rate", "spec_pass"),
    ("verdict_accuracy", "verdict_acc"),
    ("fabricated_citation_rate", "fabricated↓"),
    ("knowledge_leakage_rate", "leakage↓"),
    ("flag_recall", "flag_recall"),
]


def _pct(res, key):
    c = res["metrics"].get(key)
    return c.rate if (c and c.rate is not None) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Leaderboard across models on the golden set.")
    ap.add_argument("--testset", default="data/golden.json")
    ap.add_argument("--model", action="append", default=[],
                    help='"Display name=path/to/preds.jsonl" (repeatable).')
    ap.add_argument("--auto", action="store_true",
                    help="Auto-discover common local prediction files.")
    ap.add_argument("--out", default="results.leaderboard.md")
    args = ap.parse_args(argv)

    entries = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        entries.append((name.strip(), path.strip()))
    if args.auto:
        # our tuned model: prefer the newest local tuned preds that exist
        for path in ("tuned_preds.v6.jsonl", "tuned_preds.v5.jsonl", "tuned_preds.v4.jsonl"):
            if os.path.exists(path):
                entries.append(("Tuned Qwen3-1.7B (ours)", path)); break
        # base model, zero-shot
        for path in ("/tmp/base.v5.jsonl", "/tmp/base.v4.jsonl"):
            if os.path.exists(path):
                entries.append(("Base Qwen3-1.7B (zero-shot)", path)); break
        # frontier files
        labels = {"llama70b": "Llama-3.3-70B (zero-shot)",
                  "gptoss120b": "GPT-OSS-120B (zero-shot)",
                  "qwen3-32b": "Qwen3-32B (zero-shot)",
                  "llama4scout": "Llama-4-Scout-17B (zero-shot)"}
        for path in sorted(glob.glob("preds/frontier.*.jsonl")):
            tag = os.path.basename(path).split(".")[1]
            entries.append((labels.get(tag, tag), path))

    scenarios = load_testset(args.testset)
    rows = []
    for name, path in entries:
        if not os.path.exists(path):
            print(f"  skip (missing): {name} <- {path}"); continue
        res = compute_metrics(scenarios, load_predictions(path))
        rows.append((name, {k: _pct(res, k) for k, _ in KEY_METRICS}))

    # sort by spec_pass desc
    rows.sort(key=lambda r: (r[1].get("spec_pass_rate") or -1), reverse=True)

    hdr = "| Model | " + " | ".join(lbl for _, lbl in KEY_METRICS) + " |"
    sep = "|" + "---|" * (len(KEY_METRICS) + 1)
    lines = [f"# Model leaderboard — golden set ({len(scenarios)} records)", "", hdr, sep]
    for name, m in rows:
        cells = []
        for key, _ in KEY_METRICS:
            v = m.get(key)
            cells.append(f"{v*100:.1f}%" if v is not None else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
