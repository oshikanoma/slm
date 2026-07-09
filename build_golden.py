#!/usr/bin/env python3
"""Compose the eval golden set to the Behavior Spec §5.1 bucket mix.

The golden set MUST contain the trap buckets (true_but_unsupported, distractor,
misleading) or it cannot measure the project's forbidden failure (knowledge
leakage / fabricated citation). This composer builds a ~target-size set with the
spec distribution, while guaranteeing:

  - the original hand-labeled records are kept (trusted human labels),
  - every record passes the objective gate (no fabricated citation),
  - NO passage overlaps data/train.jsonl (train/eval stay disjoint),
  - each record carries golden metadata (must_contain / must_not_contain /
    keywords / expected_verdict / human_label).

Spec §5.1 mix: supported 30%, unsupported 25%, true_but_unsupported 20%,
distractor 15%, misleading 10%.

USAGE
  python build_golden.py --target 120 \
      --keep data/golden.json \
      --real data/tt.real.json data/wikinews.real.json \
      --synthetic data/train.synth.jsonl \
      --exclude-train data/train.jsonl \
      --out data/golden.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

from eval import Scenario, derive_golden_metadata, load_jsonl
from datagen import passes_gate

_WS = re.compile(r"\s+")

SPEC_MIX = {
    "supported": 0.30,
    "unsupported": 0.25,
    "true_but_unsupported": 0.20,
    "distractor": 0.15,
    "misleading": 0.10,
}


def key(p: str) -> str:
    return _WS.sub(" ", (p or "")).strip().lower()


def scn_from(rec: dict) -> Scenario:
    return Scenario(id=rec.get("id", "x"), bucket=rec.get("bucket", ""),
                    passage=rec.get("passage", ""), sources=rec.get("sources", []) or [],
                    gold_verdicts=rec.get("gold_verdicts", []) or [])


def to_golden(rec: dict, gid: str) -> dict:
    scn = scn_from(rec)
    mc, mnc, kw, exp = derive_golden_metadata(scn)
    return {
        "id": gid, "bucket": rec.get("bucket"), "passage": rec.get("passage"),
        "sources": rec.get("sources"), "gold_verdicts": rec.get("gold_verdicts"),
        "must_contain": mc, "must_not_contain": mnc, "keywords": kw,
        "expected_verdict": exp, "human_label": rec.get("human_label"),
    }


def bucket_of(rec: dict) -> str:
    """Map a record to a spec bucket. Real records only ever carry supported/
    unsupported/misleading; synthetic carries the full vocabulary including traps."""
    b = rec.get("bucket")
    if b in SPEC_MIX:
        return b
    v = (rec.get("gold_verdicts") or [{}])[0].get("verdict")
    return v if v in SPEC_MIX else "unsupported"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compose golden set to the spec bucket mix.")
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--keep", default="data/golden.json",
                    help="Existing golden file whose hand-labeled (non real_/synthetic) records to keep.")
    ap.add_argument("--real", nargs="*", default=["data/tt.real.json", "data/wikinews.real.json"])
    ap.add_argument("--synthetic", default="data/train.synth.jsonl")
    ap.add_argument("--exclude-train", default="data/train.jsonl",
                    help="Training file; any passage here is excluded from the golden set.")
    ap.add_argument("--out", default="data/golden.json")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    import random
    rng = random.Random(args.seed)

    train_keys = {key(json.loads(l)["passage"]) for l in open(args.exclude_train)} \
        if os.path.exists(args.exclude_train) else set()
    print(f"excluding {len(train_keys)} training passages", flush=True)

    # 1) keep the trusted hand-labeled originals (ids that are not machine-appended)
    kept: list[dict] = []
    used: set[str] = set()
    if os.path.exists(args.keep):
        for r in load_jsonl(args.keep):
            rid = str(r.get("id", ""))
            if rid.startswith("real_") or "_" in rid and rid.split("_")[0] in ("sup", "uns", "tru", "dis", "mis"):
                continue  # machine-appended; re-select below from pools
            k = key(r.get("passage", ""))
            if k and k not in train_keys and k not in used:
                kept.append(r)
                used.add(k)
    print(f"kept {len(kept)} hand-labeled records", flush=True)

    # 2) build candidate pools by bucket (real first for realism, then synthetic)
    pools: dict[str, list[dict]] = {b: [] for b in SPEC_MIX}
    def add_pool(path: str, is_synth: bool):
        if not os.path.exists(path):
            return
        recs = [json.loads(l) for l in open(path)] if path.endswith(".jsonl") else json.load(open(path))
        for r in recs:
            gv = (r.get("gold_verdicts") or [{}])[0]
            if not gv.get("verdict"):
                continue
            k = key(r.get("passage", ""))
            if not k or k in train_keys or k in used:
                continue
            r = dict(r)
            r["bucket"] = bucket_of(r)
            ok, _ = passes_gate(scn_from(r), r["bucket"])
            if ok:
                pools[r["bucket"]].append(r)
    for p in args.real:
        add_pool(p, False)
    add_pool(args.synthetic, True)
    for b in pools:
        rng.shuffle(pools[b])
    print("pool sizes:", {b: len(v) for b, v in pools.items()}, flush=True)

    # 3) target counts per bucket, minus what the kept records already cover
    kept_by_bucket = Counter(bucket_of(r) for r in kept)
    targets = {b: round(args.target * w) for b, w in SPEC_MIX.items()}
    selected: list[dict] = list(kept)
    for b, want in targets.items():
        need = max(0, want - kept_by_bucket.get(b, 0))
        take = pools[b][:need]
        for r in take:
            k = key(r["passage"])
            if k in used:
                continue
            used.add(k)
            selected.append(r)

    # 4) renumber ids per bucket and write with metadata
    n = Counter()
    out: list[dict] = []
    for r in selected:
        b = bucket_of(r)
        gid = f"g_{b[:3]}_{n[b]:03d}"
        n[b] += 1
        out.append(to_golden(r, gid))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    mix = Counter(bucket_of(r) for r in out)
    print(f"\n=== golden set: {len(out)} records -> {args.out} ===")
    print(f"bucket mix: {dict(mix)}")
    print(f"spec target: {targets}")
    print(f"kept hand-labeled: {len(kept)}  | trap-bucket coverage: "
          f"{sum(mix[b] for b in ('true_but_unsupported','distractor','misleading'))}")


if __name__ == "__main__":
    main()
