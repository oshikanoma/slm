#!/usr/bin/env python3
"""v3 rebalance: fix the over-flagging bug found in calibration.

v2 train was 73% flag-buckets / 21% supported, so the model learned to over-flag
(marks genuinely-supported claims unsupported; misses in-bundle support). Fix per the
Brainlift rule "fix the data, not the hyperparameters": raise the `supported` share.

Strategy (all free / local):
  - Keep the existing v2 train.jsonl as the base pool.
  - Pull in UNUSED real `supported` records (migrated to the v2 contract on the way in).
  - Downsample the over-represented `unsupported` bucket.
  - Target mix ~ supported 33 / unsupported 25 / true_but_unsupported 18 /
    distractor 14 / misleading 10, ~1000 total, traps preserved, golden kept disjoint.
  - Rebuild train.sft.jsonl. Every record re-passes the v2 gate.

USAGE
  python rebalance_v3.py --target 1000
"""
from __future__ import annotations
import argparse, json, os, re, random
from collections import Counter

from eval import load_jsonl
from datagen import passes_gate, to_sft_record, to_scenario_record
from migrate_v2 import scn_from, migrate_record

_WS = re.compile(r"\s+")
key = lambda p: _WS.sub(" ", (p or "")).strip().lower()

TARGET_MIX = {
    "supported": 0.33,
    "unsupported": 0.25,
    "true_but_unsupported": 0.18,
    "distractor": 0.14,
    "misleading": 0.10,
}


def bucket_of(rec: dict) -> str:
    b = rec.get("bucket")
    if b in TARGET_MIX:
        return b
    v = (rec.get("gold_verdicts") or [{}])[0].get("verdict")
    return v if v in TARGET_MIX else "unsupported"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--golden", default="data/golden.json")
    ap.add_argument("--real", nargs="*", default=["data/tt.real.json", "data/wikinews.real.json"])
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--sft-out", default="data/train.sft.jsonl")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    golden_keys = {key(r.get("passage", "")) for r in load_jsonl(args.golden)}

    # 1) bucket the existing v2 train pool
    pools: dict[str, list[dict]] = {b: [] for b in TARGET_MIX}
    seen: set[str] = set(golden_keys)
    for rec in load_jsonl(args.train):
        kk = key(rec.get("passage", ""))
        if kk in seen:
            continue
        seen.add(kk)
        pools.setdefault(bucket_of(rec), []).append(rec)
    before = {b: len(v) for b, v in pools.items()}

    # 2) add UNUSED real `supported` (migrate to v2 contract on the way in)
    added = 0
    for path in args.real:
        if not os.path.exists(path):
            continue
        for rec in load_jsonl(path):
            gv = (rec.get("gold_verdicts") or [{}])[0]
            if gv.get("verdict") != "supported":
                continue
            kk = key(rec.get("passage", ""))
            if not kk or kk in seen:
                continue
            rec = dict(rec); rec["bucket"] = "supported"
            m, why = migrate_record(rec)     # adds checked_source_url + gates
            if m is None:
                continue
            seen.add(kk); pools["supported"].append(m); added += 1

    for b in pools:
        rng.shuffle(pools[b])

    # 3) compose to target mix
    targets = {b: round(args.target * w) for b, w in TARGET_MIX.items()}
    selected: list[dict] = []
    shortfall = 0
    for b, want in targets.items():
        take = pools[b][:want]
        selected += take
        if len(take) < want:
            shortfall += want - len(take)
    # backfill any shortfall from leftover supported/unsupported to hit target
    if len(selected) < args.target:
        leftovers = [r for b in ("supported", "unsupported", "true_but_unsupported")
                     for r in pools[b][targets[b]:]]
        rng.shuffle(leftovers)
        selected += leftovers[:args.target - len(selected)]
    selected = selected[:args.target]

    # 4) re-gate + reindex
    final, n = [], Counter()
    for rec in selected:
        scn = scn_from(rec)
        ok, _ = passes_gate(scn, bucket_of(rec))
        if not ok:
            continue
        b = bucket_of(rec)
        scn.id = f"{b[:3]}_{n[b]:04d}"; n[b] += 1
        rec = dict(rec); rec["id"] = scn.id; rec["bucket"] = b
        final.append(rec)

    # 5) write train.jsonl (scenario form) + SFT
    with open(args.train, "w", encoding="utf-8") as f:
        for rec in final:
            scn = scn_from(rec)
            out = to_scenario_record(scn); out["bucket"] = rec["bucket"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    with open(args.sft_out, "w", encoding="utf-8") as f:
        for rec in final:
            f.write(json.dumps(to_sft_record(scn_from(rec)), ensure_ascii=False) + "\n")

    mix = Counter(bucket_of(r) for r in final)
    print("=== v3 rebalance ===")
    print("before (v2 pool):", before)
    print(f"added unused real supported: {added}")
    print(f"after: {dict(mix)}  total {len(final)}")
    print(f"supported share: {100*mix['supported']/len(final):.0f}% (was ~21%)")
    print(f"wrote {args.train} , {args.sft_out}")


if __name__ == "__main__":
    main()
