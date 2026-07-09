#!/usr/bin/env python3
"""Assemble the final training set for the Cited Newsroom Verifier SLM.

Merges two legally-clean streams into the exact schema eval.py / train.ipynb use:

  1. REAL scaffolds  — labeled Texas Tribune (CC republishing) + Wikinews (CC BY 2.5)
     records from ingest_texastribune.py / ingest_wikinews.py + autolabel.py. These
     give realistic `supported` / `misleading` / `unsupported` cases.
  2. SYNTHETIC        — teacher-distilled scenarios from datagen.py. These supply the
     bulk and, crucially, the trap buckets (true_but_unsupported / distractor) that
     real journalism almost never provides but that teach the model's core restraint.

Guarantees (all enforced here, not assumed):
  - EVERY record re-passes datagen.passes_gate, so no fabricated citation / non-verbatim
    span can enter the training set — same gate the eval scores against.
  - NO passage overlap with the eval golden set (data/golden.json). Real records whose
    passage already appears in golden.json are dropped. This keeps train/eval disjoint,
    which the behavior spec (§5.1) requires.
  - Optionally RESERVES a slice of surplus real records to grow the eval golden set
    toward its ~120 target (--grow-golden), appended with derived metadata.
  - Attribution for every real record is written to data/attribution.json.

OUTPUTS
  data/train.jsonl       scenario records (id, bucket, passage, sources, gold_verdicts,
                         + must_contain/must_not_contain/keywords/expected_verdict)
  data/train.sft.jsonl   chat-format records for QLoRA SFT (system+user+assistant JSON)
  data/attribution.json  source/license manifest for the real records used
  (optionally appends reserved real records to data/golden.json)

USAGE
  # after ingest + autolabel produced data/*.real.json, and datagen produced synthetic:
  python build_dataset.py \
      --real data/golden.real.json data/wikinews.real.json \
      --synthetic data/train.synth.jsonl \
      --golden data/golden.json \
      --target 1000 --grow-golden 100 \
      --out data/train.jsonl --sft-out data/train.sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

from eval import (
    Scenario,
    derive_golden_metadata,
    load_jsonl,
    norm,
)
from datagen import passes_gate, to_scenario_record, to_sft_record

_WS = re.compile(r"\s+")


def passage_key(passage: str) -> str:
    """Whitespace/case-normalized passage, for exact dedup across sets."""
    return _WS.sub(" ", (passage or "")).strip().lower()


def scn_from_record(rec: dict) -> Scenario:
    """Build a Scenario from a golden/real/synthetic record dict (ignores `_` fields)."""
    return Scenario(
        id=rec.get("id", "x"),
        bucket=rec.get("bucket", ""),
        passage=rec.get("passage", ""),
        sources=rec.get("sources", []) or [],
        gold_verdicts=rec.get("gold_verdicts", []) or [],
    )


# Real autolabel buckets -> the spec's bucket vocabulary. A real "unsupported" record
# (the link doesn't back the claim) is a genuine distractor-style negative; we keep it
# tagged `unsupported` so the mix accounting stays honest.
REAL_BUCKET_MAP = {
    "supported": "supported",
    "unsupported": "unsupported",
    "misleading": "misleading",
}


def load_real(paths: list[str], golden_keys: set[str]) -> tuple[list[Scenario], list[dict], Counter]:
    """Load + gate real labeled records. Returns (scenarios, attribution, drop_reasons).

    Drops: records already in the eval golden set (by passage), records with an empty
    verdict (never labeled), and anything that fails the objective quality gate.
    """
    scns: list[Scenario] = []
    attribution: list[dict] = []
    drops: Counter = Counter()
    seen: set[str] = set(golden_keys)  # also dedup real-vs-real
    for path in paths:
        if not os.path.exists(path):
            print(f"  [real] missing, skipping: {path}", file=sys.stderr)
            continue
        for rec in load_jsonl(path):
            gv = (rec.get("gold_verdicts") or [{}])[0]
            if not gv.get("verdict"):
                drops["unlabeled"] += 1
                continue
            key = passage_key(rec.get("passage", ""))
            if not key:
                drops["empty_passage"] += 1
                continue
            if key in seen:
                drops["dup_or_in_golden"] += 1
                continue
            rec = dict(rec)
            rec["bucket"] = REAL_BUCKET_MAP.get(gv.get("verdict"), gv.get("verdict"))
            scn = scn_from_record(rec)
            ok, reason = passes_gate(scn, scn.bucket)
            if not ok:
                drops[f"gate:{reason}"] += 1
                continue
            seen.add(key)
            scns.append(scn)
            attribution.append({
                "id": rec.get("id"),
                "article": rec.get("_source_article"),
                "title": rec.get("_article_title"),
                "byline": rec.get("_byline"),
                "license": rec.get("_license", "Texas Tribune republishing (CC)"),
                "bucket": scn.bucket,
            })
    return scns, attribution, drops


def load_synthetic(path: str | None, golden_keys: set[str], used_keys: set[str]) -> tuple[list[Scenario], Counter]:
    scns: list[Scenario] = []
    drops: Counter = Counter()
    if not path:
        return scns, drops
    if not os.path.exists(path):
        print(f"  [synthetic] missing, skipping: {path}", file=sys.stderr)
        return scns, drops
    seen = set(golden_keys) | set(used_keys)
    for rec in load_jsonl(path):
        key = passage_key(rec.get("passage", ""))
        if key and key in seen:
            drops["dup_or_in_golden"] += 1
            continue
        scn = scn_from_record(rec)
        ok, reason = passes_gate(scn, scn.bucket or "supported")
        if not ok:
            drops[f"gate:{reason}"] += 1
            continue
        if key:
            seen.add(key)
        scns.append(scn)
    return scns, drops


def reindex(scns: list[Scenario]) -> None:
    """Stable, readable ids by bucket, in place."""
    n: Counter = Counter()
    for scn in scns:
        b = (scn.bucket or "x")[:4]
        scn.id = f"{b}_{n[b]:04d}"
        n[b] += 1


def append_to_golden(golden_path: str, reserved: list[Scenario]) -> None:
    """Append reserved real records to the eval golden set with derived metadata."""
    existing = load_jsonl(golden_path) if os.path.exists(golden_path) else []
    existing_ids = {r.get("id") for r in existing}
    added = 0
    for scn in reserved:
        mc, mnc, kw, exp = derive_golden_metadata(scn)
        gid = f"real_{added:04d}"
        while gid in existing_ids:
            added += 1
            gid = f"real_{added:04d}"
        existing.append({
            "id": gid, "bucket": scn.bucket, "passage": scn.passage,
            "sources": scn.sources, "gold_verdicts": scn.gold_verdicts,
            "must_contain": mc, "must_not_contain": mnc,
            "keywords": kw, "expected_verdict": exp, "human_label": None,
        })
        existing_ids.add(gid)
        added += 1
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return added


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Assemble the final training set (real + synthetic).")
    ap.add_argument("--real", nargs="*", default=["data/golden.real.json", "data/wikinews.real.json"],
                    help="Labeled real scaffold files (TT + Wikinews).")
    ap.add_argument("--synthetic", default="data/train.synth.jsonl",
                    help="datagen.py scenario output to merge in for bulk + trap buckets.")
    ap.add_argument("--golden", default="data/golden.json",
                    help="Eval golden set — used for dedup (never mixed into training).")
    ap.add_argument("--target", type=int, default=1000, help="Desired training-set size.")
    ap.add_argument("--grow-golden", type=int, default=0,
                    help="Reserve up to N surplus real records to append to the golden set.")
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--sft-out", default="data/train.sft.jsonl")
    ap.add_argument("--attribution-out", default="data/attribution.json")
    args = ap.parse_args(argv)

    # 1) eval golden keys (for disjointness)
    golden_recs = load_jsonl(args.golden) if os.path.exists(args.golden) else []
    golden_keys = {passage_key(r.get("passage", "")) for r in golden_recs}
    print(f"eval golden set: {len(golden_recs)} records ({len(golden_keys)} unique passages)",
          file=sys.stderr)

    # 2) real records (gated, deduped vs golden + each other)
    real_scns, attribution, real_drops = load_real(args.real, golden_keys)
    used_keys = {passage_key(s.passage) for s in real_scns}
    print(f"real accepted: {len(real_scns)}  drops: {dict(real_drops)}", file=sys.stderr)

    # 3) reserve surplus real records for the eval golden set (optional)
    reserved: list[Scenario] = []
    if args.grow_golden > 0:
        # prefer reserving supported/misleading (most valuable, hardest to synthesize real)
        real_scns.sort(key=lambda s: 0 if s.bucket in ("supported", "misleading") else 1)
        reserved = real_scns[:args.grow_golden]
        real_scns = real_scns[args.grow_golden:]
        used_keys = {passage_key(s.passage) for s in real_scns}
        print(f"reserved {len(reserved)} real records to grow golden set", file=sys.stderr)

    # 4) synthetic records (gated, deduped vs golden + reserved + real)
    reserve_keys = {passage_key(s.passage) for s in reserved}
    synth_scns, synth_drops = load_synthetic(args.synthetic, golden_keys | reserve_keys, used_keys)
    print(f"synthetic accepted: {len(synth_scns)}  drops: {dict(synth_drops)}", file=sys.stderr)

    # 5) combine: real first (realism), then synthetic to fill toward target
    combined = list(real_scns)
    need = max(0, args.target - len(combined))
    combined += synth_scns[:need] if need else []
    if len(combined) > args.target:
        combined = combined[:args.target]
    reindex(combined)

    # 6) write outputs
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for scn in combined:
            f.write(json.dumps(to_scenario_record(scn), ensure_ascii=False) + "\n")
    with open(args.sft_out, "w", encoding="utf-8") as f:
        for scn in combined:
            f.write(json.dumps(to_sft_record(scn), ensure_ascii=False) + "\n")
    with open(args.attribution_out, "w", encoding="utf-8") as f:
        json.dump(attribution, f, ensure_ascii=False, indent=2)

    grown = 0
    if reserved:
        grown = append_to_golden(args.golden, reserved)

    # 7) report
    mix = Counter(s.bucket for s in combined)
    src_mix = Counter("real" if s in real_scns else "synthetic" for s in combined)
    print("\n=== build_dataset summary ===", file=sys.stderr)
    print(f"train set: {len(combined)}  (target {args.target})", file=sys.stderr)
    print(f"  bucket mix: {dict(mix)}", file=sys.stderr)
    n_real = sum(1 for s in combined if s in real_scns)
    print(f"  source mix: real={n_real}  synthetic={len(combined) - n_real}", file=sys.stderr)
    print(f"wrote: {args.out} , {args.sft_out} , {args.attribution_out}", file=sys.stderr)
    if grown:
        print(f"grew eval golden set by {grown} real records -> {args.golden}", file=sys.stderr)
    if len(combined) < args.target:
        print(f"WARNING: only {len(combined)}/{args.target}. Generate more synthetic "
              f"(datagen.py --n ...) or ingest more real articles.", file=sys.stderr)


if __name__ == "__main__":
    main()
