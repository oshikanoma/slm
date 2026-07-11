#!/usr/bin/env python3
"""Diagnose WHY spec_pass fails, per record: is it a wrong VERDICT (a real model
judgment error, fixable with data) or merely a citation/span-exactness miss (right
call, imperfect quote)? Also flags spurious extra verdicts. This separates the
fixable signal from the strict-metric noise.

USAGE
  python diagnose.py --testset data/golden.json --preds tuned_preds.v4.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict, Counter
from eval import (load_testset, load_predictions, record_spec_pass,
                  find_pred_match, citation_is_valid, verdict_structurally_valid)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="data/golden.json")
    ap.add_argument("--preds", required=True)
    args = ap.parse_args(argv)

    scns = load_testset(args.testset)
    preds = load_predictions(args.preds)

    cat = Counter()
    by_bucket = defaultdict(Counter)
    examples = defaultdict(list)

    for scn in scns:
        pred = preds.get(scn.id)
        passed = record_spec_pass(scn, pred)
        b = scn.bucket
        by_bucket[b]["total"] += 1
        if passed:
            cat["pass"] += 1; by_bucket[b]["pass"] += 1
            continue

        pv = pred.verdicts if (pred and pred.parsed) else []
        if pred is None or not pred.parsed:
            cat["invalid_json"] += 1; by_bucket[b]["invalid_json"] += 1
            continue

        # classify the PRIMARY reason this record failed
        reason = None
        # 1) any gold verdict got the wrong label?
        for g in scn.gold_verdicts:
            pm = find_pred_match(g, pv)
            if pm is None:
                reason = "missed_verdict"; break
            if pm.get("verdict") != g.get("verdict"):
                reason = "wrong_verdict"; break
        # 2) verdicts all right, but a supported citation isn't verbatim-valid?
        if reason is None:
            for g in scn.gold_verdicts:
                pm = find_pred_match(g, pv)
                if g.get("verdict") == "supported" and pm and not citation_is_valid(pm, scn):
                    reason = "bad_span_or_citation"; break
        # 3) structurally invalid verdict (missing field, non-verbatim span)
        if reason is None and not all(verdict_structurally_valid(v, scn) for v in pv):
            reason = "structural"
        # 4) spurious extra flag on a clean/all-supported record
        if reason is None:
            reason = "spurious_or_meta"

        cat[reason] += 1; by_bucket[b][reason] += 1
        if len(examples[reason]) < 3:
            examples[reason].append(scn.id)

    total = sum(by_bucket[b]["total"] for b in by_bucket)
    print(f"=== Failure diagnosis over {total} records ===\n")
    print("Overall reason breakdown:")
    for r, n in cat.most_common():
        print(f"  {r:22} {n:4}  ({100*n/total:.0f}%)")

    print("\nThe key split:")
    fixable = cat["wrong_verdict"] + cat["missed_verdict"]
    metric = cat["bad_span_or_citation"] + cat["structural"] + cat["spurious_or_meta"] + cat["invalid_json"]
    print(f"  REAL judgment errors (data-fixable): {fixable} ({100*fixable/total:.0f}%)")
    print(f"  Citation/span/format strictness:     {metric} ({100*metric/total:.0f}%)")

    print("\nPer-bucket:")
    for b in sorted(by_bucket):
        c = by_bucket[b]
        parts = [f"{k}={c[k]}" for k in c if k not in ("total",)]
        print(f"  {b:22} pass {c['pass']}/{c['total']:2}  | " + "  ".join(parts))

    print("\nExample failing ids by reason:")
    for r, ids in examples.items():
        print(f"  {r}: {ids}")


if __name__ == "__main__":
    main()
