#!/usr/bin/env python3
"""Migrate v1 dataset -> v2 contract (add checked_source_url + nearest_span).

v2 requires EVERY verdict to show the source it reviewed, even when flagging a claim
`unsupported`. This is filled in DETERMINISTICALLY with no API calls:

  - supported / misleading: checked_source_url = the existing source_url; nearest_span
    = the existing evidence_span (that IS the source it looked at).
  - unsupported: pick the closest source in the bundle by content-word overlap with
    the claim span (reusing autolabel.best_source_sentence), set checked_source_url to
    that source's url and nearest_span to the best-overlapping sentence in it.

Every migrated record is re-run through the v2 gate (datagen.passes_gate), so anything
that can't be made spec-valid is dropped rather than silently corrupting the set.

Rebuilds the SFT training file from the migrated scenarios so train.sft.jsonl matches.

USAGE
  python migrate_v2.py --train data/train.jsonl --golden data/golden.json \
      --sft-out data/train.sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from eval import Scenario, checked_source_is_valid, derive_golden_metadata, load_jsonl
from datagen import passes_gate, to_sft_record
from autolabel import best_source_sentence


def scn_from(rec: dict) -> Scenario:
    return Scenario(id=rec.get("id", "x"), bucket=rec.get("bucket", ""),
                    passage=rec.get("passage", ""), sources=rec.get("sources", []) or [],
                    gold_verdicts=rec.get("gold_verdicts", []) or [])


def closest_source(span: str, sources: list[dict]) -> tuple[str | None, str | None]:
    """Return (url, nearest_span) for the source with the highest overlap to `span`."""
    best_url, best_span, best_score = None, None, -1.0
    for s in sources:
        text = s.get("text") or ""
        if not text:
            continue
        score, sent = best_source_sentence(span, text)
        if sent and score > best_score:
            best_url, best_span, best_score = s.get("url"), sent, score
    # fall back to the first source's leading sentence if overlap found nothing usable
    if best_url is None and sources:
        s0 = sources[0]
        text = s0.get("text") or ""
        if text:
            best_url = s0.get("url")
            # first ~30-word chunk as a guaranteed verbatim substring
            best_span = " ".join(text.split()[:30])
    return best_url, best_span


def migrate_verdict(v: dict, scn: Scenario) -> dict:
    """Add checked_source_url + nearest_span to one verdict (in place-ish, returns it)."""
    v = dict(v)
    verdict = v.get("verdict")
    if verdict in ("supported", "misleading") and v.get("source_url") and v.get("evidence_span"):
        # it already looked at this source; reuse it verbatim
        v["checked_source_url"] = v["source_url"]
        v["nearest_span"] = v["evidence_span"]
    else:
        url, span = closest_source(v.get("span") or scn.passage, scn.sources)
        v["checked_source_url"] = url
        v["nearest_span"] = span
    return v


def migrate_record(rec: dict) -> tuple[dict | None, str]:
    scn = scn_from(rec)
    if not scn.gold_verdicts:
        return None, "no_verdicts"
    new_verdicts = [migrate_verdict(v, scn) for v in scn.gold_verdicts]
    rec = dict(rec)
    rec["gold_verdicts"] = new_verdicts
    scn2 = scn_from(rec)
    # verify every verdict now has a valid checked source
    for v in new_verdicts:
        if not checked_source_is_valid(v, scn2):
            return None, "checked_source_invalid"
    ok, why = passes_gate(scn2, rec.get("bucket", "supported"))
    if not ok:
        return None, f"gate:{why}"
    return rec, "ok"


def migrate_file(path: str, is_golden: bool) -> tuple[list[dict], Counter]:
    recs = load_jsonl(path)
    out, drops = [], Counter()
    for rec in recs:
        m, why = migrate_record(rec)
        if m is None:
            drops[why] += 1
            continue
        if is_golden:  # refresh derived metadata (must_contain now includes checked url)
            scn = scn_from(m)
            mc, mnc, kw, exp = derive_golden_metadata(scn)
            m.update(must_contain=mc, must_not_contain=mnc, keywords=kw,
                     expected_verdict=exp)
            m.setdefault("human_label", None)
        out.append(m)
    return out, drops


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Migrate dataset to the v2 checked_source contract.")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--golden", default="data/golden.json")
    ap.add_argument("--sft-out", default="data/train.sft.jsonl")
    args = ap.parse_args(argv)

    for path, is_golden in [(args.train, False), (args.golden, True)]:
        if not os.path.exists(path):
            print(f"skip missing {path}", file=sys.stderr)
            continue
        out, drops = migrate_file(path, is_golden)
        # write back to the same path (jsonl for train, pretty json for golden)
        with open(path, "w", encoding="utf-8") as f:
            if path.endswith(".json"):
                json.dump(out, f, ensure_ascii=False, indent=2)
            else:
                for r in out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: migrated {len(out)}  dropped {dict(drops)}", file=sys.stderr)

        # rebuild SFT from the migrated TRAIN set
        if not is_golden and args.sft_out:
            with open(args.sft_out, "w", encoding="utf-8") as f:
                for r in out:
                    f.write(json.dumps(to_sft_record(scn_from(r)), ensure_ascii=False) + "\n")
            print(f"{args.sft_out}: rebuilt {len(out)} SFT records", file=sys.stderr)

    print("v2 migration complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
