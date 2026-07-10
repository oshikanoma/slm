#!/usr/bin/env python3
"""Deterministic AP Stylebook example generator (NO API needed).

AP style rules are deterministic with objectively correct answers (Brainlift
Insight 2), so we can synthesize training/eval data by construction: build a passage
that VIOLATES a specific AP rule, and emit the gold `ap_style`/`ap_flag` verdict that
flags it and suggests the AP-correct fix — WITHOUT rewriting the passage (the fix
lives only in the `suggestion` field, preserving the model's no-rewrite guarantee).

Rules covered (verified against current AP guidance, incl. 2019 % change and the
five-letter month rule):
  - numbers:  spell out one-nine; figures for 10+
  - time:     lowercase a.m./p.m. with periods; no ":00" on the hour
  - months:   abbreviate Jan./Feb./Aug./Sept./Oct./Nov./Dec. with a date; never
              March-July
  - oxford:   no serial comma in a simple series
  - percent:  use "%" with a numeral (not the word "percent")

Each record is a self-contained scenario in the SAME schema eval.py uses, so it
merges straight into the training/golden sets. `sources` is empty (AP is checked
against the rule, not a source). Verdicts pass verdict_structurally_valid for
type=ap_style.

USAGE
  python gen_apstyle.py --n 160 --out data/apstyle.jsonl --seed 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Callable

# ---- ingredients -------------------------------------------------------------
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine"}
ABBR_MONTHS = {"January": "Jan.", "February": "Feb.", "August": "Aug.",
               "September": "Sept.", "October": "Oct.", "November": "Nov.",
               "December": "Dec."}
NEVER_ABBR = ["March", "April", "May", "June", "July"]
SUBJECTS = ["the city council", "the school board", "the committee", "the agency",
            "the department", "the team", "the panel", "the board", "the county"]
NOUNS = ["members", "seats", "projects", "buildings", "programs", "vehicles",
         "officers", "grants", "reports", "sites"]
ITEMS = ["red", "blue", "green", "yellow", "silver", "gold", "black", "white",
         "copper", "bronze"]
CITY = ["Austin", "Houston", "Dallas", "El Paso", "Laredo", "Waco", "Tyler"]


def _rec(rid: str, passage: str, span: str, rule: str, suggestion: str, expl: str) -> dict:
    """One AP scenario in the eval schema. Empty sources (AP != evidence check)."""
    return {
        "id": rid, "bucket": "ap_style", "passage": passage, "sources": [],
        "gold_verdicts": [{
            "type": "ap_style", "span": span, "verdict": "ap_flag",
            "source_url": None, "evidence_span": None,
            "checked_source_url": None, "nearest_span": None,
            "rule": rule, "suggestion": suggestion, "explanation": expl,
        }],
    }


# ---- per-rule generators (return passage, span, rule, suggestion, expl) ------

def g_numbers(rng: random.Random):
    n = rng.randint(1, 9)  # violation: a number <10 written as a figure
    subj, noun = rng.choice(SUBJECTS), rng.choice(NOUNS)
    span = f"{subj.capitalize()} has {n} {noun}."
    passage = span
    fix = f"{subj.capitalize()} has {NUM_WORDS[n]} {noun}."
    return passage, span, "numbers: spell out whole numbers one through nine; use figures for 10 and above", fix, \
        f"AP spells out numbers below 10, so '{n}' should be '{NUM_WORDS[n]}'."


def g_time(rng: random.Random):
    h = rng.randint(1, 12)
    style = rng.choice(["AM", "PM", "A.M.", "P.M."])
    hour = f"{h}:00"
    span = f"The meeting starts at {hour} {style} sharp."
    passage = span
    ap = f"{h} {'a.m.' if 'A' in style.upper() else 'p.m.'}"
    fix = f"The meeting starts at {ap} sharp."
    return passage, span, "time: use lowercase a.m./p.m. with periods; drop ':00' on the hour", fix, \
        f"AP uses '{ap}' — lowercase with periods and no ':00' on the hour."


def g_months(rng: random.Random):
    month = rng.choice(list(ABBR_MONTHS))
    day = rng.randint(1, 28)
    span = f"The vote was held on {month} {day}."
    passage = span
    fix = f"The vote was held on {ABBR_MONTHS[month]} {day}."
    return passage, span, "months: abbreviate Jan., Feb., Aug., Sept., Oct., Nov., Dec. when used with a date", fix, \
        f"With a specific date, AP abbreviates '{month}' to '{ABBR_MONTHS[month]}'."


def g_oxford(rng: random.Random):
    a, b, c = rng.sample(ITEMS, 3)
    span = f"The flag was {a}, {b}, and {c}."
    passage = span
    fix = f"The flag was {a}, {b} and {c}."
    return passage, span, "punctuation: no serial (Oxford) comma in a simple series", fix, \
        "AP omits the comma before 'and' in a simple series."


def g_percent(rng: random.Random):
    n = rng.randint(2, 80)
    span = f"Enrollment rose {n} percent this year."
    passage = span
    fix = f"Enrollment rose {n}% this year."
    return passage, span, "percent: use the % sign with a numeral (AP, 2019+)", fix, \
        f"AP now uses '{n}%' rather than spelling out 'percent'."


GENERATORS: list[tuple[str, Callable]] = [
    ("num", g_numbers), ("time", g_time), ("month", g_months),
    ("oxf", g_oxford), ("pct", g_percent),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deterministic AP-style example generator.")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--out", default="data/apstyle.jsonl")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    # validate each record against the real eval gate as we go
    from eval import Scenario, verdict_structurally_valid

    seen: set[str] = set()
    recs, i, attempts = [], 0, 0
    while len(recs) < args.n and attempts < args.n * 20:
        attempts += 1
        tag, gen = GENERATORS[len(recs) % len(GENERATORS)]  # even coverage
        passage, span, rule, suggestion, expl = gen(rng)
        if passage in seen:
            continue
        rec = _rec(f"ap_{tag}_{i:04d}", passage, span, rule, suggestion, expl)
        scn = Scenario(id=rec["id"], bucket="ap_style", passage=passage,
                       sources=[], gold_verdicts=rec["gold_verdicts"])
        if not verdict_structurally_valid(rec["gold_verdicts"][0], scn):
            continue
        seen.add(passage); recs.append(rec); i += 1

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_rule = Counter(r["id"].split("_")[1] for r in recs)
    print(f"wrote {len(recs)} AP-style records -> {args.out}", file=sys.stderr)
    print(f"rule coverage: {dict(by_rule)}", file=sys.stderr)


if __name__ == "__main__":
    main()
