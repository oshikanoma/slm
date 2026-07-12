#!/usr/bin/env python3
"""Deterministic AP Stylebook checker (pure code, no model).

Brainlift Insight 2: AP rules are deterministic with objectively correct answers, so
the RIGHT engineering is to check them in code and reserve the SLM for the
non-deterministic judgment (evidence verification). This module catches EVERY AP
violation in a sentence instantly — unlike the model, which emits ~one verdict per
sentence and prioritizes claim-verification on claim-like inputs.

Each hit: {span, rule, suggestion}. Rules mirror gen_apstyle.py and current AP
(incl. the 2019 % change and the five-letter month rule).

Usage:
  from ap_rules import ap_check
  hits = ap_check("The meeting has 5 members and starts at 3:00 PM on December 25.")
"""
from __future__ import annotations

import re

NUM_WORDS = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
             "6": "six", "7": "seven", "8": "eight", "9": "nine"}
ABBR_MONTHS = {"January": "Jan.", "February": "Feb.", "August": "Aug.",
               "September": "Sept.", "October": "Oct.", "November": "Nov.",
               "December": "Dec."}
# Months AP never abbreviates (<=5 letters): March, April, May, June, July.


def ap_check(text: str) -> list[dict]:
    hits: list[dict] = []
    seen: set[tuple] = set()

    def add(span, rule, suggestion):
        key = (span, rule)
        if key not in seen:
            seen.add(key)
            hits.append({"span": span, "rule": rule, "suggestion": suggestion})

    # 1) Numbers one-nine written as figures (skip years, ages-as-adjective, %, $, times).
    for m in re.finditer(r"(?<![\d$%.:])\b([1-9])\b(?!\s*(?:%|percent|a\.m|p\.m|:))", text):
        n = m.group(1)
        # skip if it's part of a date like "December 5" (that's fine in AP)
        before = text[max(0, m.start() - 12):m.start()]
        if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+$", before):
            continue
        add(m.group(0), "numbers: spell out one through nine; figures for 10+",
            f"replace “{n}” with “{NUM_WORDS[n]}”")

    # 2) Time: uppercase AM/PM, or ":00" on the hour.
    for m in re.finditer(r"\b(\d{1,2})(:\d{2})?\s*([APap]\.?[Mm]\.?)", text):
        hour, mins, ap = m.group(1), m.group(2), m.group(3)
        norm = "a.m." if ap.upper().startswith("A") else "p.m."
        fixed_time = hour + ("" if (not mins or mins == ":00") else mins) + " " + norm
        if ap not in ("a.m.", "p.m.") or (mins == ":00"):
            add(m.group(0), "time: lowercase a.m./p.m. with periods; drop “:00” on the hour",
                f"use “{fixed_time}”")

    # 3) Months abbreviated with a specific date.
    for full, abbr in ABBR_MONTHS.items():
        for m in re.finditer(rf"\b{full}\s+(\d{{1,2}})\b", text):
            add(m.group(0), "months: abbreviate Jan./Feb./Aug./Sept./Oct./Nov./Dec. with a date",
                f"“{abbr} {m.group(1)}”")

    # 4) Oxford/serial comma in a simple series: "..., X, and Y".
    for m in re.finditer(r"(\w+),\s+(\w+),\s+and\s+(\w+)", text):
        add(m.group(0), "punctuation: no serial (Oxford) comma in a simple series",
            m.group(0).replace(f"{m.group(2)}, and", f"{m.group(2)} and"))

    # 5) "percent" spelled out with a numeral -> % (AP 2019+).
    for m in re.finditer(r"\b(\d+)\s+percent\b", text):
        add(m.group(0), "percent: use the % sign with a numeral (AP, 2019+)",
            f"“{m.group(1)}%”")

    # 6) "over" used with a numeral quantity -> "more than".
    for m in re.finditer(r"\bover\s+(\d[\d,]*)\b", text):
        add(m.group(0), "usage: use “more than” (not “over”) with a numeral quantity",
            m.group(0).replace("over", "more than"))

    return hits


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else \
        "The meeting has 5 members and starts at 3:00 PM on December 25, 2021."
    for h in ap_check(s):
        print(f"⚑ {h['span']!r}\n   rule: {h['rule']}\n   fix: {h['suggestion']}\n")
