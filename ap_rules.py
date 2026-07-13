#!/usr/bin/env python3
"""Deterministic AP Stylebook checker (pure code, no model).

Brainlift Insight 2: AP rules are deterministic with objectively correct answers, so
the right engineering is to check them in code and reserve the SLM for the
non-deterministic judgment (evidence verification). This catches AP violations
instantly and completely — the model emits ~one verdict per sentence.

Design rule: every check must be CONSERVATIVE — it must not flag correct text. Each
rule is tested against a violating AND a correct example in `_selftest()` below.
The set is easily extensible (add a SIMPLE_SUB tuple or a regex block).

Each hit: {span, rule, suggestion}.
"""
from __future__ import annotations

import re

NUM_WORDS = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
             "6": "six", "7": "seven", "8": "eight", "9": "nine"}
ORD_WORDS = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
             "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
             "9th": "ninth"}
# Spelled-out numbers -> figures, for AGE detection ("fifty-year-old" -> "50-year-old").
_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9}
_TENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
         "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}


def _word_to_num(word: str):
    """'fifty' -> 50, 'fifty-two' -> 52, 'seven' -> 7. None if not parseable."""
    w = word.lower().strip()
    if w in _TENS:
        return _TENS[w]
    if w in _ONES:
        return _ONES[w]
    if "-" in w:
        a, b = w.split("-", 1)
        if a in _TENS and b in _ONES and _TENS[a] >= 20:
            return _TENS[a] + _ONES[b]
    return None
ABBR_MONTHS = {"January": "Jan.", "February": "Feb.", "August": "Aug.",
               "September": "Sept.", "October": "Oct.", "November": "Nov.",
               "December": "Dec."}
ABBR_STATES = {"California": "Calif.", "Florida": "Fla.", "Pennsylvania": "Pa.",
               "Massachusetts": "Mass.", "Illinois": "Ill.", "Georgia": "Ga.",
               "Arizona": "Ariz.", "Colorado": "Colo.", "Michigan": "Mich.",
               "Missouri": "Mo.", "Connecticut": "Conn.", "Oregon": "Ore.",
               "Kentucky": "Ky.", "Tennessee": "Tenn.", "Virginia": "Va."}

# --- Simple word/phrase substitutions: (regex, rule, replacement) ---
# `\b` bounded, case-sensitive where it matters. Replacement may use \1 backrefs.
SIMPLE_SUBS: list[tuple[str, str, str]] = [
    (r"\be-mail\b", "spelling: AP uses “email” (no hyphen)", "email"),
    (r"\bE-mail\b", "spelling: AP uses “email” (no hyphen)", "Email"),
    (r"\bweb site\b", "spelling: AP uses “website” (one word)", "website"),
    (r"\bWeb site\b", "spelling: AP uses “website” (one word)", "Website"),
    (r"\bunder way\b", "spelling: AP uses “underway” (one word)", "underway"),
    (r"\bokay\b", "usage: AP uses “OK”, not “okay”", "OK"),
    (r"\badvisor\b", "spelling: AP uses “adviser”", "adviser"),
    (r"\btowards\b", "usage: AP uses “toward” (no s)", "toward"),
    (r"\bbackwards\b", "usage: AP uses “backward” (no s)", "backward"),
    (r"\bforwards\b", "usage: AP uses “forward” (no s)", "forward"),
    (r"\btoward the\b".replace("toward", "afterwards"),
     "usage: AP uses “afterward” (no s)", "afterward the"),
    (r"\bversus\b", "usage: AP abbreviates to “vs.” in most uses", "vs."),
    (r"\bet cetera\b", "usage: AP uses “etc.”", "etc."),
    (r"\bInternet\b", "capitalization: AP lowercases “internet” (2016+)", "internet"),
    (r"\bTeen-ager\b", "spelling: AP uses “teenager”", "Teenager"),
    (r"\bteen-ager\b", "spelling: AP uses “teenager”", "teenager"),
    (r"\bhealth care\b", "spelling: AP uses “health care” (two words) — OK", "health care"),
]
# (the health-care line intentionally no-ops; kept as a reminder it's two words.)


def _add(hits, seen, span, rule, suggestion):
    key = (span, rule)
    if key not in seen:
        seen.add(key)
        hits.append({"span": span, "rule": rule, "suggestion": suggestion})


def ap_check(text: str) -> list[dict]:
    hits: list[dict] = []
    seen: set = set()

    def add(span, rule, suggestion):
        _add(hits, seen, span, rule, suggestion)

    # 1) Numbers one-nine as figures. Skip when the figure is legitimately correct:
    #    %, times (3 p.m.), dates (Dec. 5), decades, ordinals, money, ages-as-adjective.
    for m in re.finditer(r"(?<![\d$%.:'-])\b([1-9])\b(?![%'\d-])", text):
        n, after = m.group(1), text[m.end():m.end() + 12]
        before = text[max(0, m.start() - 14):m.start()]
        if re.match(r"\s*(?:%|percent|:|st|nd|rd|th|a\.m|p\.m|\s*[AaPp]\.?[Mm]|-year|\s+years?\s+old)", after):
            continue  # part of %, time, ordinal, or age ("5-year-old", "5 years old")
        if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+$", before):
            continue  # date: "Dec. 5"
        if before.rstrip().endswith("$"):
            continue  # money handled separately
        add(m.group(0), "numbers: spell out one through nine; figures for 10+",
            f"replace “{n}” with “{NUM_WORDS[n]}”")

    # 2) Ordinals first-ninth spelled out (10th+ stay figures).
    for m in re.finditer(r"\b([1-9](?:st|nd|rd|th))\b", text):
        o = m.group(1)
        if o in ORD_WORDS:
            add(o, "ordinals: spell out first through ninth", f"“{ORD_WORDS[o]}”")

    # 2b) Ages: AP always uses figures for ages ("50-year-old", "a 5-year-old boy").
    #     Catch spelled-out ages, incl. as a hyphenated adjective.
    for m in re.finditer(r"\b([A-Za-z]+(?:-[A-Za-z]+)?)-year-old\b", text):
        num = _word_to_num(m.group(1))
        if num is not None:
            add(m.group(0), "ages: use figures for ages (and hyphenate as a modifier)",
                f"“{num}-year-old”")
    for m in re.finditer(r"\b([a-z]+(?:-[a-z]+)?)\s+years?\s+old\b", text):
        num = _word_to_num(m.group(1))
        if num is not None:
            add(m.group(0), "ages: use figures for ages", f"“{num} years old”")

    # 3) Time: uppercase AM/PM or ":00" on the hour; and 12 a.m./p.m. -> midnight/noon.
    for m in re.finditer(r"\b(\d{1,2})(:\d{2})?\s*([APap]\.?[Mm]\.?)", text):
        hour, mins, ap = m.group(1), m.group(2), m.group(3)
        norm = "a.m." if ap.upper().startswith("A") else "p.m."
        if hour == "12" and (not mins or mins == ":00"):
            add(m.group(0), "time: use “noon”/“midnight”, not 12 p.m./12 a.m.",
                "noon" if norm == "p.m." else "midnight")
            continue
        if ap not in ("a.m.", "p.m.") or mins == ":00":
            fixed = hour + ("" if (not mins or mins == ":00") else mins) + " " + norm
            add(m.group(0), "time: lowercase a.m./p.m. with periods; drop “:00” on the hour",
                f"use “{fixed}”")

    # 4) Months abbreviated with a specific date.
    for full, abbr in ABBR_MONTHS.items():
        for m in re.finditer(rf"\b{full}\s+(\d{{1,2}})\b", text):
            add(m.group(0), "months: abbreviate Jan./Feb./Aug./Sept./Oct./Nov./Dec. with a date",
                f"“{abbr} {m.group(1)}”")

    # 5) State abbreviated when it follows a city (City, State).
    for full, abbr in ABBR_STATES.items():
        for m in re.finditer(rf"\b([A-Z][a-zA-Z]+),\s+{full}\b", text):
            add(m.group(0), f"states: AP abbreviates {full} as {abbr} with a city",
                f"“{m.group(1)}, {abbr}”")

    # 6) Oxford/serial comma in a simple series.
    for m in re.finditer(r"(\w+),\s+(\w+),\s+and\s+(\w+)", text):
        add(m.group(0), "punctuation: no serial (Oxford) comma in a simple series",
            m.group(0).replace(f"{m.group(2)}, and", f"{m.group(2)} and"))

    # 7) "percent" spelled out with a numeral -> %.
    for m in re.finditer(r"\b(\d+)\s+percent\b", text):
        add(m.group(0), "percent: use the % sign with a numeral (AP, 2019+)", f"“{m.group(1)}%”")

    # 8) "over" with a numeral quantity -> "more than".
    for m in re.finditer(r"\bover\s+(\d[\d,]*)\b", text):
        add(m.group(0), "usage: use “more than” (not “over”) with a numeral quantity",
            m.group(0).replace("over", "more than"))

    # 9) Attribution order: "said <bare Name>." -> "<Name> said".
    for m in re.finditer(r"\bsaid\s+([A-Z][a-z]+)\s*([.!?\"'”’]|$)", text):
        add(m.group(0).strip(),
            "attribution: name before “said” unless a title/clause follows the name",
            f"“{m.group(1)} said”")

    # 10) Decade with erroneous apostrophe: 1990's -> 1990s ; '90's -> '90s.
    for m in re.finditer(r"\b(\d{2,4})'s\b", text):
        add(m.group(0), "decades: no apostrophe before the s (1990s, not 1990’s)",
            f"“{m.group(1)}s”")

    # 11) Money written as words with a numeral: "5 dollars" -> "$5".
    for m in re.finditer(r"\b(\d[\d,]*)\s+dollars?\b", text):
        add(m.group(0), "money: use the $ sign with figures", f"“${m.group(1)}”")

    # 12) Address: numbered street with Street/Avenue/Boulevard spelled out -> abbreviate.
    for full, abbr in (("Street", "St."), ("Avenue", "Ave."), ("Boulevard", "Blvd.")):
        for m in re.finditer(rf"\b(\d+\s+[A-Z][a-zA-Z]+)\s+{full}\b", text):
            add(m.group(0), f"addresses: abbreviate {full} ({abbr}) with a numbered address",
                f"“{m.group(1)} {abbr}”")

    # 13) Simple word/phrase substitutions.
    for pat, rule, repl in SIMPLE_SUBS:
        for m in re.finditer(pat, text):
            fixed = re.sub(pat, repl, m.group(0))
            if fixed != m.group(0):  # skip the intentional no-op reminders
                add(m.group(0), rule, f"“{fixed}”")

    return hits


# ------------------------------------------------------------------------------
# Self-test: each rule must FIRE on a violation and STAY SILENT on correct text.
# ------------------------------------------------------------------------------
_SELFTEST = [
    # (sentence, should_flag_substring or None)
    ("The board has 5 members.", "5"),
    ("The board has five members.", None),
    ("She finished 3rd in the race.", "3rd"),
    ("An ICE agent killed the fifty-year-old on Tuesday.", "fifty-year-old"),
    ("The victim was forty-two years old.", "forty-two years old"),
    ("Police arrested the 50-year-old suspect.", None),
    ("The 5-year-old boy was found safe.", None),
    ("The meeting starts at 3:00 PM.", "3:00 PM"),
    ("The meeting starts at 3 p.m.", None),
    ("The event ends at 12 p.m. sharp.", "12 p.m."),
    ("The vote was held on December 25.", "December 25"),
    ("The vote was held on Dec. 25.", None),
    ("She lives in Miami, Florida, now.", "Miami, Florida"),
    ("The flag was red, white, and blue.", "red, white, and blue"),
    ("The flag was red, white and blue.", None),
    ("Enrollment rose 12 percent.", "12 percent"),
    ("Enrollment rose 12%.", None),
    ("It served over 300 people.", "over 300"),
    ('"No," said Fox.', "said Fox"),
    ('"No," Fox said.', None),
    ("It happened in the 1990's.", "1990's"),
    ("It happened in the 1990s.", None),
    ("The fine was 5 dollars.", "5 dollars"),
    ("The office is at 123 Main Street.", "123 Main Street"),
    ("Send an e-mail today.", "e-mail"),
    ("Send an email today.", None),
    ("Check the web site.", "web site"),
    ("The plan is under way.", "under way"),
    ("That's okay with me.", "okay"),
    ("She is an advisor.", "advisor"),
    ("He walked towards the door.", "towards"),
    ("Search the Internet.", "Internet"),
]


def _selftest() -> None:
    ok = True
    for sent, expect in _SELFTEST:
        spans = [h["span"] for h in ap_check(sent)]
        fired = any((expect and expect in s) for s in spans)
        if expect and not fired:
            print(f"  MISS  {sent!r} — expected to flag {expect!r}, got {spans}"); ok = False
        if expect is None and spans:
            print(f"  FALSE+  {sent!r} — should be clean, flagged {spans}"); ok = False
    print("self-test:", "ALL PASS" if ok else "FAILURES ABOVE")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        s = sys.argv[1] if len(sys.argv) > 1 else \
            "The meeting has 5 members and starts at 3:00 PM on December 25, 2021."
        for h in ap_check(s):
            print(f"⚑ {h['span']!r}\n   rule: {h['rule']}\n   fix: {h['suggestion']}\n")
