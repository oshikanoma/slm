#!/usr/bin/env python3
"""Evidence-based auto-labeler for the real Texas Tribune golden scaffolds.

The naive shortcut "if the Tribune linked it, the claim is supported" is WRONG often
enough to poison a golden set (see tt_0000: the linked page is about a different
shooting). This labeler instead decides each verdict from whether the SOURCE TEXT
actually contains supporting content for the claim `span`:

  - Find the source sentence with the highest content-word overlap with the claim,
    with a hard requirement that any NUMBER in the claim also appear in that sentence
    (numbers are the crux of most factual claims).
  - High overlap  -> verdict `supported`, evidence_span = that source sentence.
  - Low overlap   -> verdict `unsupported`, null citation (the link doesn't back it).
  - Middle band   -> `supported` but flagged LOW-CONFIDENCE for you to spot-check.

Everything is then run through eval.py's objective gate (span verbatim in passage,
evidence verbatim in source, url in sources), so nothing fabricated slips through.
No API key, no network needed.

USAGE
  python autolabel.py --in data/golden.real.json --out data/golden.real.json
  # review only the flagged records it prints, then score:
  #   python eval.py score --testset data/golden.real.json --preds preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from eval import citation_is_valid, extract_json, is_verbatim_substring, norm, Scenario

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z0-9]+")
NUM = re.compile(r"\d[\d,\.]*")

STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "who", "whom", "which", "what", "will", "would", "said",
    "has", "have", "had", "not", "no", "than", "then", "so", "also", "into", "out",
    "up", "down", "over", "after", "before", "about", "more", "most", "some", "any",
    "one", "two", "there", "when", "where", "how", "you", "we", "i", "our",
}

HIGH = 0.55  # >= -> supported
LOW = 0.30   # < -> unsupported; between -> supported but low-confidence


def content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text) if w.lower() not in STOP and len(w) > 2}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}


def best_source_sentence(span: str, source_text: str) -> tuple[float, str]:
    """Return (overlap_score, best_sentence) for the claim span vs source sentences."""
    span_words = content_words(span)
    span_nums = numbers(span)
    if not span_words:
        return 0.0, ""
    best_score, best_sent = 0.0, ""
    for sent in SENT_SPLIT.split(source_text):
        sent = sent.strip()
        if len(sent) < 15:
            continue
        sw = content_words(sent)
        if not sw:
            continue
        recall = len(span_words & sw) / len(span_words)
        # A claim with numbers must share at least one number to count as supported.
        if span_nums and not (span_nums & numbers(sent)):
            recall *= 0.4
        if recall > best_score:
            best_score, best_sent = recall, sent
    return best_score, best_sent


TEACHER_LABEL_SYSTEM = """You label fact-check training data. You are given a PASSAGE
claim (a SPAN to check) and ONE candidate SOURCE (url + text). Decide whether the
SOURCE TEXT supports the specific claim in the SPAN.

Rules (evidence-bounded — real-world truth is irrelevant, only the source text counts):
- "supported": the source text directly backs the claim. Set evidence_span to a quote
  copied VERBATIM (exact substring) from the source text that backs it.
- "unsupported": the source does not actually back the specific claim (even if related).
  Set evidence_span to null.
- "misleading": the span is a quote altered from / stripped of context vs. the source.
  Set evidence_span to the true quote, verbatim from the source.

Return ONLY JSON: {"verdict":"supported|unsupported|misleading","type":"claim|quote|link","evidence_span":"<verbatim source substring>"|null,"explanation":"one line"}."""


def teacher_label(rec: dict, client, model: str) -> str:
    """Label one record with the teacher model, then verify with the objective gate."""
    gv = rec["gold_verdicts"][0]
    passage = rec.get("passage", "")
    span = gv.get("span") or passage
    if not is_verbatim_substring(span, passage):
        span, gv["span"] = passage, passage
    sources = rec.get("sources") or []
    if not sources or not sources[0].get("text"):
        return "no_source"
    src_url, src_text = sources[0].get("url"), sources[0]["text"]
    user = (f"PASSAGE:\n{passage}\n\nSPAN TO CHECK:\n{span}\n\n"
            f"SOURCE url: {src_url}\nSOURCE text:\n{src_text}\n\nReturn the JSON now.")
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": TEACHER_LABEL_SYSTEM},
                      {"role": "user", "content": user}])
        obj = extract_json(resp.choices[0].message.content or "") or {}
    except Exception as e:
        print(f"  [teacher error] {rec.get('id')}: {e}", file=sys.stderr)
        obj = {}

    verdict = obj.get("verdict")
    gv["type"] = obj.get("type", "claim")
    gv["explanation"] = obj.get("explanation", "")
    if verdict in ("supported", "misleading"):
        gv.update(verdict=verdict, source_url=src_url, evidence_span=obj.get("evidence_span"))
        if not citation_is_valid(gv, scn_from(rec)):  # gate: caught fabrication
            gv.update(verdict="unsupported", source_url=None, evidence_span=None)
            rec["bucket"] = "unsupported"
            return "unsupported"
        rec["bucket"] = verdict
        return verdict
    gv.update(verdict="unsupported", source_url=None, evidence_span=None)
    rec["bucket"] = "unsupported"
    return "unsupported"


def scn_from(rec: dict) -> Scenario:
    return Scenario(id=rec.get("id", "x"), bucket=rec.get("bucket", ""),
                    passage=rec.get("passage", ""), sources=rec.get("sources", []) or [],
                    gold_verdicts=rec.get("gold_verdicts", []) or [])


def label_record(rec: dict) -> str:
    """Mutate rec's gold_verdicts + bucket in place. Return a status tag."""
    gv = rec["gold_verdicts"][0]
    passage = rec.get("passage", "")
    span = gv.get("span") or passage
    if not is_verbatim_substring(span, passage):
        span = passage  # fall back to the whole paragraph if the span drifted
        gv["span"] = span
    sources = rec.get("sources") or []
    src_text = sources[0].get("text", "") if sources else ""
    src_url = sources[0].get("url") if sources else None

    if not src_text:
        return "no_source"  # handled by caller (set aside)

    score, evidence = best_source_sentence(span, src_text)

    def set_unsupported():
        gv.update(verdict="unsupported", source_url=None, evidence_span=None,
                  explanation="No provided source sentence backs the specific claim.")
        rec["bucket"] = "unsupported"

    if score >= LOW and evidence:
        # exact verbatim substring of the source (whitespace-normalized check)
        gv.update(verdict="supported", source_url=src_url, evidence_span=evidence,
                  explanation="Source sentence overlaps the claim's key terms/numbers.")
        rec["bucket"] = "supported"
        # objective gate: the citation must really check out
        if not citation_is_valid(gv, scn_from(rec)):
            set_unsupported()
            return "unsupported"
        return "supported" if score >= HIGH else "supported_lowconf"
    set_unsupported()
    return "unsupported"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evidence-based auto-labeler for real golden scaffolds.")
    ap.add_argument("--in", dest="inp", default="data/golden.real.json")
    ap.add_argument("--out", default="data/golden.real.json")
    ap.add_argument("--excluded-out", default="data/golden.real.excluded.json",
                    help="Records with no usable source text land here (paywalled/blocked).")
    ap.add_argument("--teacher", action="store_true",
                    help="Use the teacher model (needs OPENAI_API_KEY) for paraphrase-aware labels.")
    ap.add_argument("--model", default="gpt-4o", help="Teacher model for --teacher mode.")
    args = ap.parse_args(argv)

    client = None
    if args.teacher:
        from datagen import teacher_client
        client = teacher_client(None)

    with open(args.inp, encoding="utf-8") as f:
        records = json.load(f)

    labeled, excluded = [], []
    status_by_id: dict[str, str] = {}
    tally: dict[str, int] = {}
    for rec in records:
        if not rec.get("gold_verdicts"):
            excluded.append(rec)
            continue
        status = (teacher_label(rec, client, args.model) if client
                  else label_record(rec))
        tally[status] = tally.get(status, 0) + 1
        if status == "no_source":
            excluded.append(rec)
        else:
            status_by_id[rec["id"]] = status
            labeled.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    if excluded:
        with open(args.excluded_out, "w", encoding="utf-8") as f:
            json.dump(excluded, f, ensure_ascii=False, indent=2)

    print("=== autolabel summary ===", file=sys.stderr)
    print(f"labeled: {len(labeled)}  excluded(no source text): {len(excluded)}", file=sys.stderr)
    print(f"breakdown: {tally}", file=sys.stderr)
    print(f"wrote -> {args.out}" + (f" , {args.excluded_out}" if excluded else ""), file=sys.stderr)

    # The records most worth a human glance: the middle-band supported ones and any
    # unsupported (in case the source genuinely backs it in paraphrase we missed).
    priority = [r for r in labeled if status_by_id[r["id"]] in ("supported_lowconf", "unsupported")]
    print(f"\nSPOT-CHECK ({len(priority)} of {len(labeled)}) — auto-labeler was unsure here:",
          file=sys.stderr)
    for r in priority:
        gv = r["gold_verdicts"][0]
        print(f"  {r['id']} [{status_by_id[r['id']]}] -> {gv['verdict']}", file=sys.stderr)
        print(f"      claim:    {gv['span'][:90]!r}", file=sys.stderr)
        if gv.get("evidence_span"):
            print(f"      evidence: {gv['evidence_span'][:90]!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
