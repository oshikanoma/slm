#!/usr/bin/env python3
"""Ingest REAL English Wikinews articles into golden-set scaffolds for labeling.

Why Wikinews: every article is released under Creative Commons Attribution 2.5
(CC BY 2.5, https://en.wikinews.org/wiki/Wikinews:Copyright), so — like the Texas
Tribune — its text is legally reusable with attribution, unlike paywalled or
scrape-prohibited outlets (AP/NYT/WaPo/Reuters/etc.). Attribution for each record
is preserved in `_source_article`, `_article_title`, and `_license` fields.

WHAT THIS DOES (mirrors ingest_texastribune.py):
- Harvests real (passage, candidate-source-bundle) pairs. A Wikinews article's body
  paragraph plus the external sources it cites is a genuine claim+evidence pair.
- Does NOT assign verdicts. `verdict` / `evidence_span` / `bucket` are left blank for
  autolabel.py (or a human) to fill. Wikinews lists its sources as {{source}}
  templates at the foot of the article rather than inline anchors, so — unlike the
  TT ingester — the WHOLE source set becomes the candidate bundle for each passage.
  That matches the spec's "source bundle with distractors" shape: the model must
  read every source and decide which (if any) backs the claim.

PIPELINE
  Category:Published -> article titles -> for each: plaintext extract (paragraphs)
  + external citation URLs -> fetch each source's readable text -> emit one scaffold
  record per usable body paragraph, in the eval.py golden schema, verdict blank.

USAGE
  pip install requests beautifulsoup4
  python ingest_wikinews.py --max-articles 120 --per-article 2 --out data/wikinews.real.json
  python autolabel.py --in data/wikinews.real.json --out data/wikinews.real.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from urllib.parse import urlparse

import requests

# Reuse the TT ingester's readable-text extractor and source filters — same job.
from ingest_texastribune import (
    ANCHOR_BLOCK,  # noqa: F401  (kept for parity / future anchor use)
    URL_BLOCK,
    extract_main_text,
)

API = "https://en.wikinews.org/w/api.php"
UA = "Mozilla/5.0 (SLM class project; CC-BY dataset builder; contact: student)"
LICENSE = "CC BY 2.5 (https://en.wikinews.org/wiki/Wikinews:Copyright)"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
DATELINE = re.compile(r"^\w+,\s+\w+\s+\d{1,2},\s+\d{4}\s*$")  # "Sunday, May 3, 2026"

# Wikinews-specific link noise: share widgets, the project's own pages, archives of
# itself, and mail links are never real external evidence.
WN_URL_BLOCK = URL_BLOCK + (
    "wikinews.org", "wikimedia.org", "wikipedia.org", "creativecommons.org",
    "web.archive.org/web/", "mailto:", "facebook.com/sharer", "twitter.com/intent",
    "x.com/intent", "/Special:", "reddit.com/submit", "t.me/share",
)


def api_get(session: requests.Session, params: dict, timeout: int = 25) -> dict:
    params = {**params, "action": params.get("action", "query"), "format": "json"}
    r = session.get(API, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def published_titles(session: requests.Session, limit: int) -> list[str]:
    """Newest-first list of published Wikinews article titles."""
    titles: list[str] = []
    cont: dict = {}
    while len(titles) < limit:
        data = api_get(session, {
            "list": "categorymembers", "cmtitle": "Category:Published",
            "cmlimit": "500", "cmtype": "page", "cmsort": "timestamp",
            "cmdir": "desc", **cont,
        })
        titles += [c["title"] for c in data["query"]["categorymembers"]]
        if "continue" in data:
            cont = data["continue"]
        else:
            break
    return titles[:limit]


def article_extract(session: requests.Session, title: str) -> str:
    data = api_get(session, {
        "prop": "extracts", "explaintext": "1", "titles": title, "redirects": "1",
    })
    pages = data["query"]["pages"]
    for pg in pages.values():
        return pg.get("extract", "") or ""
    return ""


def article_source_urls(session: requests.Session, title: str) -> list[str]:
    """External citation URLs for an article, minus share/self/archive noise."""
    urls: list[str] = []
    cont: dict = {}
    while True:
        data = api_get(session, {
            "prop": "extlinks", "titles": title, "ellimit": "200",
            "redirects": "1", **cont,
        })
        for pg in data["query"]["pages"].values():
            for e in pg.get("extlinks", []) or []:
                u = e.get("*") or e.get("url") or ""
                if u.startswith("//"):
                    u = "https:" + u
                if not u.startswith("http"):
                    continue
                low = u.lower()
                if any(b in low for b in WN_URL_BLOCK):
                    continue
                urls.append(u.split("#")[0])
        if "continue" in data:
            cont = data["continue"]
        else:
            break
    # de-dup, preserve order
    seen: set[str] = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def body_paragraphs(extract: str) -> list[str]:
    """Split a plaintext extract into usable body paragraphs.

    Drops the leading dateline, section headers, and the trailing Sources/Related
    boilerplate that Wikinews appends.
    """
    paras: list[str] = []
    for block in extract.split("\n"):
        p = re.sub(r"\s+", " ", block).strip()
        if not p or DATELINE.match(p):
            continue
        # section headers in extracts look like "Sources" / "Related news"
        if p.lower() in ("sources", "related news", "external links", "references"):
            break  # everything after Sources is boilerplate
        if len(p) < 80 or len(p) > 700:
            continue
        paras.append(p)
    return paras


def build_bundle(session: requests.Session, source_urls: list[str], max_sources: int,
                 delay: float) -> list[dict]:
    """Fetch readable text for up to max_sources citation URLs -> source bundle."""
    bundle: list[dict] = []
    for u in source_urls:
        if len(bundle) >= max_sources:
            break
        try:
            time.sleep(delay)
            resp = requests.get(u, headers={"User-Agent": UA}, timeout=15)
            if "html" not in resp.headers.get("Content-Type", ""):
                continue
            text = extract_main_text(resp.text)
        except Exception as e:
            print(f"    [source fetch failed] {u}: {e}", file=sys.stderr)
            continue
        if len(text) < 120:  # too thin to support anything
            continue
        bundle.append({"url": u, "text": text})
    return bundle


def records_from_article(session: requests.Session, title: str, per_article: int,
                         max_sources: int, delay: float) -> list[dict]:
    extract = article_extract(session, title)
    if not extract:
        return []
    paras = body_paragraphs(extract)
    if not paras:
        return []
    src_urls = article_source_urls(session, title)
    if not src_urls:
        return []  # no external evidence -> not a usable claim+source pair
    bundle = build_bundle(session, src_urls, max_sources, delay)
    if not bundle:
        return []

    art_url = "https://en.wikinews.org/wiki/" + title.replace(" ", "_")
    out = []
    for para in paras[:per_article]:
        # default span = first sentence of the paragraph (autolabel/human trims it)
        first_sent = SENT_SPLIT.split(para)[0].strip()
        out.append({
            "id": None,  # filled by caller
            "bucket": "REVIEW",
            "passage": para,
            "sources": bundle,  # whole citation bundle (natural distractors)
            "gold_verdicts": [{
                "type": "claim",
                "span": first_sent if len(first_sent) >= 20 else para,
                "verdict": "",
                "source_url": bundle[0]["url"],
                "evidence_span": "",
                "explanation": "",
            }],
            "_source_article": art_url,
            "_article_title": title,
            "_byline": "Wikinews contributors",
            "_license": LICENSE,
            "_note": "Hand-label or autolabel: set bucket, verdict, span, evidence_span. "
                     "If unsupported, set source_url & evidence_span to null.",
        })
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest CC BY 2.5 Wikinews articles into golden-set scaffolds.")
    ap.add_argument("--max-articles", type=int, default=120)
    ap.add_argument("--per-article", type=int, default=2, help="Max records per article.")
    ap.add_argument("--max-records", type=int, default=300)
    ap.add_argument("--max-sources", type=int, default=3, help="Max sources per bundle.")
    ap.add_argument("--delay", type=float, default=0.6, help="Politeness delay (s).")
    ap.add_argument("--out", default="data/wikinews.real.json",
                    help="Output path (.json pretty array, or .jsonl one-per-line).")
    args = ap.parse_args(argv)

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    print("Fetching published Wikinews titles ...", file=sys.stderr)
    titles = published_titles(session, args.max_articles)
    print(f"  {len(titles)} titles", file=sys.stderr)

    records: list[dict] = []
    for i, title in enumerate(titles, 1):
        if len(records) >= args.max_records:
            break
        print(f"[{i}/{len(titles)}] {title}", file=sys.stderr)
        time.sleep(args.delay)
        try:
            recs = records_from_article(session, title, args.per_article,
                                        args.max_sources, args.delay)
        except Exception as e:
            print(f"  [skip article] {title}: {e}", file=sys.stderr)
            continue
        for rec in recs:
            rec["id"] = f"wn_{len(records):04d}"
            records.append(rec)
            if len(records) >= args.max_records:
                break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        if args.out.endswith(".json"):
            json.dump(records, f, ensure_ascii=False, indent=2)
        else:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== wikinews ingest summary ===", file=sys.stderr)
    print(f"wrote {len(records)} scaffold records -> {args.out}", file=sys.stderr)
    print(f"license: {LICENSE}", file=sys.stderr)
    print("NEXT: run autolabel.py to assign evidence-based verdicts, then build_dataset.py.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
