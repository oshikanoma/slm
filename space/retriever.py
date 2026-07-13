#!/usr/bin/env python3
"""Thin retrieval layer for the Cited Newsroom Verifier SLM (spec §8).

Turns a passage into a source BUNDLE the model can verify against:
    passage -> search(query) -> top-K urls -> fetch+extract -> [{url, text}]

The model is unchanged and stays evidence-bounded: it only ever quotes text this
layer places in front of it, so it cannot fabricate. The retriever does the
"source hunting"; the SLM does the disciplined verification.

Search backend is PLUGGABLE and auto-selected by which API key is present:
  - Brave     : env BRAVE_API_KEY      (free tier ~2k/mo; brave.com/search/api)
  - Serper     : env SERPER_API_KEY     (serper.dev)
  - Tavily     : env TAVILY_API_KEY     (tavily.com)
  - DuckDuckGo : no key (free HTML endpoint; lower quality, default fallback)

Fetch + readable-text extraction REUSE the already-tested code in
ingest_texastribune.py (fetch, extract_main_text).

USAGE (as a library)
  from retriever import build_bundle
  sources = build_bundle("Unemployment fell to 3.2% in March, officials said.")
  # -> [{"url": "...", "text": "..."}, ...]  ready to drop into the model prompt
"""
from __future__ import annotations

import os
import re
import sys
import time
from urllib.parse import quote_plus, urlparse

import requests

from ingest_texastribune import fetch, extract_main_text

UA = "Mozilla/5.0 (SLM verifier retriever; contact: student)"
_WS = re.compile(r"\s+")
_QUOTED = re.compile(r"[\"“](.+?)[\"”]")
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "said", "that", "this",
    "his", "her", "their", "they", "it", "its", "he", "she", "we", "you", "will",
}


# ------------------------------------------------------------------------------
# Query construction: distill a passage into a good search query
# ------------------------------------------------------------------------------

def passage_to_query(passage: str, max_terms: int = 12) -> str:
    """Prefer a quoted span (great for quote checks); else keep salient words."""
    passage = _WS.sub(" ", passage or "").strip()
    q = _QUOTED.search(passage)
    if q and len(q.group(1).split()) >= 3:
        return q.group(1)[:200]
    words = re.findall(r"[A-Za-z0-9$%.]+", passage)
    salient = [w for w in words if w.lower() not in _STOP and (len(w) > 2 or any(c.isdigit() for c in w))]
    return " ".join(salient[:max_terms]) or passage[:120]


# ------------------------------------------------------------------------------
# Pluggable search backends -> list of URLs
# ------------------------------------------------------------------------------

def _search_brave(query: str, k: int) -> list[str]:
    key = os.environ["BRAVE_API_KEY"]
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params={"q": query, "count": k}, timeout=15)
    r.raise_for_status()
    return [item["url"] for item in r.json().get("web", {}).get("results", [])][:k]


def _search_serper(query: str, k: int) -> list[str]:
    key = os.environ["SERPER_API_KEY"]
    r = requests.post("https://google.serper.dev/search",
                      headers={"X-API-KEY": key, "Content-Type": "application/json"},
                      json={"q": query, "num": k}, timeout=15)
    r.raise_for_status()
    return [o["link"] for o in r.json().get("organic", [])][:k]


def _search_tavily(query: str, k: int) -> list[str]:
    """Tavily search. Free tier, no credit card. Key: env TAVILY_API_KEY (tvly-...).
    Auth via Bearer header (current API); api_key in body also accepted."""
    key = os.environ["TAVILY_API_KEY"]
    r = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": query, "max_results": k, "search_depth": "basic"}, timeout=25)
    r.raise_for_status()
    return [item["url"] for item in r.json().get("results", [])][:k]


def _search_wikipedia(query: str, k: int) -> list[str]:
    """No-key, no-card fallback via the MediaWiki search API. Real citable article
    URLs (encyclopedia-scoped, not open web). Reliable where DuckDuckGo HTML is not."""
    r = requests.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": k}, headers={"User-Agent": UA}, timeout=15)
    r.raise_for_status()
    out = []
    for h in r.json().get("query", {}).get("search", []):
        out.append("https://en.wikipedia.org/wiki/" + quote_plus(h["title"].replace(" ", "_")))
    return out[:k]


def _search_duckduckgo(query: str, k: int) -> list[str]:
    """No-key fallback: DuckDuckGo HTML endpoint. Lower quality but free."""
    r = requests.get("https://html.duckduckgo.com/html/",
                     params={"q": query}, headers={"User-Agent": UA}, timeout=15)
    r.raise_for_status()
    # result links look like /l/?uddg=<encoded real url> or direct https
    urls = re.findall(r'href="(https?://[^"]+)"', r.text)
    from urllib.parse import unquote
    out = []
    for u in urls:
        m = re.search(r"uddg=([^&]+)", u)
        real = unquote(m.group(1)) if m else u
        host = urlparse(real).netloc
        if host and "duckduckgo.com" not in host and real.startswith("http"):
            out.append(real.split("#")[0])
        if len(out) >= k:
            break
    return out


def _active_backend() -> tuple[str, callable]:
    # Prefer an open-web API key if present; else free Wikipedia (no key/card).
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily", _search_tavily
    if os.environ.get("BRAVE_API_KEY"):
        return "brave", _search_brave
    if os.environ.get("SERPER_API_KEY"):
        return "serper", _search_serper
    return "wikipedia", _search_wikipedia


def search(query: str, k: int = 5) -> list[str]:
    name, fn = _active_backend()
    try:
        urls = fn(query, k)
    except Exception as e:
        print(f"  [search:{name}] {e}", file=sys.stderr)
        urls = []
    # de-dup preserving order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


# ------------------------------------------------------------------------------
# Build the source bundle
# ------------------------------------------------------------------------------

def build_bundle(passage: str, k: int = 4, per_source_chars: int = 1800,
                 delay: float = 0.4, verbose: bool = False) -> list[dict]:
    """passage -> [{url, text}] bundle the model can verify against.

    Fetches up to k search hits, extracts readable text, drops sources that fail to
    fetch or are too thin. The bundle intentionally may include non-supporting
    sources (natural distractors), which is exactly what the model must read past.
    """
    query = passage_to_query(passage)
    if verbose:
        name, _ = _active_backend()
        print(f"  [retriever] backend={name} query={query!r}", file=sys.stderr)
    urls = search(query, k=k + 2)  # over-fetch; some will fail to extract
    bundle: list[dict] = []
    for u in urls:
        if len(bundle) >= k:
            break
        try:
            time.sleep(delay)
            resp = fetch(u)
            if "html" not in resp.headers.get("Content-Type", ""):
                continue
            text = extract_main_text(resp.text, max_chars=per_source_chars)
        except Exception as e:
            if verbose:
                print(f"    [fetch fail] {u}: {e}", file=sys.stderr)
            continue
        if len(text) < 120:
            continue
        bundle.append({"url": u, "text": text})
    return bundle


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Build a source bundle for a passage.")
    ap.add_argument("passage", help="The passage/claim to retrieve sources for.")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()
    b = build_bundle(args.passage, k=args.k, verbose=True)
    print(json.dumps(b, ensure_ascii=False, indent=2))
    print(f"\n{len(b)} sources", file=sys.stderr)
