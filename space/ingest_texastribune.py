#!/usr/bin/env python3
"""Ingest REAL Texas Tribune articles into golden-set scaffolds for hand-labeling.

Why the Texas Tribune: it is a nonprofit newsroom that publishes under a Creative
Commons / free-republishing policy (https://www.texastribune.org/republishing-guidelines/),
so — unlike paywalled outlets (Statesman/Chronicle) or scrape-prohibited ones
(NYT/WaPo/AP) — its text can be reused with attribution. Attribution for every
record is preserved in the `_source_article`, `_article_title`, and `_byline` fields.

WHAT THIS DOES (and does NOT do):
- It harvests real (passage, real-linked-source) pairs. A news paragraph plus the
  page it hyperlinks to is a genuine claim+evidence pair — perfect for this task.
- It does NOT assign verdicts. `verdict` / `evidence_span` / `bucket` are left BLANK
  for you to fill by hand (the trustworthy way to build a golden EVAL set). Real
  journalism has almost no clean negative examples, so build your trap buckets
  (true_but_unsupported / distractor) with datagen.py; use these real records mainly
  for `supported` / `misleading` cases and eval realism.

PIPELINE
  RSS feed -> article URLs -> each article's body paragraphs -> keep paragraphs that
  contain a usable inline hyperlink (real source) -> fetch that source's text ->
  emit a scaffold record in the eval.py golden schema with blank verdict fields.

USAGE
  pip install requests beautifulsoup4
  python ingest_texastribune.py --max-records 30 --out data/golden.real.jsonl
  # then hand-label each record (see the printed instructions), and score with:
  #   python eval.py score --testset data/golden.real.jsonl --preds preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

FEED_MAIN = "https://feeds.texastribune.org/feeds/main/"
SITEMAP_INDEX = "https://www.texastribune.org/sitemap.xml"
UA = "Mozilla/5.0 (SLM class project; golden-set eval builder; contact: student)"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
ARTICLE_PATH = re.compile(r"^/\d{4}/\d{2}/\d{2}/")  # /YYYY/MM/DD/slug/
LOC_RE = re.compile(r"<loc>\s*([^<]+?)\s*</loc>")
POST_SITEMAP_RE = re.compile(r"post-sitemap\d+\.xml")

# Anchor text that is navigation/CTA noise, never a real source.
ANCHOR_BLOCK = {
    "sign up", "sign up for the brief", "donate now", "other", "get tickets today!",
    "the texas tribune festival", "learn more about membership", "list of them here",
    "join the list", "subscribe", "read more", "here",
}
# URL fragments that mark promo / social / bio links, not evidence.
URL_BLOCK = (
    "directory.texastribune.org", "support.texastribune.org", "trib.it",
    "/newsletters/", "/support-us/", "/about/", "/donate", "facebook.com/sharer",
    "twitter.com/intent", "x.com/intent", "/republishing", "utm_source=eoacta",
)


def fetch(url: str, timeout: int = 15) -> requests.Response:
    return requests.get(url, headers={"User-Agent": UA}, timeout=timeout)


def get_feed_article_urls(feed_url: str, limit: int) -> list[str]:
    """Parse the RSS/Atom feed and return item link URLs (newest first).

    Texas Tribune uses FeedPress tracking links (feeds.texastribune.org/link/...)
    that 302-redirect to the real article, so we return those as-is and resolve the
    canonical URL later via the redirect when fetching each article.
    """
    resp = fetch(feed_url)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    urls: list[str] = []
    for node in root.iter():
        if node.tag.split("}")[-1] not in ("item", "entry"):
            continue
        link_url = None
        for child in node:
            if child.tag.split("}")[-1] != "link":
                continue
            if child.text and child.text.strip().startswith("http"):  # RSS
                link_url = child.text.strip()
            elif child.get("href"):  # Atom
                link_url = child.get("href").strip()
        if link_url:
            urls.append(link_url)
        if len(urls) >= limit:
            break
    return urls


def get_sitemap_article_urls(limit: int, delay: float = 0.5) -> list[str]:
    """Discover article URLs from the WordPress sitemap index (newest first).

    The RSS feed only exposes the ~30 most recent stories; the sitemaps expose the
    full archive. We read the sitemap index, walk the `post-sitemap*.xml` children
    from the highest-numbered (most recent) down, and collect /YYYY/MM/DD/slug/ URLs.
    This is a robots-permitted, canonical discovery path (no crawling of the site
    itself), and every URL is a real Texas Tribune article page we then fetch once.
    """
    resp = fetch(SITEMAP_INDEX)
    resp.raise_for_status()
    child_maps = [u for u in LOC_RE.findall(resp.text) if POST_SITEMAP_RE.search(u)]

    def _map_num(u: str) -> int:
        m = re.search(r"post-sitemap(\d+)\.xml", u)
        return int(m.group(1)) if m else 0

    child_maps.sort(key=_map_num, reverse=True)  # newest sitemap first
    urls: list[str] = []
    seen: set[str] = set()
    for cm in child_maps:
        if len(urls) >= limit:
            break
        try:
            time.sleep(delay)
            r = fetch(cm)
            r.raise_for_status()
        except Exception as e:
            print(f"  [skip sitemap] {cm}: {e}", file=sys.stderr)
            continue
        # within a child sitemap, URLs are oldest->newest; reverse for newest-first
        locs = [u for u in LOC_RE.findall(r.text) if ARTICLE_PATH.match(urlparse(u).path)]
        for u in reversed(locs):
            cu = u.split("?")[0].split("#")[0]
            if cu in seen:
                continue
            seen.add(cu)
            urls.append(cu)
            if len(urls) >= limit:
                break
    return urls


def is_usable_source(href: str, anchor: str) -> bool:
    if not href.startswith("http"):
        return False
    low = href.lower()
    if any(b in low for b in URL_BLOCK):
        return False
    if anchor.strip().lower() in ANCHOR_BLOCK:
        return False
    if re.fullmatch(r"\$?\d+", anchor.strip()):  # "$18", "25"
        return False
    host = urlparse(href).netloc.lower()
    if host.endswith("texastribune.org"):
        # keep only real TT article pages as (secondary) sources
        return bool(ARTICLE_PATH.match(urlparse(href).path))
    return True  # external primary source


def extract_main_text(html: str, max_chars: int = 1800) -> str:
    """Pull readable body text from a source page (TT or external), truncated."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
        tag.decompose()
    container = soup.select_one("div.entry-content") or soup.find("article") or soup.body or soup
    parts = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = " ".join(t for t in parts if len(t) > 40)
    if not text:
        text = container.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def article_meta(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    title_tag = soup.select_one('meta[property="og:title"]')
    title = title_tag["content"].strip() if title_tag and title_tag.get("content") else (
        soup.title.get_text(strip=True) if soup.title else url)
    author_tag = (soup.select_one('meta[name="author"]')
                  or soup.select_one('meta[property="article:author"]'))
    byline = author_tag["content"].strip() if author_tag and author_tag.get("content") else ""
    return title, byline


def sentence_with(anchor: str, paragraph: str) -> str:
    for s in SENT_SPLIT.split(paragraph):
        if anchor and anchor[:30].lower() in s.lower():
            return s.strip()
    return paragraph.strip()


def records_from_article(url: str, per_article: int, delay: float) -> list[dict]:
    try:
        resp = fetch(url)  # follows the FeedPress redirect to the real article
        resp.raise_for_status()
    except Exception as e:
        print(f"  [skip article] {url}: {e}", file=sys.stderr)
        return []
    url = resp.url.split("?")[0]  # canonical article URL after redirect
    if not ARTICLE_PATH.match(urlparse(url).path):
        print(f"  [skip non-article] {url}", file=sys.stderr)
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    title, byline = article_meta(soup, url)
    body = soup.select_one("div.entry-content") or soup.find("article")
    if not body:
        return []

    out = []
    seen_para = set()
    for p in body.find_all("p"):
        if len(out) >= per_article:
            break
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(text) < 80 or len(text) > 700 or text in seen_para:
            continue
        if text.lower().startswith("disclosure:"):
            continue
        # first usable inline link in this paragraph
        link = next((a for a in p.find_all("a", href=True)
                     if is_usable_source(a["href"], a.get_text(strip=True))), None)
        if link is None:
            continue
        src_url = link["href"].split("#")[0]
        anchor = link.get_text(" ", strip=True)
        try:
            time.sleep(delay)
            sresp = fetch(src_url)
            src_text = extract_main_text(sresp.text) if "html" in sresp.headers.get(
                "Content-Type", "") else ""
        except Exception as e:
            print(f"    [source fetch failed] {src_url}: {e}", file=sys.stderr)
            src_text = ""
        seen_para.add(text)
        out.append({
            "id": None,  # filled by caller
            "bucket": "REVIEW",  # <-- set to supported|unsupported|true_but_unsupported|distractor|misleading
            "passage": text,
            "sources": [{"url": src_url, "text": src_text}],
            "gold_verdicts": [{
                "type": "claim",  # <-- claim | quote | link
                "span": sentence_with(anchor, text),  # <-- trim to the exact clause you're checking
                "verdict": "",  # <-- FILL: supported | unsupported | misleading
                "source_url": src_url,  # <-- set null if you decide the claim is unsupported
                "evidence_span": "",  # <-- FILL: verbatim substring of the source text that backs it
                "explanation": "",  # <-- one line
            }],
            "_source_article": url,
            "_article_title": title,
            "_byline": byline,
            "_anchor_text": anchor,
            "_note": "Hand-label: set bucket, verdict, span, evidence_span. If unsupported, set source_url & evidence_span to null.",
        })
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest Texas Tribune articles into golden-set scaffolds.")
    ap.add_argument("--source", choices=["feed", "sitemap"], default="sitemap",
                    help="URL discovery: 'sitemap' (full archive, hundreds+; default) "
                         "or 'feed' (RSS, ~30 newest).")
    ap.add_argument("--feed", default=FEED_MAIN, help="RSS feed URL (when --source feed).")
    ap.add_argument("--max-articles", type=int, default=200)
    ap.add_argument("--per-article", type=int, default=3, help="Max records per article (diversity).")
    ap.add_argument("--max-records", type=int, default=40)
    ap.add_argument("--delay", type=float, default=1.0, help="Politeness delay between requests (s).")
    ap.add_argument("--out", default="data/golden.real.json",
                    help="Output path. Use .json for a readable pretty array, .jsonl for one-per-line.")
    args = ap.parse_args(argv)

    if args.source == "sitemap":
        print(f"Discovering article URLs via sitemap index: {SITEMAP_INDEX}", file=sys.stderr)
        urls = get_sitemap_article_urls(args.max_articles, args.delay)
    else:
        print(f"Fetching feed: {args.feed}", file=sys.stderr)
        urls = get_feed_article_urls(args.feed, args.max_articles)
    print(f"  {len(urls)} article URLs", file=sys.stderr)

    records: list[dict] = []
    for i, url in enumerate(urls, 1):
        if len(records) >= args.max_records:
            break
        print(f"[{i}/{len(urls)}] {url}", file=sys.stderr)
        time.sleep(args.delay)
        for rec in records_from_article(url, args.per_article, args.delay):
            rec["id"] = f"tt_{len(records):04d}"
            records.append(rec)
            if len(records) >= args.max_records:
                break

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        if args.out.endswith(".json"):  # pretty array: easy to hand-label
            json.dump(records, f, ensure_ascii=False, indent=2)
        else:  # one object per line
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== ingest summary ===", file=sys.stderr)
    print(f"wrote {len(records)} scaffold records -> {args.out}", file=sys.stderr)
    print("NEXT: open the file and for each record fill in:", file=sys.stderr)
    print("  1. bucket           (supported | unsupported | true_but_unsupported | distractor | misleading)", file=sys.stderr)
    print("  2. gold_verdicts[0].verdict         (supported | unsupported | misleading)", file=sys.stderr)
    print("  3. gold_verdicts[0].span            (trim to the exact clause you're verifying)", file=sys.stderr)
    print("  4. gold_verdicts[0].evidence_span   (verbatim quote from the source that backs it;", file=sys.stderr)
    print("                                       set it AND source_url to null if unsupported)", file=sys.stderr)
    print("must_contain / expected_verdict auto-derive in eval.py once verdicts are filled.", file=sys.stderr)
    print("Attribution is preserved in _source_article / _article_title / _byline.", file=sys.stderr)


if __name__ == "__main__":
    main()
