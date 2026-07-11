#!/usr/bin/env python3
"""Teacher-model data generation for the Cited Newsroom Verifier SLM.

Distills training examples from a frontier "teacher" model (any OpenAI-compatible
API) in the EXACT schema that eval.py enforces, then filters hard against the same
quality gate. Data that can't pass the gate never enters the dataset.

Each example is a self-contained scenario:
  passage + sources (with distractors) + gold verdicts (cited or flagged).

Buckets (the spec's §5.1 distribution, reused for training):
  supported            - a real source backs the claim; must cite it verbatim.
  unsupported          - no source backs it; must flag, cite nothing.
  true_but_unsupported - claim is TRUE in reality but ABSENT from sources -> flag.
  distractor           - a plausible-but-non-supporting source is present -> don't cite it.
  misleading           - quote altered / out of context / link doesn't support -> flag.

The true_but_unsupported and distractor buckets are the whole point: they teach
the model restraint (the forbidden failure = knowledge leakage / fabricated cites).

USAGE
  export OPENAI_API_KEY=sk-...
  python datagen.py --n 300 --out data/train.jsonl --sft-out data/train.sft.jsonl
  # then split off a held-out eval set:
  python datagen.py --n 80 --out data/testset.jsonl --seed 999

Requires: pip install openai
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter as MultiCounter
from typing import Optional

# Reuse the SAME validators/schema the eval uses, so data can't cheat the gate.
from eval import (
    Scenario,
    build_messages,
    checked_source_is_valid,
    citation_is_valid,
    derive_golden_metadata,
    extract_json,
    is_verbatim_substring,
    norm_url,
    VALID_TYPES,
    VALID_VERDICTS,
)

# ------------------------------------------------------------------------------
# Teacher client
# ------------------------------------------------------------------------------

def teacher_client(base_url: Optional[str]):
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise SystemExit("datagen needs the `openai` package: pip install openai")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY (the teacher/distillation model).")
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    # Optional LangSmith tracing: no-op unless LANGSMITH_TRACING=true + key are set.
    try:
        from langsmith.wrappers import wrap_openai  # type: ignore
        client = wrap_openai(client)
    except ImportError:
        pass
    return client


# ------------------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------------------

DATAGEN_SYSTEM = """You generate TRAINING DATA for a newsroom fact-verification model.

You produce one self-contained scenario as a single JSON object with these fields:
- "passage": a short news passage (1-3 sentences) containing one checkable claim, quote, or link.
- "sources": a list of 1-3 objects, each {"url": "...", "text": "..."} — the retrieved evidence bundle.
- "verdicts": a list with ONE verdict object for the checkable item in the passage:
    {"type": "claim|quote|link",
     "span": "<a span COPIED EXACTLY from the passage>",
     "verdict": "supported|unsupported|misleading",
     "source_url": "<a url COPIED EXACTLY from sources>" or null,
     "evidence_span": "<a span COPIED EXACTLY from that source's text>" or null,
     "checked_source_url": "<the url COPIED EXACTLY from sources that you reviewed / is closest to the claim>",
     "nearest_span": "<a span COPIED EXACTLY from that checked source's text — the closest content to the claim>",
     "explanation": "<one sentence>"}

HARD RULES (data is rejected if violated):
1. "span" MUST be an exact substring of "passage".
2. For verdict "supported": "source_url" MUST exactly equal one url in "sources",
   and "evidence_span" MUST be an exact substring of THAT source's "text", and it
   must genuinely back the claim.
3. For verdict "unsupported": "source_url" and "evidence_span" MUST be null, and
   NO source may actually contain the claimed fact.
4. ALWAYS (every verdict, including unsupported): "checked_source_url" MUST exactly
   equal one url in "sources", and "nearest_span" MUST be an exact substring of THAT
   source's "text". This shows which source was reviewed. On "supported" it is
   normally the same url as source_url; on "unsupported" it is the closest
   non-supporting source. Never invent a checked_source_url.
5. Use realistic-looking but FICTIONAL urls and facts (e.g. numbers, names). Do not
   rely on real-world knowledge being checkable — only the provided sources count.
6. Output ONLY the JSON object, no prose."""

BUCKET_INSTRUCTIONS = {
    "supported": (
        "Make a scenario where a provided source DIRECTLY supports the claim. "
        "verdict='supported' with a correct source_url and a verbatim evidence_span."
    ),
    "unsupported": (
        "Make a scenario where the passage states a specific fact (a number, date, "
        "cost, or superlative) that NONE of the provided sources contain. The sources "
        "should be on-topic but silent on that specific fact. verdict='unsupported', "
        "source_url=null, evidence_span=null."
    ),
    "true_but_unsupported": (
        "Make a scenario where the passage states something that is TRUE in the real "
        "world (common knowledge) but is NOT stated in any provided source. The sources "
        "are on a related topic but never assert the claim. The correct behavior is to "
        "refuse to vouch: verdict='unsupported', source_url=null, evidence_span=null. "
        "This teaches the model not to leak outside knowledge."
    ),
    "distractor": (
        "Make a scenario with 2-3 sources where one source LOOKS relevant (mentions the "
        "topic, or speculates/rumors) but does NOT actually confirm the specific claim, "
        "and no other source does either. The model must NOT cite the distractor. "
        "verdict='unsupported', source_url=null, evidence_span=null."
    ),
    "misleading": (
        "Make a scenario where the passage contains a QUOTE that is altered from, or "
        "stripped of context from, what the source transcript actually says (changing "
        "the meaning). Provide the true quote in a source. verdict='misleading', "
        "source_url=<that source>, evidence_span=<the true quote, verbatim from source>."
    ),
}

TOPIC_SEEDS = [
    "a city budget report", "a corporate earnings filing", "a public-health agency notice",
    "a court ruling summary", "a sports team announcement", "an election results bulletin",
    "a university research press release", "a transportation infrastructure update",
    "a labor statistics report", "a climate/weather agency statement",
    "a tech product launch", "a government official's press briefing",
    "a nonprofit annual report", "a police department incident report",
    "a housing market analysis", "a school board meeting recap",
]

DEFAULT_MIX = {
    "supported": 0.30,
    "unsupported": 0.25,
    "true_but_unsupported": 0.20,
    "distractor": 0.15,
    "misleading": 0.10,
}


# ------------------------------------------------------------------------------
# Quality gate (mirrors eval.py's checks + bucket-specific invariants)
# ------------------------------------------------------------------------------

def passes_gate(scn: Scenario, bucket: str) -> tuple[bool, str]:
    if not scn.passage.strip():
        return False, "empty passage"
    if not scn.sources:
        return False, "no sources"
    if len(scn.gold_verdicts) != 1:
        return False, f"expected 1 verdict, got {len(scn.gold_verdicts)}"

    v = scn.gold_verdicts[0]
    if v.get("type") not in VALID_TYPES:
        return False, f"bad type {v.get('type')!r}"
    if v.get("verdict") not in VALID_VERDICTS:
        return False, f"bad verdict {v.get('verdict')!r}"
    if not is_verbatim_substring(v.get("span"), scn.passage):
        return False, "span not verbatim in passage"

    # v2 contract: every verdict must show a real checked source it reviewed.
    if not checked_source_is_valid(v, scn):
        return False, "checked_source_url/nearest_span not verbatim / url not in sources"

    verdict = v.get("verdict")
    if verdict == "supported":
        if not citation_is_valid(v, scn):
            return False, "supported: citation not verbatim / url not in sources"
    elif verdict == "unsupported":
        if v.get("source_url") is not None or v.get("evidence_span") is not None:
            return False, "unsupported must have null citation"
        # soft check: the claim must NOT actually appear verbatim in any source
        for s in scn.sources:
            if is_verbatim_substring(v.get("span"), s.get("text")):
                return False, "unsupported but span appears in a source"
    elif verdict == "misleading":
        if v.get("source_url") is not None:
            if norm_url(v.get("source_url")) not in scn.source_urls:
                return False, "misleading: source_url not in sources"
            src = scn.source_text_for(v.get("source_url"))
            if not is_verbatim_substring(v.get("evidence_span"), src):
                return False, "misleading: evidence_span not verbatim in source"

    # bucket-specific: unsupported-family must not be `supported`
    if bucket in ("unsupported", "true_but_unsupported", "distractor") and verdict != "unsupported":
        return False, f"bucket {bucket} requires verdict=unsupported, got {verdict}"
    if bucket == "supported" and verdict != "supported":
        return False, f"bucket supported requires verdict=supported, got {verdict}"
    if bucket == "misleading" and verdict != "misleading":
        return False, f"bucket misleading requires verdict=misleading, got {verdict}"
    return True, "ok"


# ------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------

class FatalAPIError(Exception):
    """Non-retryable API failure (bad key, inactive billing) — abort, don't hammer."""


# Substrings that mean "stop immediately, retrying won't help."
_FATAL_MARKERS = (
    "billing_not_active", "account is not active", "invalid_api_key",
    "incorrect api key", "no such organization", "insufficient_quota",
    "401", "403",
)
# Substrings that mean "transient — back off and retry."
_RETRY_MARKERS = ("rate limit", "rate_limit", "429", "timeout", "timed out",
                  "temporarily", "overloaded", "503", "502", "connection")


def _create_with_retry(client, *, max_retries: int = 5, **kwargs):
    """Call chat.completions.create with backoff on rate limits; abort on fatal errors."""
    delay = 2.0
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — classify below
            msg = str(e).lower()
            if any(m in msg for m in _FATAL_MARKERS):
                raise FatalAPIError(str(e))
            last_exc = e
            if any(m in msg for m in _RETRY_MARKERS) and attempt < max_retries - 1:
                print(f"  [rate-limited, retrying in {delay:.0f}s] {str(e)[:120]}",
                      file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise
    if last_exc:
        raise last_exc


def gen_one(client, model: str, bucket: str, topic: str, temperature: float) -> Optional[Scenario]:
    user = (
        f"Topic: {topic}.\n"
        f"Bucket: {bucket}.\n"
        f"{BUCKET_INSTRUCTIONS[bucket]}\n"
        "Return the JSON object now."
    )
    try:
        resp = _create_with_retry(
            client,
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": DATAGEN_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
    except FatalAPIError:
        raise  # bubble up to main() for a clean abort
    except Exception as e:
        print(f"  [api error] {e}", file=sys.stderr)
        return None
    obj = extract_json(resp.choices[0].message.content or "")
    if not obj:
        return None
    return Scenario(
        id="tmp",
        bucket=bucket,
        passage=obj.get("passage", ""),
        sources=obj.get("sources", []) or [],
        gold_verdicts=obj.get("verdicts", []) or [],
    )


def pick_bucket(mix: dict[str, float], rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for b, w in mix.items():
        acc += w
        if r <= acc:
            return b
    return next(iter(mix))


# ------------------------------------------------------------------------------
# JUNK mode: schema-valid scenarios WITHOUT any API (Day-2 loop smoke test)
# ------------------------------------------------------------------------------

_TRUE_FACTS = [
    "Water boils at 100 degrees Celsius at sea level.",
    "The Pacific is the largest ocean on Earth.",
    "The human heart has four chambers.",
    "Light travels faster than sound.",
    "Mount Everest is the tallest mountain above sea level.",
    "A triangle has three sides.",
]


def gen_junk(bucket: str, i: int, rng: random.Random) -> Scenario:
    """Deterministically build a gate-passing scenario, no teacher call needed."""
    n = rng.randint(1, 999)
    amt = rng.randint(1, 900)
    year = rng.choice([2021, 2022, 2023, 2024])
    topic = rng.choice(["budget", "jobs report", "vaccine trial", "bridge project"])
    url = f"https://source{n}.example/doc{i}"

    if bucket == "supported":
        span = f"revenue rose to ${amt} million in {year}"
        passage = f"City report {i}: {span}, officials said."
        src = f"Filing {n}: {span}, up from the previous fiscal year."
        return Scenario("tmp", bucket, passage,
            [{"url": url, "text": src}],
            [{"type": "claim", "span": span, "verdict": "supported",
              "source_url": url, "evidence_span": span, "explanation": "junk-supported"}])

    if bucket == "unsupported":
        span = f"The {topic} cost ${amt} million to complete"
        passage = f"{span}, according to officials."
        src = f"Report {n}: the {topic} began in {year} and employed {rng.randint(50,900)} people."
        return Scenario("tmp", bucket, passage,
            [{"url": url, "text": src}],
            [{"type": "claim", "span": span, "verdict": "unsupported",
              "source_url": None, "evidence_span": None, "explanation": "junk-unsupported"}])

    if bucket == "true_but_unsupported":
        fact = rng.choice(_TRUE_FACTS)
        span = fact.rstrip(".")
        passage = f"The article noted: {fact}"
        src = f"Report {n}: the {topic} for {year} listed unrelated logistics figures."
        return Scenario("tmp", bucket, passage,
            [{"url": url, "text": src}],
            [{"type": "claim", "span": span, "verdict": "unsupported",
              "source_url": None, "evidence_span": None, "explanation": "junk-true-but-unsupported"}])

    if bucket == "distractor":
        span = f"doubled its revenue in {year}"
        passage = f"Company{n} {span}, the report said."
        url2 = f"https://rumor{n}.example/post{i}"
        return Scenario("tmp", bucket, passage,
            [{"url": url, "text": f"Company{n} announced a new CEO in {year}."},
             {"url": url2, "text": f"Analysts speculate Company{n} may have grown revenue, but figures are unconfirmed."}],
            [{"type": "claim", "span": span, "verdict": "unsupported",
              "source_url": None, "evidence_span": None, "explanation": "junk-distractor"}])

    # misleading
    true_quote = f'"We are cautiously optimistic about the {topic}," the official said'
    span = f'"We are thrilled about the {topic}," the official said.'
    passage = span
    src = f"Transcript {n}: {true_quote} at the briefing."
    return Scenario("tmp", bucket, passage,
        [{"url": url, "text": src}],
        [{"type": "quote", "span": span, "verdict": "misleading",
          "source_url": url, "evidence_span": true_quote, "explanation": "junk-misleading"}])


def to_sft_record(scn: Scenario) -> dict:
    """Chat-format record for SFT: system+user prompt with the gold JSON as target."""
    messages = build_messages(scn)
    target = {
        "clean": len(scn.gold_verdicts) == 0,
        "verdicts": scn.gold_verdicts,
    }
    messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return {"id": scn.id, "messages": messages}


def to_scenario_record(scn: Scenario) -> dict:
    """Golden-set record: inputs + outputs + auto-derived metadata (must_contain,
    must_not_contain, keywords, expected_verdict) so every record is gradeable."""
    mc, mnc, kw, exp = derive_golden_metadata(scn)
    return {
        "id": scn.id,
        "bucket": scn.bucket,
        "passage": scn.passage,
        "sources": scn.sources,
        "gold_verdicts": scn.gold_verdicts,
        "must_contain": mc,
        "must_not_contain": mnc,
        "keywords": kw,
        "expected_verdict": exp,
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate cited-verifier training data.")
    ap.add_argument("--n", type=int, required=True, help="Number of ACCEPTED examples to produce.")
    ap.add_argument("--out", required=True, help="Scenario JSONL (same schema as testset / eval).")
    ap.add_argument("--sft-out", help="Also write chat-format SFT JSONL for training.")
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "gpt-4o"))
    ap.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="Cap total teacher calls (0 = 4x --n).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--junk", action="store_true",
                    help="Generate schema-valid JUNK locally (no API). Day-2 loop smoke test.")
    ap.add_argument("--only-bucket", default=None, choices=list(DEFAULT_MIX),
                    help="Generate ONLY this bucket (e.g. 'supported' to fix over-caution).")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    client = None if args.junk else teacher_client(args.base_url)
    max_attempts = args.max_attempts or (args.n * 4)

    accepted: list[Scenario] = []
    reject_reasons: MultiCounter = MultiCounter()
    accepted_buckets: MultiCounter = MultiCounter()
    attempts = 0
    t0 = time.time()

    while len(accepted) < args.n and attempts < max_attempts:
        attempts += 1
        bucket = args.only_bucket or pick_bucket(DEFAULT_MIX, rng)
        topic = rng.choice(TOPIC_SEEDS)
        if args.junk:
            scn = gen_junk(bucket, len(accepted), rng)
        else:
            try:
                scn = gen_one(client, args.model, bucket, topic, args.temperature)
            except FatalAPIError as e:
                raise SystemExit(
                    "\nFATAL: the teacher API rejected the request and retrying won't help:\n"
                    f"  {e}\n\n"
                    "Common causes:\n"
                    "  - OpenAI 'billing_not_active': your org has no API credits (ask your admin),\n"
                    "    or the key is under a different org. This is NOT a code problem.\n"
                    "  - Bad/rotated key.\n\n"
                    "Free alternatives (no OpenAI billing): point datagen at another provider:\n"
                    "  export TEACHER_BASE_URL='https://api.groq.com/openai/v1'   # free Groq key\n"
                    "  export TEACHER_MODEL='llama-3.3-70b-versatile'\n"
                    "  export OPENAI_API_KEY='gsk_...'\n"
                    "or run a local Ollama teacher (TEACHER_BASE_URL=http://localhost:11434/v1)."
                )
        if scn is None:
            reject_reasons["unparseable"] += 1
            continue
        ok, reason = passes_gate(scn, bucket)
        if not ok:
            reject_reasons[reason] += 1
            continue
        scn.id = f"{bucket[:3]}_{len(accepted):04d}"
        accepted.append(scn)
        accepted_buckets[bucket] += 1
        if len(accepted) % 10 == 0 or len(accepted) == args.n:
            print(f"  accepted {len(accepted)}/{args.n} "
                  f"(attempts {attempts}, {time.time()-t0:.0f}s)", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for scn in accepted:
            f.write(json.dumps(to_scenario_record(scn), ensure_ascii=False) + "\n")
    if args.sft_out:
        with open(args.sft_out, "w", encoding="utf-8") as f:
            for scn in accepted:
                f.write(json.dumps(to_sft_record(scn), ensure_ascii=False) + "\n")

    # Report
    yield_rate = len(accepted) / attempts if attempts else 0
    print("\n=== datagen summary ===", file=sys.stderr)
    print(f"accepted: {len(accepted)}  attempts: {attempts}  yield: {yield_rate:.0%}", file=sys.stderr)
    print(f"bucket mix (accepted): {dict(accepted_buckets)}", file=sys.stderr)
    print(f"top reject reasons: {dict(reject_reasons.most_common(8))}", file=sys.stderr)
    print(f"wrote: {args.out}" + (f" , {args.sft_out}" if args.sft_out else ""), file=sys.stderr)
    if len(accepted) < args.n:
        print(f"WARNING: only produced {len(accepted)}/{args.n}. Raise --max-attempts "
              f"or loosen prompts.", file=sys.stderr)


if __name__ == "__main__":
    main()
