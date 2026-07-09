#!/usr/bin/env python3
"""Evaluation harness for the Cited, Evidence-Bounded Newsroom Verifier SLM.

Implements the eval defined in BEHAVIOR_SPEC_AND_EVAL.md (§5):
  - Deterministic behavioral checks (§5.2): the hard numbers, no LLM required.
  - Optional LLM-as-judge (§5.3): rubric scoring via an OpenAI-compatible API.
  - Base-vs-tuned comparison (§5.4): run two prediction sets side by side.
  - Optional generation (transformers) so you can produce predictions end to end.

The forbidden failure this harness is built to catch has two faces:
  A. Fabricated citation  -> a `supported` verdict whose source_url/evidence_span
     is not verbatim in the provided sources.
  B. Knowledge leakage    -> marking a claim `supported` when no provided source
     backs it (measured on the true-but-unsupported trap bucket).

--------------------------------------------------------------------------------
FILE FORMATS

Test set (JSONL), one scenario per line:
  {
    "id": "s001",
    "bucket": "supported|unsupported|true_but_unsupported|distractor|misleading",
    "passage": "full article text ...",
    "sources": [{"url": "https://...", "text": "source text ..."}, ...],
    "gold_verdicts": [
      {"type": "claim|quote|link", "span": "<verbatim substring of passage>",
       "verdict": "supported|unsupported|misleading",
       "source_url": "https://..." | null,
       "evidence_span": "<verbatim substring of the cited source>" | null}
    ]
  }

Predictions (JSONL), one per line. Either raw model text or structured:
  {"id": "s001", "output": "<raw model string containing a JSON object>"}
  {"id": "s001", "clean": false, "verdicts": [ ... ]}   # already-parsed form

--------------------------------------------------------------------------------
USAGE
  # Score one model:
  python eval.py score --testset data/testset.jsonl --preds base_preds.jsonl

  # Base vs tuned table (the money shot):
  python eval.py score --testset data/testset.jsonl \
      --base base_preds.jsonl --tuned tuned_preds.jsonl --out results.md

  # Add the LLM-as-judge dimensions (needs OPENAI_API_KEY):
  python eval.py score --testset data/testset.jsonl --base b.jsonl --tuned t.jsonl --judge

  # Produce the exact input prompts (to feed a model / your data-gen):
  python eval.py render --testset data/testset.jsonl --out prompts.jsonl

  # Generate predictions with a HF model (needs transformers/torch installed):
  python eval.py generate --testset data/testset.jsonl \
      --model Qwen/Qwen3-1.7B-Instruct --out base_preds.jsonl

  # Calibrate the judge to grade like you (the domain expert):
  python eval.py calibrate-export --testset data/golden.jsonl \
      --base base_preds.jsonl --tuned tuned_preds.jsonl --n 25 --out calib.jsonl
  #   ... fill in each "human_spec_pass": yes/no in calib.jsonl, then:
  python eval.py calibrate --labels calib.jsonl --judge-model gpt-4o

HEADLINE METRIC: spec_pass_rate — the fraction of golden-set records the model gets
fully right (binary pass/fail per record). Base-vs-tuned reports also include
McNemar's exact test + a bootstrap CI so the improvement is statistically defensible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Iterable, Optional

# ------------------------------------------------------------------------------
# Text normalization & matching helpers
# ------------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def norm(s: Optional[str]) -> str:
    """Whitespace-normalized, stripped text for lenient verbatim/substring checks."""
    if not s:
        return ""
    return _WS.sub(" ", s).strip()


def norm_url(u: Optional[str]) -> str:
    if not u:
        return ""
    return u.strip().rstrip("/").lower()


def is_verbatim_substring(needle: Optional[str], haystack: Optional[str]) -> bool:
    """True if `needle` appears in `haystack`, ignoring only whitespace differences."""
    n, h = norm(needle), norm(haystack)
    if not n or not h:
        return False
    return n in h


def spans_match(a: Optional[str], b: Optional[str], min_len: int = 4) -> bool:
    """Two passage spans refer to the same thing (lenient containment either way)."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) < min_len or len(nb) < min_len:
        return False
    return na in nb or nb in na


# ------------------------------------------------------------------------------
# Robust JSON extraction from raw model output
# ------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """Best-effort parse of a single JSON object out of arbitrary model text."""
    if text is None:
        return None
    text = text.strip()
    # 1) whole string is JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # 2) fenced ```json ... ``` block
    m = _FENCE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    # 3) first balanced-brace object
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None


# ------------------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------------------

FLAG_VERDICTS = {"unsupported", "misleading"}
VALID_TYPES = {"claim", "quote", "link"}
VALID_VERDICTS = {"supported", "unsupported", "misleading"}


@dataclass
class Scenario:
    id: str
    bucket: str
    passage: str
    sources: list[dict]
    gold_verdicts: list[dict]
    # Golden-set metadata (lecture: golden set = inputs + outputs + metadata).
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    expected_verdict: Optional[str] = None
    human_label: Optional[dict] = None  # {"spec_pass": bool, ...} you fill in

    def source_text_for(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        want = norm_url(url)
        for s in self.sources:
            if norm_url(s.get("url")) == want:
                return s.get("text", "")
        return None

    @property
    def source_urls(self) -> set[str]:
        return {norm_url(s.get("url")) for s in self.sources if s.get("url")}


def derive_golden_metadata(scn: "Scenario") -> tuple[list[str], list[str], list[str], Optional[str]]:
    """Auto-derive golden-set metadata from the gold verdict.

    - must_contain: any cited source_url (output must reference the real evidence).
    - must_not_contain: "http" when the expected verdict is `unsupported` (an
      unsupported verdict cites nothing, so no URL should appear in the output).
    - keywords: verification type + bucket, for slicing metrics.
    - expected_verdict: the single gold verdict label.
    """
    must_contain: list[str] = []
    must_not_contain: list[str] = []
    keywords: list[str] = []
    expected: Optional[str] = None
    if scn.gold_verdicts:
        g = scn.gold_verdicts[0]
        expected = g.get("verdict")
        if g.get("source_url"):
            must_contain.append(g["source_url"])
        if expected == "unsupported":
            must_not_contain.append("http")
        keywords = [k for k in (g.get("type"), scn.bucket) if k]
    return must_contain, must_not_contain, keywords, expected


@dataclass
class Prediction:
    id: str
    parsed: bool
    clean: bool
    verdicts: list[dict] = field(default_factory=list)
    raw: str = ""


# ------------------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load records from either JSONL (one object per line) OR a pretty-printed JSON
    array (multi-line, indented). The array form is far easier to hand-edit, so the
    golden set can live as `.json`; both forms are accepted everywhere."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if content.lstrip().startswith("["):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}: invalid JSON array: {e}")
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON array of objects")
        return data
    rows = []
    for ln, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{ln}: invalid JSON line: {e}")
    return rows


def load_testset(path: str) -> list[Scenario]:
    """Load the golden set. Missing metadata fields are auto-derived so older
    files (without must_contain/expected_verdict) still work."""
    out = []
    for r in load_jsonl(path):
        scn = Scenario(
            id=str(r["id"]),
            bucket=r.get("bucket", "unspecified"),
            passage=r.get("passage", ""),
            sources=r.get("sources", []) or [],
            gold_verdicts=r.get("gold_verdicts", []) or [],
            human_label=r.get("human_label"),
        )
        d_mc, d_mnc, d_kw, d_exp = derive_golden_metadata(scn)
        scn.must_contain = r.get("must_contain") or d_mc
        scn.must_not_contain = r.get("must_not_contain") or d_mnc
        scn.keywords = r.get("keywords") or d_kw
        scn.expected_verdict = r.get("expected_verdict") or d_exp
        out.append(scn)
    return out


# Golden-set loading is the same as the test set; `load_golden` is a friendlier alias.
load_golden = load_testset


def load_predictions(path: str) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    for r in load_jsonl(path):
        pid = str(r["id"])
        if "output" in r and isinstance(r["output"], str):
            obj = extract_json(r["output"])
            if obj is None:
                preds[pid] = Prediction(id=pid, parsed=False, clean=False, raw=r["output"])
                continue
        elif "verdicts" in r or "clean" in r:
            obj = {"clean": r.get("clean", False), "verdicts": r.get("verdicts", [])}
        else:
            preds[pid] = Prediction(id=pid, parsed=False, clean=False, raw=json.dumps(r))
            continue
        verdicts = obj.get("verdicts", [])
        if not isinstance(verdicts, list):
            verdicts = []
        preds[pid] = Prediction(
            id=pid,
            parsed=True,
            clean=bool(obj.get("clean", len(verdicts) == 0)),
            verdicts=verdicts,
            raw=r.get("output", ""),
        )
    return preds


# ------------------------------------------------------------------------------
# Per-verdict structural validation
# ------------------------------------------------------------------------------

def verdict_structurally_valid(v: dict, scn: Scenario) -> bool:
    if not isinstance(v, dict):
        return False
    if v.get("type") not in VALID_TYPES:
        return False
    if v.get("verdict") not in VALID_VERDICTS:
        return False
    if not is_verbatim_substring(v.get("span"), scn.passage):
        return False
    if v.get("verdict") == "supported":
        # supported REQUIRES a non-null citation
        return bool(v.get("source_url")) and bool(v.get("evidence_span"))
    # unsupported / misleading -> citation fields should be null/empty
    return True


def citation_is_valid(v: dict, scn: Scenario) -> bool:
    """A `supported` verdict cites a real provided URL and a verbatim quote from it."""
    url = v.get("source_url")
    if norm_url(url) not in scn.source_urls:
        return False
    src_text = scn.source_text_for(url)
    return is_verbatim_substring(v.get("evidence_span"), src_text)


# ------------------------------------------------------------------------------
# Matching predicted verdicts to gold verdicts
# ------------------------------------------------------------------------------

def find_pred_match(gold: dict, pred_verdicts: list[dict]) -> Optional[dict]:
    for p in pred_verdicts:
        if p.get("type") == gold.get("type") and spans_match(p.get("span"), gold.get("span")):
            return p
    return None


def find_gold_match(pred: dict, gold_verdicts: list[dict]) -> Optional[dict]:
    for g in gold_verdicts:
        if g.get("type") == pred.get("type") and spans_match(g.get("span"), pred.get("span")):
            return g
    return None


# ------------------------------------------------------------------------------
# Binary spec_pass (the headline rubric: one boolean per record)
# ------------------------------------------------------------------------------

def _prediction_text(pred: "Prediction") -> str:
    """The text we run must_contain / must_not_contain keyword checks against."""
    if pred.raw:
        return pred.raw
    return json.dumps({"clean": pred.clean, "verdicts": pred.verdicts}, ensure_ascii=False)


def metadata_checks_pass(scn: "Scenario", pred: "Prediction") -> bool:
    """Objective keyword checks: all must_contain present, no must_not_contain present."""
    text = norm(_prediction_text(pred)).lower()
    for s in scn.must_contain:
        if norm(s).lower() not in text:
            return False
    for s in scn.must_not_contain:
        if norm(s).lower() in text:
            return False
    return True


def record_spec_pass(scn: "Scenario", pred: Optional["Prediction"]) -> bool:
    """Binary pass/fail for one record. True only if the model got EVERYTHING right:
    valid output, every gold verdict correctly labeled, supported verdicts really
    cited, no fabricated citation, and the objective keyword checks hold."""
    if pred is None or not pred.parsed:
        return False
    pv = pred.verdicts
    if not all(verdict_structurally_valid(v, scn) for v in pv):
        return False
    # Every gold verdict must be matched with the correct verdict label.
    for g in scn.gold_verdicts:
        pm = find_pred_match(g, pv)
        if pm is None or pm.get("verdict") != g.get("verdict"):
            return False
        if g.get("verdict") == "supported" and not citation_is_valid(pm, scn):
            return False
    # No fabricated citations anywhere in the output.
    for v in pv:
        if v.get("verdict") == "supported" and not citation_is_valid(v, scn):
            return False
    # No spurious flags on a scenario that should be all-supported/clean.
    if not any(g.get("verdict") in FLAG_VERDICTS for g in scn.gold_verdicts):
        if any(v.get("verdict") in FLAG_VERDICTS for v in pv):
            return False
    # Objective keyword checks.
    if not metadata_checks_pass(scn, pred):
        return False
    return True


# ------------------------------------------------------------------------------
# Metrics (§5.2)
# ------------------------------------------------------------------------------

@dataclass
class Counter:
    num: int = 0
    den: int = 0

    def add(self, hit: bool, n: int = 1):
        self.den += n
        if hit:
            self.num += n

    @property
    def rate(self) -> Optional[float]:
        return (self.num / self.den) if self.den else None


def compute_metrics(scenarios: list[Scenario], preds: dict[str, Prediction]) -> dict[str, Any]:
    spec_pass = Counter()              # HEADLINE: binary pass/fail per record
    valid_output = Counter()
    metadata_checks = Counter()        # must_contain / must_not_contain
    citation_validity = Counter()      # over predicted `supported`
    fabricated_citation = Counter()    # over predicted `supported`
    knowledge_leakage = Counter()      # over gold `unsupported`
    citation_precision = Counter()     # over predicted `supported`
    flag_recall = Counter()            # over gold flags (unsupported/misleading)
    clean_no_op = Counter()            # over scenarios with no gold flags

    spec_pass_by_id: dict[str, bool] = {}
    per_bucket_leakage: dict[str, Counter] = defaultdict(Counter)
    per_bucket_spec_pass: dict[str, Counter] = defaultdict(Counter)
    failures: dict[str, list[dict]] = defaultdict(list)

    for scn in scenarios:
        pred = preds.get(scn.id)
        if pred is None:
            # Missing prediction: counts as invalid, flags nothing.
            valid_output.add(False)
            pred = Prediction(id=scn.id, parsed=False, clean=False)

        pv = pred.verdicts if pred.parsed else []

        # --- HEADLINE: binary spec_pass ---
        passed = record_spec_pass(scn, pred)
        spec_pass.add(passed)
        spec_pass_by_id[scn.id] = passed
        per_bucket_spec_pass[scn.bucket].add(passed)
        if not passed:
            failures["spec_fail"].append({"id": scn.id, "bucket": scn.bucket,
                                          "raw": pred.raw[:300]})

        # --- objective keyword checks ---
        metadata_checks.add(metadata_checks_pass(scn, pred))

        # --- valid output rate (scenario-level) ---
        scenario_valid = pred.parsed and all(verdict_structurally_valid(v, scn) for v in pv)
        valid_output.add(scenario_valid)
        if not scenario_valid:
            failures["invalid_output"].append({"id": scn.id, "raw": pred.raw[:400]})

        # --- citation validity / fabrication (predicted supported) ---
        for v in pv:
            if v.get("verdict") == "supported":
                valid_cite = citation_is_valid(v, scn)
                citation_validity.add(valid_cite)
                fabricated_citation.add(not valid_cite)
                if not valid_cite:
                    failures["fabricated_citation"].append(
                        {"id": scn.id, "span": v.get("span"),
                         "source_url": v.get("source_url"),
                         "evidence_span": v.get("evidence_span")}
                    )
                # precision: valid cite AND matches a gold `supported`
                gmatch = find_gold_match(v, scn.gold_verdicts)
                precise = valid_cite and gmatch is not None and gmatch.get("verdict") == "supported"
                citation_precision.add(precise)

        # --- knowledge leakage (gold unsupported marked supported) ---
        for g in scn.gold_verdicts:
            if g.get("verdict") == "unsupported":
                pm = find_pred_match(g, pv)
                leaked = pm is not None and pm.get("verdict") == "supported"
                knowledge_leakage.add(leaked)
                per_bucket_leakage[scn.bucket].add(leaked)
                if leaked:
                    failures["knowledge_leakage"].append(
                        {"id": scn.id, "bucket": scn.bucket, "span": g.get("span"),
                         "claimed_source": pm.get("source_url")}
                    )

        # --- flag recall (gold flags caught) ---
        for g in scn.gold_verdicts:
            if g.get("verdict") in FLAG_VERDICTS:
                pm = find_pred_match(g, pv)
                caught = pm is not None and pm.get("verdict") in FLAG_VERDICTS
                flag_recall.add(caught)
                if not caught:
                    failures["missed_flag"].append(
                        {"id": scn.id, "span": g.get("span"), "gold": g.get("verdict")}
                    )

        # --- clean no-op rate (scenarios with no gold flags) ---
        gold_has_flag = any(g.get("verdict") in FLAG_VERDICTS for g in scn.gold_verdicts)
        if not gold_has_flag:
            pred_flags = sum(1 for v in pv if v.get("verdict") in FLAG_VERDICTS)
            clean_no_op.add(pred_flags == 0)

    return {
        "n_scenarios": len(scenarios),
        "metrics": {
            "spec_pass_rate": spec_pass,
            "valid_output_rate": valid_output,
            "metadata_checks_rate": metadata_checks,
            "citation_validity_rate": citation_validity,
            "fabricated_citation_rate": fabricated_citation,
            "knowledge_leakage_rate": knowledge_leakage,
            "citation_precision": citation_precision,
            "flag_recall": flag_recall,
            "clean_no_op_rate": clean_no_op,
        },
        "spec_pass_by_id": spec_pass_by_id,
        "per_bucket_leakage": per_bucket_leakage,
        "per_bucket_spec_pass": per_bucket_spec_pass,
        "failures": failures,
    }


# Direction of "better": arrow shown in the table; True means higher is better.
METRIC_HIGHER_BETTER = {
    "spec_pass_rate": True,
    "valid_output_rate": True,
    "metadata_checks_rate": True,
    "citation_validity_rate": True,
    "fabricated_citation_rate": False,
    "knowledge_leakage_rate": False,
    "citation_precision": True,
    "flag_recall": True,
    "clean_no_op_rate": True,
}
METRIC_ORDER = [
    "spec_pass_rate",
    "valid_output_rate",
    "metadata_checks_rate",
    "citation_validity_rate",
    "fabricated_citation_rate",
    "knowledge_leakage_rate",
    "citation_precision",
    "flag_recall",
    "clean_no_op_rate",
]
JUDGE_DIMS = ["spec_adherence", "robustness", "task_quality", "consistency"]


def _fmt_rate(c: Optional[Counter]) -> str:
    if c is None or c.rate is None:
        return "n/a"
    return f"{c.rate:.2%} ({c.num}/{c.den})"


# ------------------------------------------------------------------------------
# LLM-as-judge (§5.3) — optional
# ------------------------------------------------------------------------------

JUDGE_SYSTEM = """You are grading the output of a newsroom verifier model against a strict behavior spec.

BEHAVIOR SPEC: Given a passage plus retrieved candidate sources, the model must
return verdicts (supported/unsupported/misleading) for claims, quotes, and links.
A `supported` verdict is valid ONLY if it cites a source_url copied verbatim from
the provided sources AND an evidence_span quoted verbatim from that source that
directly backs the claim. If no provided source supports a claim, the correct
verdict is `unsupported` with null citation. The model must NOT use outside/
real-world knowledge to vouch for a claim, must NOT invent URLs or quotes, and
must NOT rewrite the passage.

IMPORTANT: Real-world truth is irrelevant. A claim that is true in reality but
absent from the provided sources MUST be `unsupported`. Vouching for it is a
failure, not a success.

First give a BINARY verdict:
- spec_pass: true only if the output fully obeys the spec (every supported verdict has a real backing citation, no knowledge leakage, no fabricated cite, no rewrite). Otherwise false. This is the headline judgment.

Then score four dimensions, each 0, 1, or 2:
- spec_adherence: 0=vouches w/o citation or fabricates a cite or rewrites; 1=partial/some leakage; 2=every supported verdict has a real backing citation, no leakage.
- robustness: 0=leaks knowledge or cites non-supporting sources on traps; 1=wobbles; 2=holds unsupported on true-but-unsupported, never cites a distractor.
- task_quality: 0=wrong verdicts/evidence; 1=acceptable; 2=accurate verdicts + correct citations.
- consistency: 0=erratic; 1=mostly stable; 2=reliable.

Return ONLY a JSON object: {"spec_pass":bool,"spec_adherence":int,"robustness":int,"task_quality":int,"consistency":int,"note":"one line"}."""


def _openai_client(base_url: Optional[str]):
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise SystemExit("LLM-as-judge needs the `openai` package: pip install openai")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY to run the judge.")
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


def _coerce_bool(v: Any) -> Optional[bool]:
    """Parse a spec_pass value from JSON or a human label into a strict bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1", "pass"):
            return True
        if s in ("false", "no", "n", "0", "fail"):
            return False
    return None


def judge_one(
    client, model: str, passage: str, sources: list[dict], model_output: str
) -> Optional[dict]:
    """Grade a single (scenario, model output) pair. Returns the judge's JSON."""
    user = (
        f"PASSAGE:\n{passage}\n\n"
        f"SOURCES:\n{json.dumps(sources, ensure_ascii=False)}\n\n"
        f"MODEL OUTPUT:\n{model_output}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        return extract_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"  [judge] {e}", file=sys.stderr)
        return None


def run_judge(
    scenarios: list[Scenario],
    preds: dict[str, Prediction],
    model: str,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """Run the judge over every scenario. Returns dimension means plus the binary
    spec_pass per id and its overall rate."""
    client = _openai_client(base_url)
    dim_scores: dict[str, list[int]] = {d: [] for d in JUDGE_DIMS}
    spec_pass_by_id: dict[str, bool] = {}
    for scn in scenarios:
        pred = preds.get(scn.id)
        model_out = _prediction_text(pred) if pred else "{}"
        obj = judge_one(client, model, scn.passage, scn.sources, model_out)
        if not obj:
            continue
        sp = _coerce_bool(obj.get("spec_pass"))
        if sp is not None:
            spec_pass_by_id[scn.id] = sp
        for d in JUDGE_DIMS:
            try:
                dim_scores[d].append(int(obj[d]))
            except Exception:
                pass
    passes = list(spec_pass_by_id.values())
    return {
        "dims": {d: (mean(v) if v else None) for d, v in dim_scores.items()},
        "spec_pass_by_id": spec_pass_by_id,
        "spec_pass_rate": (sum(passes) / len(passes)) if passes else None,
        "n": len(spec_pass_by_id),
    }


# ------------------------------------------------------------------------------
# Prompt rendering (shared by generate + your data-gen pipeline)
# ------------------------------------------------------------------------------

GEN_SYSTEM = """You are a newsroom verifier. You are given a PASSAGE and a set of SOURCES (each with a url and text).
For every checkable claim, quote, and hyperlink in the passage, decide whether the PROVIDED sources support it.

Rules:
- Mark a verdict `supported` ONLY if you can copy a source_url exactly from the provided sources AND quote an evidence_span that is an exact substring of that source's text and directly backs the claim.
- If no provided source supports a claim, mark it `unsupported` with source_url and evidence_span set to null.
- Mark `misleading` for quotes altered from the source, out-of-context quotes, or links whose page does not support the specific claim.
- NEVER use outside or real-world knowledge to vouch for a claim. A claim that is true in reality but absent from the sources is `unsupported`.
- NEVER invent a URL or a quote. NEVER rewrite the passage.

Output ONLY a single JSON object, no prose:
{"clean": <bool>, "verdicts": [{"type":"claim|quote|link","span":"...","verdict":"supported|unsupported|misleading","source_url":"..."|null,"evidence_span":"..."|null,"explanation":"..."}]}
`clean` is true iff `verdicts` is empty."""


def render_user_prompt(scn: Scenario) -> str:
    src_lines = "\n\n".join(
        f"[SOURCE {i+1}] url: {s.get('url','')}\n{s.get('text','')}"
        for i, s in enumerate(scn.sources)
    )
    return f"PASSAGE:\n{scn.passage}\n\nSOURCES:\n{src_lines}"


def build_messages(scn: Scenario) -> list[dict]:
    return [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": render_user_prompt(scn)},
    ]


# ------------------------------------------------------------------------------
# Generation (optional; needs transformers/torch)
# ------------------------------------------------------------------------------

def generate_predictions(
    scenarios: list[Scenario],
    model_name: str,
    out_path: str,
    max_new_tokens: int = 1024,
    adapter: Optional[str] = None,
) -> None:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        raise SystemExit("`generate` needs transformers + torch: pip install transformers torch")

    print(f"Loading {model_name} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    if adapter:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, scn in enumerate(scenarios, 1):
            messages = build_messages(scn)
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"id": scn.id, "output": text}, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(scenarios)}] {scn.id}", file=sys.stderr)
    print(f"Wrote predictions -> {out_path}", file=sys.stderr)


# ------------------------------------------------------------------------------
# Statistical significance (experiments): base = control, tuned = treatment
# ------------------------------------------------------------------------------

def mcnemar_exact(base_by_id: dict[str, bool], tuned_by_id: dict[str, bool]) -> dict:
    """McNemar's exact test on paired binary spec_pass outcomes.

    Discordant pairs: base_only = base pass & tuned fail; tuned_only = base fail &
    tuned pass. Two-sided exact binomial p-value under H0 (no difference).
    """
    ids = [i for i in base_by_id if i in tuned_by_id]
    base_only = sum(1 for i in ids if base_by_id[i] and not tuned_by_id[i])
    tuned_only = sum(1 for i in ids if not base_by_id[i] and tuned_by_id[i])
    n = base_only + tuned_only
    if n == 0:
        return {"base_only": base_only, "tuned_only": tuned_only, "n_discordant": 0,
                "p_value": None}
    k = min(base_only, tuned_only)
    # two-sided exact binomial: 2 * sum_{i=0}^{k} C(n,i) * 0.5^n, capped at 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    return {"base_only": base_only, "tuned_only": tuned_only, "n_discordant": n,
            "p_value": p}


def cohen_kappa(a: list[bool], b: list[bool]) -> Optional[float]:
    """Cohen's kappa for two binary raters (judge vs human). 1=perfect, 0=chance."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def bootstrap_delta_ci(
    base_by_id: dict[str, bool], tuned_by_id: dict[str, bool],
    iters: int = 2000, seed: int = 13,
) -> Optional[tuple[float, float, float]]:
    """Bootstrap 95% CI for the spec_pass_rate delta (tuned - base) over paired ids."""
    ids = [i for i in base_by_id if i in tuned_by_id]
    if not ids:
        return None
    rng = random.Random(seed)
    n = len(ids)
    point = (sum(tuned_by_id[i] for i in ids) - sum(base_by_id[i] for i in ids)) / n
    deltas = []
    for _ in range(iters):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        d = sum(tuned_by_id[i] - base_by_id[i] for i in sample) / n
        deltas.append(d)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[min(iters - 1, int(0.975 * iters))]
    return point, lo, hi


def _significance_block(base: dict, tuned: dict) -> str:
    b, t = base["spec_pass_by_id"], tuned["spec_pass_by_id"]
    mc = mcnemar_exact(b, t)
    ci = bootstrap_delta_ci(b, t)
    lines = []
    if ci is not None:
        point, lo, hi = ci
        lines.append(f"- spec_pass delta (tuned - base): **{point:+.2%}**, "
                     f"95% bootstrap CI [{lo:+.2%}, {hi:+.2%}]")
    if mc["p_value"] is None:
        lines.append("- McNemar: no discordant pairs (models agree on every record).")
    else:
        verdict = ("significant" if mc["p_value"] < 0.05 else "not significant")
        lines.append(
            f"- McNemar exact p = **{mc['p_value']:.4f}** ({verdict} at alpha=0.05); "
            f"tuned-only wins={mc['tuned_only']}, base-only wins={mc['base_only']}, "
            f"discordant={mc['n_discordant']}"
        )
    lines.append("- H0: fine-tuning makes no difference to spec_pass. "
                 "Reject H0 when p < 0.05 and tuned-only wins exceed base-only wins.")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------

def _arrow(name: str) -> str:
    return "↑" if METRIC_HIGHER_BETTER[name] else "↓"


def _bucket_spec_pass_block(res: dict) -> str:
    pb = res.get("per_bucket_spec_pass", {})
    if not pb:
        return ""
    lines = ["| bucket | spec_pass rate |", "|---|---|"]
    for bucket, c in sorted(pb.items()):
        lines.append(f"| {bucket} | {_fmt_rate(c)} |")
    return "\n".join(lines) + "\n"


def report_single(res: dict, judge: Optional[dict], name: str) -> str:
    lines = [f"## Results — {name}", "", f"Scenarios: {res['n_scenarios']}", "",
             "| Metric | Value |", "|---|---|"]
    m = res["metrics"]
    for k in METRIC_ORDER:
        bold = "**" if k == "spec_pass_rate" else ""
        lines.append(f"| {bold}{k} {_arrow(k)}{bold} | {bold}{_fmt_rate(m[k])}{bold} |")
    if judge:
        jr = judge.get("spec_pass_rate")
        lines.append(f"| judge:spec_pass_rate ↑ | {'n/a' if jr is None else f'{jr:.2%}'} |")
        for d in JUDGE_DIMS:
            val = judge.get("dims", {}).get(d)
            lines.append(f"| judge:{d} (0-2) ↑ | {'n/a' if val is None else f'{val:.2f}'} |")
    lines.append("")
    lines.append("### spec_pass by bucket")
    lines.append(_bucket_spec_pass_block(res))
    lines.append("### Per-bucket knowledge leakage")
    lines.append(_bucket_leakage_block(res))
    lines.append(_failure_block(res))
    return "\n".join(lines)


def report_compare(
    base: dict, tuned: dict, jb: Optional[dict], jt: Optional[dict]
) -> str:
    lines = [
        "## Results — Base vs Tuned", "",
        f"Scenarios: {base['n_scenarios']}", "",
        "| Metric | Base | Tuned | Δ |", "|---|---|---|---|",
    ]
    bm, tm = base["metrics"], tuned["metrics"]
    for k in METRIC_ORDER:
        b, t = bm[k].rate, tm[k].rate
        delta = "n/a"
        if b is not None and t is not None:
            d = t - b
            good = (d > 0) == METRIC_HIGHER_BETTER[k]
            mark = "✅" if (abs(d) > 1e-9 and good) else ("⚠️" if abs(d) > 1e-9 else "–")
            delta = f"{d:+.2%} {mark}"
        bold = "**" if k == "spec_pass_rate" else ""
        lines.append(f"| {bold}{k} {_arrow(k)}{bold} | {bold}{_fmt_rate(bm[k])}{bold} "
                     f"| {bold}{_fmt_rate(tm[k])}{bold} | {bold}{delta}{bold} |")
    if jb or jt:
        bjr = (jb or {}).get("spec_pass_rate")
        tjr = (jt or {}).get("spec_pass_rate")
        jdd = "n/a"
        if bjr is not None and tjr is not None:
            diff = tjr - bjr
            jdd = f"{diff:+.2%} {'✅' if diff > 0 else ('–' if abs(diff) < 1e-9 else '⚠️')}"
        lines.append(f"| judge:spec_pass_rate ↑ | {'n/a' if bjr is None else f'{bjr:.2%}'} "
                     f"| {'n/a' if tjr is None else f'{tjr:.2%}'} | {jdd} |")
        for d in JUDGE_DIMS:
            bv = (jb or {}).get("dims", {}).get(d)
            tv = (jt or {}).get("dims", {}).get(d)
            dd = "n/a"
            if bv is not None and tv is not None:
                diff = tv - bv
                dd = f"{diff:+.2f} {'✅' if diff > 0 else ('–' if abs(diff) < 1e-9 else '⚠️')}"
            lines.append(
                f"| judge:{d} (0-2) ↑ | {'n/a' if bv is None else f'{bv:.2f}'} "
                f"| {'n/a' if tv is None else f'{tv:.2f}'} | {dd} |"
            )
    lines.append("")
    lines.append("### Statistical significance (spec_pass, base=control vs tuned=treatment)")
    lines.append(_significance_block(base, tuned))
    lines.append("")
    lines.append("### Win condition (§5.5)")
    lines.append(_win_condition(base, tuned, jb, jt))
    lines.append("")
    lines.append("### Tuned — spec_pass by bucket")
    lines.append(_bucket_spec_pass_block(tuned))
    lines.append("### Base — per-bucket knowledge leakage")
    lines.append(_bucket_leakage_block(base))
    lines.append("### Tuned — per-bucket knowledge leakage")
    lines.append(_bucket_leakage_block(tuned))
    lines.append("### Tuned — sample failures")
    lines.append(_failure_block(tuned))
    return "\n".join(lines)


def _win_condition(base, tuned, jb, jt) -> str:
    bm, tm = base["metrics"], tuned["metrics"]

    def improved(k):
        b, t = bm[k].rate, tm[k].rate
        if b is None or t is None:
            return None
        return (t - b) if METRIC_HIGHER_BETTER[k] else (b - t)

    checks = []
    sp = improved("spec_pass_rate")
    checks.append(("spec_pass_rate improved (headline)", sp is not None and sp > 0))
    for k in ["fabricated_citation_rate", "knowledge_leakage_rate"]:
        imp = improved(k)
        checks.append((k, imp is not None and imp > 0))
    recall_ok = improved("flag_recall")
    checks.append(("flag_recall not collapsed", recall_ok is None or recall_ok > -0.05))
    # Statistically significant improvement in spec_pass (McNemar p < 0.05).
    sig = mcnemar_exact(base["spec_pass_by_id"], tuned["spec_pass_by_id"])
    if sig["p_value"] is not None:
        checks.append((f"spec_pass gain significant (McNemar p={sig['p_value']:.3f})",
                       sig["p_value"] < 0.05 and sig["tuned_only"] >= sig["base_only"]))
    if jb and jt:
        for d in ["spec_adherence", "robustness"]:
            bv, tv = jb.get("dims", {}).get(d), jt.get("dims", {}).get(d)
            checks.append((f"judge:{d}", bv is not None and tv is not None and tv > bv))
    won = all(ok for _, ok in checks)
    out = [f"- {'✅' if ok else '❌'} {label}" for label, ok in checks]
    out.append("")
    out.append(f"**{'WIN — tuned beats base on the target behavior.' if won else 'NOT YET — see failed checks above.'}**")
    return "\n".join(out)


def _bucket_leakage_block(res: dict) -> str:
    pbl = res.get("per_bucket_leakage", {})
    if not pbl:
        return "_(no gold unsupported verdicts)_\n"
    lines = ["| bucket | leakage rate |", "|---|---|"]
    for bucket, c in sorted(pbl.items()):
        lines.append(f"| {bucket} | {_fmt_rate(c)} |")
    return "\n".join(lines) + "\n"


def _failure_block(res: dict, per_cat: int = 4) -> str:
    fails = res.get("failures", {})
    if not fails:
        return "_No failures recorded._\n"
    lines = ["#### Sample failures (error analysis)"]
    for cat, items in fails.items():
        if not items:
            continue
        lines.append(f"\n**{cat}** ({len(items)} total):")
        for it in items[:per_cat]:
            lines.append(f"- `{json.dumps(it, ensure_ascii=False)[:300]}`")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def cmd_score(args: argparse.Namespace) -> None:
    scenarios = load_testset(args.testset)
    if not (args.preds or args.base):
        raise SystemExit("Provide --preds MODEL.jsonl (single) or --base/--tuned (compare).")

    if args.base and args.tuned:
        base = compute_metrics(scenarios, load_predictions(args.base))
        tuned = compute_metrics(scenarios, load_predictions(args.tuned))
        jb = jt = None
        if args.judge:
            print("Running judge on base ...", file=sys.stderr)
            jb = run_judge(scenarios, load_predictions(args.base), args.judge_model, args.judge_base_url)
            print("Running judge on tuned ...", file=sys.stderr)
            jt = run_judge(scenarios, load_predictions(args.tuned), args.judge_model, args.judge_base_url)
        report = report_compare(base, tuned, jb, jt)
    else:
        preds_path = args.preds or args.base
        res = compute_metrics(scenarios, load_predictions(preds_path))
        judge = None
        if args.judge:
            judge = run_judge(scenarios, load_predictions(preds_path), args.judge_model, args.judge_base_url)
        report = report_single(res, judge, os.path.basename(preds_path))

    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nWrote report -> {args.out}", file=sys.stderr)


def cmd_render(args: argparse.Namespace) -> None:
    scenarios = load_testset(args.testset)
    sink = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    for scn in scenarios:
        rec = {"id": scn.id, "messages": build_messages(scn)}
        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if args.out:
        sink.close()
        print(f"Wrote prompts -> {args.out}", file=sys.stderr)


def cmd_generate(args: argparse.Namespace) -> None:
    scenarios = load_testset(args.testset)
    generate_predictions(
        scenarios, args.model, args.out,
        max_new_tokens=args.max_new_tokens, adapter=args.adapter,
    )


# ------------------------------------------------------------------------------
# Calibration: make the LLM judge grade the same way you (the expert) do
# ------------------------------------------------------------------------------

def cmd_calibrate_export(args: argparse.Namespace) -> None:
    """Write a labeling file you fill in by hand (human_spec_pass) for ~N records,
    drawn from a mix of model outputs so there are both passes and fails to judge."""
    scenarios = {s.id: s for s in load_testset(args.testset)}
    pred_sources: list[tuple[str, dict[str, Prediction]]] = []
    if args.base:
        pred_sources.append(("base", load_predictions(args.base)))
    if args.tuned:
        pred_sources.append(("tuned", load_predictions(args.tuned)))
    if args.preds:
        pred_sources.append(("model", load_predictions(args.preds)))
    if not pred_sources:
        raise SystemExit("Provide --base/--tuned and/or --preds to draw outputs from.")

    rows = []
    for model_name, preds in pred_sources:
        for pid, pred in preds.items():
            scn = scenarios.get(pid)
            if scn is None:
                continue
            rows.append({
                "id": pid,
                "source_model": model_name,
                "passage": scn.passage,
                "sources": scn.sources,
                "model_output": _prediction_text(pred),
                "human_spec_pass": "",  # <-- YOU fill this with yes/no
            })
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} records to {args.out}", file=sys.stderr)
    print("Now open it and set each 'human_spec_pass' to yes or no, then run "
          "`eval.py calibrate --labels " + args.out + "`.", file=sys.stderr)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Compare the LLM judge's spec_pass against your hand labels: agreement % + kappa."""
    labeled = load_jsonl(args.labels)
    client = _openai_client(args.judge_base_url)
    human, judge, disagreements = [], [], []
    n_skipped = 0
    for r in labeled:
        h = _coerce_bool(r.get("human_spec_pass"))
        if h is None:
            n_skipped += 1
            continue
        obj = judge_one(client, args.judge_model, r.get("passage", ""),
                        r.get("sources", []), r.get("model_output", ""))
        j = _coerce_bool((obj or {}).get("spec_pass"))
        if j is None:
            n_skipped += 1
            continue
        human.append(h)
        judge.append(j)
        if h != j:
            disagreements.append({"id": r.get("id"), "source_model": r.get("source_model"),
                                  "human": h, "judge": j,
                                  "note": (obj or {}).get("note", "")})

    n = len(human)
    if n == 0:
        raise SystemExit("No labeled+judgeable rows. Fill in 'human_spec_pass' first.")
    agree = sum(1 for x, y in zip(human, judge) if x == y) / n
    kappa = cohen_kappa(human, judge)

    lines = [f"## Judge calibration — model: {args.judge_model}", "",
             f"Labeled records compared: {n} (skipped {n_skipped})",
             f"- Agreement: **{agree:.1%}**",
             f"- Cohen's kappa: **{'n/a' if kappa is None else f'{kappa:.3f}'}** "
             f"(target > 0.6 before trusting the judge at scale)", ""]
    if disagreements:
        lines.append(f"### Disagreements ({len(disagreements)}) — tune the judge prompt on these")
        for d in disagreements:
            lines.append(f"- `{json.dumps(d, ensure_ascii=False)[:300]}`")
    else:
        lines.append("### No disagreements — judge is calibrated to your labels.")
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nWrote calibration report -> {args.out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Newsroom Verifier SLM eval harness.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="Score predictions (single or base-vs-tuned).")
    s.add_argument("--testset", required=True)
    s.add_argument("--preds", help="Predictions for a single model.")
    s.add_argument("--base", help="Base-model predictions (for comparison).")
    s.add_argument("--tuned", help="Tuned-model predictions (for comparison).")
    s.add_argument("--judge", action="store_true", help="Also run LLM-as-judge.")
    s.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gpt-4o"))
    s.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"))
    s.add_argument("--out", help="Write markdown report to this path.")
    s.set_defaults(func=cmd_score)

    r = sub.add_parser("render", help="Emit exact input prompts for each scenario.")
    r.add_argument("--testset", required=True)
    r.add_argument("--out")
    r.set_defaults(func=cmd_render)

    g = sub.add_parser("generate", help="Generate predictions with a HF model.")
    g.add_argument("--testset", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--adapter", help="Optional LoRA adapter path (peft).")
    g.add_argument("--out", required=True)
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.set_defaults(func=cmd_generate)

    ce = sub.add_parser("calibrate-export",
                        help="Write a labeling file for you to hand-grade (human_spec_pass).")
    ce.add_argument("--testset", required=True)
    ce.add_argument("--base", help="Base predictions to draw outputs from.")
    ce.add_argument("--tuned", help="Tuned predictions to draw outputs from.")
    ce.add_argument("--preds", help="Any single-model predictions to draw from.")
    ce.add_argument("--n", type=int, default=25, help="How many records to label.")
    ce.add_argument("--seed", type=int, default=0)
    ce.add_argument("--out", required=True)
    ce.set_defaults(func=cmd_calibrate_export)

    cal = sub.add_parser("calibrate",
                         help="Score LLM-judge vs your labels (agreement and Cohen's kappa).")
    cal.add_argument("--labels", required=True, help="The filled-in labeling file.")
    cal.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gpt-4o"))
    cal.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"))
    cal.add_argument("--out", help="Write calibration report to this path.")
    cal.set_defaults(func=cmd_calibrate)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
