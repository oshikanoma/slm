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
"""
from __future__ import annotations

import argparse
import json
import os
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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: invalid JSON line: {e}")
    return rows


def load_testset(path: str) -> list[Scenario]:
    out = []
    for r in load_jsonl(path):
        out.append(
            Scenario(
                id=str(r["id"]),
                bucket=r.get("bucket", "unspecified"),
                passage=r.get("passage", ""),
                sources=r.get("sources", []) or [],
                gold_verdicts=r.get("gold_verdicts", []) or [],
            )
        )
    return out


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
    valid_output = Counter()
    citation_validity = Counter()      # over predicted `supported`
    fabricated_citation = Counter()    # over predicted `supported`
    knowledge_leakage = Counter()      # over gold `unsupported`
    citation_precision = Counter()     # over predicted `supported`
    flag_recall = Counter()            # over gold flags (unsupported/misleading)
    clean_no_op = Counter()            # over scenarios with no gold flags

    per_bucket_leakage: dict[str, Counter] = defaultdict(Counter)
    failures: dict[str, list[dict]] = defaultdict(list)

    for scn in scenarios:
        pred = preds.get(scn.id)
        if pred is None:
            # Missing prediction: counts as invalid, flags nothing.
            valid_output.add(False)
            pred = Prediction(id=scn.id, parsed=False, clean=False)

        pv = pred.verdicts if pred.parsed else []

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
            "valid_output_rate": valid_output,
            "citation_validity_rate": citation_validity,
            "fabricated_citation_rate": fabricated_citation,
            "knowledge_leakage_rate": knowledge_leakage,
            "citation_precision": citation_precision,
            "flag_recall": flag_recall,
            "clean_no_op_rate": clean_no_op,
        },
        "per_bucket_leakage": per_bucket_leakage,
        "failures": failures,
    }


# Direction of "better": arrow shown in the table; True means higher is better.
METRIC_HIGHER_BETTER = {
    "valid_output_rate": True,
    "citation_validity_rate": True,
    "fabricated_citation_rate": False,
    "knowledge_leakage_rate": False,
    "citation_precision": True,
    "flag_recall": True,
    "clean_no_op_rate": True,
}
METRIC_ORDER = [
    "valid_output_rate",
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

Score the model output on four dimensions, each 0, 1, or 2:
- spec_adherence: 0=vouches w/o citation or fabricates a cite or rewrites; 1=partial/some leakage; 2=every supported verdict has a real backing citation, no leakage.
- robustness: 0=leaks knowledge or cites non-supporting sources on traps; 1=wobbles; 2=holds unsupported on true-but-unsupported, never cites a distractor.
- task_quality: 0=wrong verdicts/evidence; 1=acceptable; 2=accurate verdicts + correct citations.
- consistency: 0=erratic; 1=mostly stable; 2=reliable.

Return ONLY a JSON object: {"spec_adherence":int,"robustness":int,"task_quality":int,"consistency":int,"note":"one line"}."""


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
    return OpenAI(**kwargs)


def run_judge(
    scenarios: list[Scenario],
    preds: dict[str, Prediction],
    model: str,
    base_url: Optional[str] = None,
) -> dict[str, Optional[float]]:
    client = _openai_client(base_url)
    dim_scores: dict[str, list[int]] = {d: [] for d in JUDGE_DIMS}
    for scn in scenarios:
        pred = preds.get(scn.id)
        model_out = pred.raw if (pred and pred.raw) else json.dumps(
            {"clean": pred.clean, "verdicts": pred.verdicts} if pred else {}
        )
        user = (
            f"PASSAGE:\n{scn.passage}\n\n"
            f"SOURCES:\n{json.dumps(scn.sources, ensure_ascii=False)}\n\n"
            f"MODEL OUTPUT:\n{model_out}\n"
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
            obj = extract_json(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"  [judge] {scn.id}: {e}", file=sys.stderr)
            obj = None
        if not obj:
            continue
        for d in JUDGE_DIMS:
            try:
                dim_scores[d].append(int(obj[d]))
            except Exception:
                pass
    return {d: (mean(v) if v else None) for d, v in dim_scores.items()}


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
# Reporting
# ------------------------------------------------------------------------------

def _arrow(name: str) -> str:
    return "↑" if METRIC_HIGHER_BETTER[name] else "↓"


def report_single(res: dict, judge: Optional[dict], name: str) -> str:
    lines = [f"## Results — {name}", "", f"Scenarios: {res['n_scenarios']}", "",
             "| Metric | Value |", "|---|---|"]
    m = res["metrics"]
    for k in METRIC_ORDER:
        lines.append(f"| {k} {_arrow(k)} | {_fmt_rate(m[k])} |")
    if judge:
        for d in JUDGE_DIMS:
            val = judge.get(d)
            lines.append(f"| judge:{d} (0-2) ↑ | {'n/a' if val is None else f'{val:.2f}'} |")
    lines.append("")
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
        lines.append(f"| {k} {_arrow(k)} | {_fmt_rate(bm[k])} | {_fmt_rate(tm[k])} | {delta} |")
    if jb or jt:
        for d in JUDGE_DIMS:
            bv = (jb or {}).get(d)
            tv = (jt or {}).get(d)
            dd = "n/a"
            if bv is not None and tv is not None:
                diff = tv - bv
                dd = f"{diff:+.2f} {'✅' if diff > 0 else ('–' if abs(diff) < 1e-9 else '⚠️')}"
            lines.append(
                f"| judge:{d} (0-2) ↑ | {'n/a' if bv is None else f'{bv:.2f}'} "
                f"| {'n/a' if tv is None else f'{tv:.2f}'} | {dd} |"
            )
    lines.append("")
    lines.append("### Win condition (§5.5)")
    lines.append(_win_condition(base, tuned, jb, jt))
    lines.append("")
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
    for k in ["fabricated_citation_rate", "knowledge_leakage_rate"]:
        imp = improved(k)
        checks.append((k, imp is not None and imp > 0))
    recall_ok = improved("flag_recall")
    checks.append(("flag_recall not collapsed", recall_ok is None or recall_ok > -0.05))
    if jb and jt:
        for d in ["spec_adherence", "robustness"]:
            bv, tv = jb.get(d), jt.get(d)
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
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
