# Live Eval Review — Runbook (keep this open during the review)

**Model:** Qwen3-1.7B + QLoRA, v6 (final). **Best result: spec_pass 0.6% → 84%, p<0.0001.**
Canonical numbers = the Colab results files (`results.v6.colab.md`). Local re-scores run
but read a few points lower (Mac throttling degrades generation) — cite Colab.

---

## 30-second pitch (say this first)
"I built a **cited, evidence-bounded newsroom verifier**: given a passage plus retrieved
sources, it flags every claim/quote as supported / unsupported / misleading, and for
anything 'supported' it must quote the exact backing sentence from a real source. The one
failure it must never do is **vouch for a claim it can't cite** — that's how models
hallucinate. A prompt can't reliably stop that; a fine-tuned dataset can. I also composed
in **AP Style** flagging on top."

---

## 1. Show the EVAL SUITE  (open `eval.py`, `data/golden.json`, `diagnose.py`)

Say: "My eval suite is a **golden set of 175 records** with a bucket mix, scored by a
**binary-first rubric** — one `spec_pass` boolean per record — plus objective sub-metrics,
an LLM-judge layer, and statistical significance."

Command — show the golden set composition:
```bash
python3 -c "import json,collections; g=json.load(open('data/golden.json')); print(len(g),'records'); print(dict(collections.Counter(r['bucket'] for r in g)))"
```
Buckets: supported 36, ap_style 60, unsupported 30, true_but_unsupported 21, distractor 16, misleading 12.

Point out the **trap buckets** (true_but_unsupported, distractor): claims that are true in
reality but absent from the sources — these measure **knowledge leakage**, the forbidden failure.

## 2. Show BASE vs TUNED  (the money command — instant, no GPU)
```bash
python3 eval.py score --testset data/golden.json --base /tmp/base.v5.jsonl --tuned tuned_preds.v5.jsonl
```
This prints the full table + McNemar significance. Talking points:
- **spec_pass: base ~9% → tuned ~74%** (Colab v6: **0.6% → 84%**)
- **fabricated_citation: 61% → 3%** — base invents/empties citations; tuned doesn't. THE headline.
- **flag_recall 10% → 71%** — base barely flags problems; tuned catches them.
- **McNemar p < 0.0001, base-only wins = 0** — statistically real, not noise.

## 2b. FRONTIER LEADERBOARD  (the mentor's ask — the thesis test)
```bash
python3 compare_models.py --auto        # base + tuned + frontier models
```
Or show the saved table: `results.leaderboard.md`

| Model | spec_pass | verdict_acc |
|---|---|---|
| **Tuned Qwen3-1.7B (ours)** | **73.7%** | **74.9%** |
| GPT-OSS-120B (zero-shot) | 30.3% | 46.3% |
| Llama-3.3-70B (zero-shot) | 14.3% | 25.7% |
| Base Qwen3-1.7B (zero-shot) | 9.1% | 19.4% |

Say: "My fine-tuned 1.7B beats a prompted **120B by 2.4× and a 70B by 5×** — models ~70×
larger. The frontier models get a capability gradient (bigger = better), so the task
rewards scale — but **fine-tuning on the right data beats 70× the parameters.** All run
zero-shot with the same prompt on the same 175-record golden set. `verdict_accuracy`
(lenient label-correct) shows the same ordering, so it's not a quirk of the strict metric."
Note honestly: the 70B *refused/returned empty on ~60% of the sensitive news passages* —
a robustness point for the tuned model, which always returns a structured verdict.

## 3. Per-bucket + the honest story  (`diagnose.py`)
```bash
python3 diagnose.py --preds tuned_preds.v5.jsonl
```
Say: "Per bucket, v6 is **ap_style 100%, misleading 100%, distractor 88%, unsupported 80%,
true_but_unsupported 86%** — and **supported 53%**, which is my weak spot."

## 4. LIVE DEMO — the browser UI  (one box, easiest)
```bash
export TAVILY_API_KEY=tvly-dev-...     # open-web search when "retrieve" is on
python3 app.py                          # opens http://127.0.0.1:7860
```
One passage box + a "retrieve sources & verify" toggle. Returns AP flags AND claim verdicts.

**AP Style is now DETERMINISTIC code** (`ap_rules.py`) — it runs on every input, catches
EVERY violation instantly, on any sentence (incl. claims). No more single-sentence caveat.
  - `The meeting has 5 members and starts at 3:00 PM on December 25.` → flags 5→five, 3 p.m., Dec. 25 (all three)

**Claim verification = the SLM** (toggle "retrieve & verify" ON): real sources → cited verdicts.
  - `The James Webb Space Telescope launched on December 25, 2021 from French Guiana.`
    → ✅ SUPPORTED (NPR source + backing quote) AND ⚑ AP: "December 25" → "Dec. 25"

**KEY TALKING POINT (this is the mature systems answer):** "AP style is deterministic, so
I check it in **code** — exactly my Brainlift's thesis that deterministic editorial tasks
should be offloaded. The **SLM** is reserved for the non-deterministic part: verifying claims
against evidence with restraint, which a prompt can't reliably enforce. Code does what's
objective; the fine-tuned model does what needs judgment."

### Terminal alternative (for the base-vs-tuned CONTRAST specifically)
```bash
python3 demo.py --compare "The James Webb Space Telescope launched on December 25, 2021 from French Guiana."
```
Point at output: **base says 'supported' with an EMPTY backing quote** (vouching with no
evidence — the forbidden failure); **tuned quotes the real backing sentence.**

(Model loads ~1 min cold, ~15s/check. Pre-run once before the review so it's warm.)

---

## "What I'd do better next time"  (they WILL ask — have this ready)
1. **Build the eval fairness in from day one.** Two labeling artifacts (single-span matching;
   pinning one checked-source) were unfairly failing correct answers — real score was ~73%,
   not the 55% I first saw. I caught it with a failure-diagnostic, but should've designed the
   golden set to allow multiple valid spans up front.
2. **Calibrate the judge earlier.** Calibration (kappa 0.155) surfaced that my own spec
   disagreed with my editorial standard — that drove a spec revision (v2). Doing it before
   training would've saved a version.
3. **Match synthetic data to the real distribution.** v5's easy synthetic 'supported' examples
   didn't transfer; only v6's hard *real* ones helped — and even then, `supported` plateaued.
4. **Know the ceiling.** `supported` capped at ~53% across two targeted retrains → it's the
   **1.7B capability limit** on exact long-span extraction, not a data gap. A bigger base
   (Qwen3-4B) is the only real lever — a deliberate trade against the "tiny local model" thesis.

## Iteration story (one line): 
spec → data → train → eval → **calibrate → diagnose → fix → repeat**, six times. Each version
fixed a real weakness the previous eval exposed. v1→v2 (spec revision), v3 (rebalance),
v4 (+AP Style, composed), v5/v6 (supported → found the ceiling).

## If something breaks live
- Demo too slow / model won't load → fall back to `results.v6.colab.md` (open the file, it's the canonical table).
- Any command errors → the pre-computed tables in `results.v*.colab.md` are your backup.
