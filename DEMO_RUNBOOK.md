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

**AP-only (toggle OFF, faster):** use a PLAIN STYLE sentence, ONE violation:
  - `The meeting starts at 3:00 PM.`   → ⚑ time → "3 p.m." ✅
  - `Enrollment rose 12 percent this year.`   → ⚑ percent → "12%" ✅
  - `The vote was held on December 25.`   → ⚑ months → "Dec. 25" ✅

  ⚠️ TWO honest limits to demo AROUND (not into):
  (a) **one AP flag per sentence** (training had one violation each);
  (b) on a sentence that is ALSO a factual news claim (e.g. the JWST one), the model
      prioritizes *claim verification* over AP — so it may mark it unsupported instead
      of flagging AP. Keep AP examples to plain style sentences.

**Verify (toggle ON):** paste a claim → real sources → cited verdicts.
  - `The James Webb Space Telescope launched on December 25, 2021 from French Guiana.`

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
