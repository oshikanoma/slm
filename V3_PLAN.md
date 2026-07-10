# v3 Plan — Thin Retriever + Evidence-Bounded SLM (spec §8 demo architecture)

**Decision:** keep the v2 SLM exactly as-is (hallucination-safe: only quotes provided
text). Add a THIN retrieval LAYER around it that finds/fetches candidate sources and
drops them into the bundle before the model verifies. The retriever does the
source-hunting; the model stays disciplined. This is what the user wanted
("the model should find sources") done without breaking the anti-hallucination thesis.

Pipeline:  claim/passage → retriever (search + fetch) → source bundle in context →
           existing v2 SLM emits cited verdicts over that bundle

## Why this is the right build (not a compromise)
- The SLM cannot search the web (it's a 1.7B LM); retrieval is a separate tool. So
  "source-hunting" is necessarily a layer, not a model change.
- Model unchanged ⇒ fabrication guarantee intact, eval stays objectively gradable,
  no retrain needed for the core. Most work is the retriever + a small demo.

## Components
1. `retriever.py` (NEW)
   - `search(query) -> [urls]`: pluggable backend. Free options: DuckDuckGo HTML
     endpoint or Wikipedia/Wikinews API (no key). Better: Brave/Serper/Tavily (key).
   - `fetch_and_extract(url) -> text`: REUSE `ingest_texastribune.extract_main_text`
     + `fetch` (already written, already used).
   - `build_bundle(passage) -> [{url, text}]`: derive query terms from the passage
     (key noun phrases / quoted spans), search, fetch top-K, add 1 distractor, cap
     text length. Output is exactly the `sources` shape eval/model already consume.
2. `demo.py` (NEW)
   - CLI/notebook: paste a passage → retriever builds bundle → run v2 SLM
     (predict_local.py inference path) → print cited verdicts with clickable URLs.
   - Shows the base model fabricating vs tuned model citing real retrieved URLs
     (spec §8 "money shot").
3. NO change to eval.py / the model / the dataset for the core. Optional: an
   end-to-end retrieval eval (does the retriever surface a supporting source when one
   exists on the web?) — separate, subjective, lower priority.

## Also fold in the v2 model bugs (cheap, high-value)
Calibration found the model OVER-FLAGS supported claims (#4 cited wrong line though
the stat was present; #12/#24 missed genuine support). That's a data-balance issue:
- v3 dataset tweak: raise the `supported` share and add "support is present but
  phrased differently / buried mid-source" examples so the model learns to FIND
  in-bundle support, not just flag. Then one more Colab retrain.
- This directly improves the retriever pipeline too (more sources present = model
  must be good at spotting real support among them).

## Cost / blockers
- Fetch half: FREE, reuse existing code — buildable now.
- Search half: free backend (DuckDuckGo/Wikipedia) works with no key; a real search
  API (Brave free tier / Serper / Tavily) is better — needs a key.
- Retrain for the over-flagging fix: ~40 min Colab, same one-click flow.

## Suggested order
1. `retriever.py` with a free search backend + reused fetch/extract — verify it
   builds a real bundle from a pasted claim. (free, now)
2. `demo.py` wiring retriever → v2 adapter inference → cited output. (free, now)
3. v3 dataset rebalance (more supported) → retrain → re-eval → re-calibrate. (later)
