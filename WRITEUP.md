# Cited Newsroom Verifier SLM — Results & Writeup

**Author:** Tiffany Lam · **Base model:** Qwen3-1.7B-Instruct (QLoRA via Unsloth)
**Repo:** github.com/oshikanoma/slm · **Deliverables:** dataset, fine-tuned adapters (v1–v3), eval harness, thin-retriever demo.

---

## 1. The behavior

A **cited, evidence-bounded newsroom verifier.** Given a passage plus a bundle of candidate sources (each a URL + text), the model flags every claim/quote/link as `supported`, `unsupported`, or `misleading`, and:

- For `supported`, it must cite a **real** `source_url` (verbatim from the bundle) and quote the **verbatim** `evidence_span` that backs the claim.
- It never asserts `supported` from its own memory, and never invents a URL or quote.

The single forbidden failure is **uncited assertion / fabricated citation** — vouching for a claim without real, traceable evidence. This is the IFCN principle ("the burden of proof rests on the claim") made trainable. It was chosen over "AP-style corrections" because that knowledge is promptable; **restraint under evidence is not** — a prompt can't reliably stop a small model from inventing citations, but a dataset can.

## 2. Why an SLM (the thesis)

A 1.7B model fine-tuned on a clean, purpose-built dataset reliably performs one disciplined behavior that the *untrained* base model cannot — cheaper, faster, and fully local (no data leaves the machine). The results below are the evidence: on the exact same inputs, the base model fabricates citations ~89% of the time; the tuned model does not.

## 3. Data (the ~80% of the work)

- **1,000 training + 115 golden eval records**, kept strictly disjoint (0 passage overlap, verified).
- **Legally clean sources only:** Texas Tribune (CC republishing) + Wikinews (CC BY 2.5) for real `supported`/`unsupported` cases; teacher-distilled **synthetic** for the trap buckets (`true_but_unsupported`, `distractor`, `misleading`) that real journalism never provides but that teach the core restraint.
- Every record passes an **objective gate** (span verbatim in passage, citation verbatim in source) — nothing fabricated can enter the set.
- Golden set built to the spec's bucket mix so **knowledge leakage is measurable** (it can only be caught by the trap buckets).

Pipeline scripts: `ingest_texastribune.py`, `ingest_wikinews.py`, `autolabel.py`, `datagen.py`, `build_dataset.py`, `build_golden.py`.

## 4. The eval harness

Implemented in `eval.py`, run identically on base and tuned:
- **Headline:** `spec_pass_rate` — fraction of records fully correct (one boolean each).
- **Objective sub-metrics:** fabricated_citation_rate, knowledge_leakage_rate, citation_precision, flag_recall, etc. (no LLM needed).
- **LLM-as-judge:** 0/1/2 rubric (spec adherence, robustness, task quality, consistency) + calibration against the human expert (Cohen's kappa).
- **Significance:** McNemar's exact test + bootstrap CI on the base-vs-tuned spec_pass delta.

## 5. Results — the experiment

**Framing:** base = control, tuned = treatment, single manipulated variable = the fine-tuning. H0 = fine-tuning makes no difference to spec_pass.

### Headline (v3, 115-record golden set)

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| **spec_pass_rate ↑** | **0.00%** | **47.83%** | **+47.83%** |
| fabricated_citation_rate ↓ | 89.39% | 9.52% | **−79.87%** |
| knowledge_leakage_rate ↓ | 8.96% | 1.49% | −7.46% |
| citation_validity_rate ↑ | 10.61% | 90.48% | +79.87% |
| citation_precision ↑ | 1.52% | 66.67% | +65.15% |
| flag_recall ↑ | 24.05% | 69.62% | +45.57% |

**Significance:** McNemar exact **p < 0.0001**; tuned-only wins = 55, base-only wins = **0**; bootstrap 95% CI on the delta = [+39.1%, +56.5%]. **H0 rejected.** All §5.5 win-condition checks pass — **WIN**.

### Per-bucket (v3 tuned)
misleading **100%**, unsupported 53%, true_but_unsupported 48%, supported 39%, distractor 19%. The model is strongest exactly where it matters (holding the line on quotes and unsupported claims) and weakest on distractors (hardest bucket).

## 6. Iteration — the loop, closed three times

Each version was driven by a finding from the previous eval. This is the methodology working, not a single lucky run.

| Ver | Change (and what drove it) | Result |
|---|---|---|
| **v1** | Initial 1,000-example dataset | spec_pass 0.9%→61%, p<0.0001 — **but calibration exposed a spec disagreement** (below) |
| **v2** | Added `checked_source_url`/`nearest_span` to every verdict (spec revision) | Full win incl. fabrication/leakage; calibration kappa 0.155→0.286 |
| **v3** | Rebalanced training mix, supported 21%→28% (calibration found over-flagging) | supported-bucket 31%→39%, clean-passage handling 33%→53% |

### The calibration finding (the most valuable moment)
Calibrating the LLM judge against the human expert produced **kappa 0.155** — too low to trust. The disagreements weren't noise: the expert's editorial standard ("always show me the source you checked, even when flagging") **differed from the v1 spec** ("unsupported = cite nothing"). This is exactly what calibration exists to surface. Resolution: **revise the spec** (v2's `checked_source_url` — the model now always shows its work), which is a stronger design *and* raised agreement. A second round then exposed **over-flagging** (the model marked genuinely-supported claims unsupported), traced to a data imbalance (73% of training was flag-buckets) and fixed in v3 by rebalancing — per the project rule, *fix the data, not the hyperparameters*.

## 7. Live demo — thin retriever (spec §8)

`retriever.py` + `demo.py` wrap the tuned model with a retrieval layer:
`passage → web search → fetch + extract → source bundle → SLM emits cited verdicts`.
The model stays evidence-bounded (only quotes what the retriever supplies, so it cannot fabricate); the retriever does the source-hunting. Search backend is pluggable (Tavily/Brave/Serper by API key, free Wikipedia fallback with no key).

**Observed base-vs-tuned contrast** on the same retrieved bundle (claim: *"The JWST launched on December 25, 2021 from French Guiana"*):
- **Tuned:** `supported`, real URL, a real backing span quoted.
- **Base:** `supported`, real URL, but **`evidence_span: ""`** — it vouched for the claim while quoting *nothing*. This is the forbidden failure caught live.

## 8. Honest limitations

- **Evidence-span precision:** the model reliably picks the right *source* and correct verdict, but sometimes quotes a loose *backing span* rather than the ideal sentence (seen in the demo). A v4 data fix (more "exact backing span" examples) would tighten this.
- **`supported` recall (39%):** improved across versions but still the model's weaker half — it errs toward caution. For a *verifier*, over-caution is the safer failure mode, but there's headroom.
- **Judge quality:** calibration used a free Groq model that made some grading errors; a stronger judge (GPT-4o) would give a cleaner final kappa. The objective metrics need no LLM and are airtight.
- **Retriever scope:** the free Wikipedia backend is encyclopedia-scoped; a search-API key unlocks open-web retrieval.

## 9. Bottom line

A tiny, local, fine-tuned model reliably does the one disciplined thing the base model can't: **only say what it can cite.** The improvement is large (spec_pass 0→48%), statistically significant (p<0.0001), and directional across three data-driven iterations. The thesis holds.
