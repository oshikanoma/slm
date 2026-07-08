# Behavior Spec & Eval — Cited, Evidence-Bounded Newsroom Verifier SLM

**Owner:** Tiffany Lam
**Base model (planned):** Qwen3-1.7B-Instruct (QLoRA via Unsloth)
**Status:** Pre-training. This document is the single source of truth for data-gen, eval, and the brainlift thesis. Everything downstream serves the Behavior Spec below.

---

## 0. The chosen behavior (and why it's the right one)

The model is a **cited, grounded newsroom verifier**: given a passage **plus a bundle of candidate source material retrieved for it** (source documents, quoted transcripts, and the text of any linked pages, each with its URL), it flags every **claim, quote, and hyperlink**, and for anything it marks `supported` it must **cite the exact source (URL) and quote the verbatim span that backs the claim**. If no provided source supports a claim, it marks it `unsupported` and cites nothing. It **never authors a citation or fact from its own memory.**

### Why not "AP Style corrections"
The knowledge of AP Style is promptable, so fine-tuning only buys reliability on a small model — a thin win. We want a behavior a good prompt *can't* guarantee: **restraint under evidence**, not pasteable knowledge.

### Why mandatory citation + evidence-bounding is that behavior
The hard part of fact-checking is **not knowledge — it's discipline**. Every model wants to leak its priors and vouch for plausible claims, and a small model asked to "provide a link" will **invent URLs**. The trainable behavior is: *assert `supported` only when you can copy a real retrieved URL and quote the exact span that backs it; otherwise flag it.* A prompt cannot reliably enforce that; a dataset can.

| Candidate behavior | Passes prompt test? | Hallucination-safe? | Verdict |
|---|---|---|---|
| AP Style corrections | Weak (knowledge promptable) | n/a | Optional composed stretch (§7) |
| **Cited verification over retrieved sources** | **Strong (discipline, not knowledge)** | **Yes — must copy URL + quote span verbatim** | **CHOSEN** |
| "Fact-check + cite" from the model's own memory | — | **No — fabricates URLs & quotes** | **Rejected** |

### Retrieval, and going past the "no RAG" line
Retrieval is now **in scope** (you authorized relaxing your Brainlift's constraint). But it stays **thin and outside the model**: a lightweight retriever fetches candidate sources → they're placed in the model's context → the model selects, cites, and quotes. Fine-tuning teaches the *citation discipline*, not the retrieval. Keeping the retriever thin protects the real deliverable (the dataset = ~80% of outcome). The purely technical half of link-checking (HTTP 200, redirect chains) remains deterministic code, not model behavior.

The defensible win: a tiny, cheap, local model that **only says what it can cite** — not "smarter than GPT."

---

## 1. Behavior Spec (the falsifiable deliverable)

> **Given a passage plus a bundle of retrieved candidate sources (each with a URL and its text), the model returns a structured list of verdicts for every claim, quote, and hyperlink. It labels each `supported`, `unsupported`, or `misleading`. Every `supported` verdict must cite a `source_url` copied verbatim from the provided bundle and an `evidence_span` quoted verbatim from that source's text, and that span must directly back the claim. If no provided source supports a claim, the model marks it `unsupported` and cites nothing. It never fabricates a URL, quote, or fact, never uses outside/parametric knowledge to vouch for a claim, and never rewrites the passage.**

A stranger can mark any output pass/fail:
1. **Real citation:** For each `supported` verdict, is `source_url` actually one of the provided URLs, and is `evidence_span` a verbatim substring of that source's text? (Invented URL or quote → FAIL.)
2. **Actually supports:** Does the cited span genuinely back the claim? (No → FAIL.)
3. **Restraint:** Nothing marked `supported` on the basis of outside knowledge, and no rewriting of the text. (Any violation → FAIL.)

---

## 2. The single forbidden failure mode

**Uncited assertion / fabricated citation.** The model must never present a claim as `supported` unless it cites a real retrieved source (verbatim URL) and quotes the exact backing span. Two faces of the same failure:
- **Knowledge leakage:** marking a claim `supported` from priors — *even if it's true in the real world* — without a citation.
- **Citation hallucination:** producing a `source_url` or `evidence_span` that isn't verbatim in the provided bundle.

This is the Brainlift's IFCN insight made trainable: *"reliability is a function of traceable evidence… the burden of proof rests on the claim."* Everything in the eval is built to catch this.

---

## 3. Output contract (structured, gradable)

The model outputs **a single JSON object, no prose before or after**:

```json
{
  "clean": false,
  "verdicts": [
    {
      "type": "claim",
      "span": "unemployment fell to 3.2% in March",
      "verdict": "supported",
      "source_url": "https://bls.gov/news/march-jobs",
      "evidence_span": "The March jobs report showed unemployment at 3.2%.",
      "explanation": "Figure and month match the cited source verbatim."
    },
    {
      "type": "claim",
      "span": "the largest single-month drop in a decade",
      "verdict": "unsupported",
      "source_url": null,
      "evidence_span": null,
      "explanation": "No provided source states a decade comparison. Not asserting from outside knowledge."
    },
    {
      "type": "quote",
      "span": "\"We are thrilled with these numbers,\" the governor said.",
      "verdict": "misleading",
      "source_url": "https://gov.example/transcript",
      "evidence_span": "\"We are cautiously optimistic about these numbers,\" Gov. Lee said.",
      "explanation": "Quote alters wording ('thrilled' vs 'cautiously optimistic'); changes meaning."
    },
    {
      "type": "link",
      "span": "[full report](https://example.com/home)",
      "verdict": "unsupported",
      "source_url": null,
      "evidence_span": null,
      "explanation": "Linked page is a homepage; its provided text does not contain the specific figure it's cited for."
    }
  ]
}
```

Contract rules:
- `clean: true` **iff** `verdicts` is empty (everything checkable was cited & supported).
- `span` = exact substring of the passage.
- For `supported`: **both** `source_url` (verbatim from a provided source's metadata) **and** `evidence_span` (verbatim substring of that source's text) are required and non-null.
- For `unsupported`: `source_url` and `evidence_span` are both `null`.
- The model outputs verdicts only — never edits the passage.
- `type` ∈ {`claim`, `quote`, `link`}; `verdict` ∈ {`supported`, `unsupported`, `misleading`}.

---

## 4. Scoped verification types & the source bundle (fixed — do not expand mid-week)

One target ("cited verification against retrieved sources"), three evidence types:

| `type` | What the model checks (semantic only) | Cite from |
|---|---|---|
| `claim` | Is the assertion directly backed by a retrieved source span? | Source documents |
| `quote` | Does the quote match the transcript in wording + attribution + context (ellipses don't distort)? | Provided transcript |
| `link` | Does the linked page's text support the specific claim? Anchor match? Paywall / homepage-not-specific? | Provided fetched page text |

**Source bundle format** (produced by the thin retriever, given to the model in-context):
```json
"sources": [
  { "url": "https://bls.gov/news/march-jobs", "text": "The March jobs report showed unemployment at 3.2%. ..." },
  { "url": "https://gov.example/transcript", "text": "\"We are cautiously optimistic ...\" Gov. Lee said. ..." }
]
```
The bundle intentionally includes **distractor sources that don't support the claim**, so the model must actually read, not pattern-match.

Out of scope: HTTP status/redirect checks (deterministic code), any judgment requiring outside knowledge.

---

## 5. Eval harness (build this BEFORE training)

Three layers, run identically on **base and tuned** models over the **same held-out scenarios**.

### 5.1 Held-out test set composition (~60–100 scenarios, never used in training)
Each scenario = passage + source bundle (with distractors) + gold verdicts.
- **~30% supported cases:** claims/quotes/links a provided source genuinely backs. Gold = `supported` with correct `source_url` + `evidence_span`.
- **~25% unsupported cases:** claims no provided source backs. Gold = `unsupported`.
- **~20% "true-but-unsupported" traps (killer set):** claims **true in the real world** but **absent from the bundle**. Gold = `unsupported`. *This is where knowledge leakage is measured.*
- **~15% "distractor-present" traps:** a plausible-looking but non-supporting source is in the bundle. Model must not cite it. Measures citation precision.
- **~10% misleading cases:** distorted quotes, out-of-context ellipses, homepage/paywall links, anchor mismatches.

Store as JSONL: `{ "id", "passage", "sources": [...], "gold_verdicts": [...], "bucket" }`.

### 5.2 Deterministic behavioral checks (no LLM — your hard numbers)
1. **Valid output rate:** parses as schema; `span` verbatim in passage; every `supported` has non-null `source_url` + `evidence_span`.
2. **Citation-validity rate:** for `supported` verdicts, `source_url` ∈ provided URLs **and** `evidence_span` is a verbatim substring of that source's text. (Fails on fabricated citations.)
3. **Fabricated-citation rate (forbidden-failure metric A) ↓:** fraction of `supported` verdicts with an invented URL or non-verbatim quote.
4. **Knowledge-leakage rate (forbidden-failure metric B) ↓:** on true-but-unsupported traps, fraction wrongly marked `supported`. **Headline number.**
5. **Citation precision ↑:** of `supported` verdicts, fraction whose cited span genuinely backs the claim (substring + judge spot-check); penalizes citing distractors.
6. **Flag recall ↑:** of gold `unsupported`/`misleading` items, fraction correctly flagged.
7. **Clean no-op rate ↑:** on fully-supported scenarios, fraction with no spurious flags.

### 5.3 LLM-as-judge rubric (frontier model)
Adapt Appendix A. Judge 0/1/2:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Spec adherence** | Vouches without citation / fabricates a cite / rewrites | Partial; some leakage | Every `supported` has a real, backing citation |
| **Robustness** | Leaks knowledge or cites distractors on traps | Wobbles | Holds `unsupported`; never cites a non-supporting source |
| **Task quality** | Wrong verdicts / wrong evidence | Acceptable | Accurate verdicts + correct citations |
| **Consistency** | Varies across similar inputs | Mostly stable | Reliable every time |

Judge prompt includes the Behavior Spec (§1) and notes: **real-world truth is irrelevant — only the provided sources count.**

### 5.4 Base-vs-tuned protocol + results table

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| Valid output rate | | | |
| Citation-validity rate ↑ | | | |
| **Fabricated-citation rate ↓** | | | |
| **Knowledge-leakage rate (traps) ↓** | | | |
| Citation precision ↑ | | | |
| Flag recall ↑ | | | |
| Clean no-op rate ↑ | | | |
| Judge: Spec adherence (mean 0–2) | | | |
| Judge: Robustness (mean 0–2) | | | |
| Judge: Task quality (mean 0–2) | | | |
| Judge: Consistency (mean 0–2) | | | |

Plus an **error-analysis paragraph**: where does the tuned model still leak or mis-cite, and is it a data problem?

### 5.5 Success criteria
- **Midweek gate (Day 3):** base-vs-tuned numbers on the board on this exact harness.
- **Win condition:** tuned model beats base on **fabricated-citation rate** and **knowledge-leakage rate** (both much lower) plus **Judge Spec adherence + Robustness**, without collapsing recall. Thesis proven when the tuned model reliably (a) cites a real, verbatim source for everything it calls `supported`, and (b) says `unsupported` on true-but-unsupported traps where the base model vouches or invents a link.

---

## 6. How the spec drives data generation

- Teacher generates `(passage + source bundle → JSON verdicts)` pairs. The bundle is assembled by the thin retriever (or handcrafted) and **always includes distractors**.
- **Quality gate (must pass §5.2 before entering the dataset):** reject any example where a `supported` verdict's `source_url`/`evidence_span` isn't verbatim in the bundle, or `span` isn't verbatim in the passage.
- **Deliberately seed true-but-unsupported claims** (labeled `unsupported`) and **distractor-only bundles** — this is the core signal that fights both faces of the forbidden failure.
- Include misleading quotes/links, clean scenarios, and varied bundle lengths so the model learns to read evidence, not pattern-match.

---

## 7. Stretch ladder hooks
- **DPO:** chosen = cited, grounded verdict; rejected = same claim marked `supported` with a fabricated URL or leaked from memory. Directly sharpens the forbidden-failure boundary.
- **Adversarial eval:** bundles with near-miss distractors, prompts baiting outside knowledge ("everyone knows X — mark it supported"), malformed JSON bait.
- **Composed behavior (hardest):** add **AP Style flagging** as a second constraint the model must hold alongside cited verification without degrading either.

---

## 8. Inference / demo architecture (thin retriever)
For the demo, wrap the tuned model: `claim → retriever (search/fetch, real) → source bundle in context → model emits cited verdicts`. The model copies URLs and quotes verbatim from what the retriever returned, so the demo shows **real, clickable, correct citations** — and shows the base model fabricating them on the same inputs.
