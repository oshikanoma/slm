---
title: Cited Newsroom Verifier SLM
emoji: 📰
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: A fine-tuned Qwen3-1.7B that verifies news claims against real sources and flags AP Style.
---

# The Verifier — Cited Newsroom Verifier SLM

A fine-tuned **Qwen3-1.7B** (QLoRA) that acts as a newsroom copy desk:

- **Verifies claims** against retrieved sources — it only marks a claim `supported` when it
  can quote the verbatim backing sentence from a real source, and refuses to vouch otherwise
  (the anti-hallucination behavior a prompt can't guarantee).
- **Verifies links in your copy** — paste text with URLs and it checks each link is alive
  (dead / redirect / alive) *and* whether the linked page actually backs the adjacent claim.
- **Checks whole documents** — upload a `.txt`, `.md`, `.pdf`, or `.docx` and it splits the
  document into sections, verifies every claim and link, and tallies unbacked factual
  assertions (the defamation/libel-risk items).
- **Flags AP Style** issues deterministically (numbers, dates, times, ages, attribution, …).

**Thesis:** a tiny fine-tuned model beats prompted frontier models at this narrow task —
it beats a prompted 120B by ~2.4× on strict spec-pass. Behavior from data, not scale.

## Configuration
- **Model:** loaded from the Hub — set env `HF_ADAPTER_REPO` to your adapter repo
  (default points to a placeholder; edit `app.py` or set the Space variable).
- **Web retrieval:** set a Space **secret** `TAVILY_API_KEY` for open-web search; without it,
  the app falls back to a free Wikipedia search backend.

## Notes
- On the free **CPU** tier the 1.7B model is slow (~30–60s per verification). Upgrading the
  Space to a GPU makes it fast. AP-style checks (toggle retrieval off) are instant either way.
- Full project, dataset, eval harness, and results: see the GitHub repo.
