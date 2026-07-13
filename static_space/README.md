---
title: The Verifier — Cited Newsroom Verifier SLM
emoji: 📰
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: Fine-tuned Qwen3-1.7B newsroom verifier — live AP-style checker + eval results.
---

# The Verifier

A static demo for the **Cited Newsroom Verifier SLM** (Qwen3-1.7B + QLoRA).

- **Live AP Style checker** — runs entirely in your browser (JS port of the deterministic
  rule engine), flags AP violations instantly.
- **Claim verification** — showcased with a real base-vs-tuned run; the model itself is
  published on the [HF Hub](https://huggingface.co/tiffuhknee/qwen3-1.7b-newsroom-verifier)
  and runs in the notebook / local app.
- **Results** — frontier-model leaderboard + six-round training progression.

Full code, dataset, and eval harness: [github.com/oshikanoma/slm](https://github.com/oshikanoma/slm)
