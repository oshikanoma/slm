#!/usr/bin/env python3
"""Browser UI for the Cited Newsroom Verifier SLM (v6) — friendlier than the terminal.

Two tabs:
  1. AP Style   — paste a sentence; the model flags AP Stylebook violations + suggests
                  fixes. No sources/retrieval needed (AP is a rule check, not evidence).
  2. Verify     — paste a passage; a retriever pulls real web sources, then the model
                  emits cited verdicts (supported / unsupported / misleading) over them.

Run:
  export TAVILY_API_KEY=tvly-...     # for the Verify tab's open-web retrieval (optional)
  python app.py                       # opens http://127.0.0.1:7860

The model loads once on first use (~1 min), then each check is ~10-15s on Apple Silicon.
"""
from __future__ import annotations

import json
import os

import gradio as gr

from eval import Scenario, build_messages
from retriever import build_bundle, _active_backend

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "artifacts/qwen3-verifier-lora-v6"

_MODEL = {"m": None, "tok": None, "dev": None}  # lazy singleton


def _load():
    if _MODEL["m"] is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16 if dev == "mps" else torch.float32).to(dev)
    m = PeftModel.from_pretrained(m, ADAPTER).to(dev)
    m.eval()
    _MODEL.update(m=m, tok=tok, dev=dev)


def _run(passage: str, sources: list[dict]) -> dict:
    _load()
    import torch
    tok, m, dev = _MODEL["tok"], _MODEL["m"], _MODEL["dev"]
    scn = Scenario(id="ui", bucket="", passage=passage, sources=sources, gold_verdicts=[])
    prompt = tok.apply_chat_template(build_messages(scn), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    inp = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=512, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(text)
    except Exception:
        return {"_raw": text}


# ---- Tab 1: AP Style ---------------------------------------------------------

def ap_check(sentence: str):
    if not sentence.strip():
        return "Enter a sentence."
    obj = _run(sentence, [])  # no sources — AP is a rule check
    aps = [v for v in obj.get("verdicts", []) if v.get("type") == "ap_style"]
    if "_raw" in obj:
        return f"(model returned non-JSON)\n\n{obj['_raw']}"
    if not aps:
        return "✅ No AP Style issues flagged."
    lines = []
    for v in aps:
        lines.append(f"⚑ **{v.get('span','')}**")
        lines.append(f"   • Rule: {v.get('rule','')}")
        lines.append(f"   • Suggested: {v.get('suggestion','')}")
        if v.get("explanation"):
            lines.append(f"   • Why: {v.get('explanation')}")
        lines.append("")
    return "\n".join(lines)


# ---- Tab 2: Verify (retrieve + cite) -----------------------------------------

def verify(passage: str, k: int):
    if not passage.strip():
        return "Enter a passage.", ""
    backend, _ = _active_backend()
    bundle = build_bundle(passage, k=int(k))
    if not bundle:
        return (f"No sources retrieved (backend: {backend}). "
                "Set TAVILY_API_KEY for open-web search.", "")
    src_md = "\n".join(f"- [{s['url']}]({s['url']})" for s in bundle)
    obj = _run(passage, bundle)
    if "_raw" in obj:
        return f"(model returned non-JSON)\n\n{obj['_raw']}", src_md
    if not obj.get("verdicts"):
        return "✅ Clean — nothing to flag.", src_md
    icons = {"supported": "✅", "unsupported": "❌", "misleading": "⚠️", "ap_flag": "⚑"}
    lines = []
    for v in obj["verdicts"]:
        vd = v.get("verdict", "")
        lines.append(f"{icons.get(vd,'•')} **{vd.upper()}** — {v.get('span','')}")
        if v.get("source_url"):
            lines.append(f"   • Cited: {v['source_url']}")
            lines.append(f"   • Backing: \"{v.get('evidence_span','')}\"")
        elif v.get("checked_source_url"):
            lines.append(f"   • Checked (no support found): {v['checked_source_url']}")
        if v.get("explanation"):
            lines.append(f"   • {v['explanation']}")
        lines.append("")
    return "\n".join(lines), src_md


with gr.Blocks(title="Newsroom Verifier SLM (v6)") as app:
    gr.Markdown("# 📰 Cited Newsroom Verifier SLM (Qwen3-1.7B, v6)\n"
                "A tiny local model that flags AP Style issues and verifies claims against "
                "**real retrieved sources** — only saying 'supported' when it can quote the evidence.")
    with gr.Tab("AP Style check"):
        gr.Markdown("Paste a sentence. The model flags AP Stylebook violations and suggests fixes "
                    "(it does **not** rewrite your text).")
        ap_in = gr.Textbox(label="Sentence", lines=2,
                           placeholder="The meeting has 5 members and starts at 3:00 PM on December 8.")
        ap_btn = gr.Button("Check AP Style", variant="primary")
        ap_out = gr.Markdown(label="Result")
        ap_btn.click(ap_check, ap_in, ap_out)
        gr.Examples([["The meeting has 5 members and starts at 3:00 PM on December 8."],
                     ["Enrollment rose 12 percent, and the flag was red, white, and blue."]], ap_in)
    with gr.Tab("Verify a claim (with sources)"):
        gr.Markdown("Paste a passage. A retriever pulls real web sources, then the model "
                    "returns cited verdicts. (Set `TAVILY_API_KEY` for open-web search.)")
        v_in = gr.Textbox(label="Passage / claim", lines=3,
                         placeholder="The James Webb Space Telescope launched on December 25, 2021 from French Guiana.")
        v_k = gr.Slider(2, 6, value=4, step=1, label="Sources to retrieve")
        v_btn = gr.Button("Retrieve + verify", variant="primary")
        v_out = gr.Markdown(label="Verdicts")
        v_src = gr.Markdown(label="Retrieved sources")
        v_btn.click(verify, [v_in, v_k], [v_out, v_src])


if __name__ == "__main__":
    print("Loading UI... model loads on first check (~1 min).")
    app.launch(inbrowser=True)
