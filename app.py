#!/usr/bin/env python3
"""Browser UI for the Cited Newsroom Verifier SLM (v6).

ONE input box. Paste a passage; the model returns BOTH:
  - AP Style flags (rule + suggested fix; it never rewrites your text), and
  - claim verdicts (supported / unsupported / misleading) against retrieved sources.

Toggle "retrieve sources" off for a fast AP-only pass (no web calls, shorter output).

Run:
  export TAVILY_API_KEY=tvly-...     # open-web retrieval (optional; Wikipedia fallback)
  python app.py                       # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os

import gradio as gr

from eval import Scenario, build_messages
from retriever import build_bundle, _active_backend

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "artifacts/qwen3-verifier-lora-v6"
_MODEL = {"m": None, "tok": None, "dev": None}


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


def _run(passage: str, sources: list[dict], max_new_tokens: int) -> dict:
    _load()
    import torch
    tok, m, dev = _MODEL["tok"], _MODEL["m"], _MODEL["dev"]
    scn = Scenario(id="ui", bucket="", passage=passage, sources=sources, gold_verdicts=[])
    prompt = tok.apply_chat_template(build_messages(scn), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    inp = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(text)
    except Exception:
        return {"_raw": text}


def _fmt(obj: dict) -> str:
    if "_raw" in obj:
        return f"_(model returned non-JSON)_\n\n```\n{obj['_raw'][:600]}\n```"
    verdicts = obj.get("verdicts", [])
    if not verdicts:
        return "✅ **Clean** — no AP issues and nothing to flag."
    icons = {"supported": "✅ SUPPORTED", "unsupported": "❌ UNSUPPORTED",
             "misleading": "⚠️ MISLEADING", "ap_flag": "⚑ AP STYLE"}
    ap, claims = [], []
    for v in verdicts:
        vd = v.get("verdict", "")
        head = f"**{icons.get(vd, vd)}** — {v.get('span','')}"
        if vd == "ap_flag":
            body = [f"  • Rule: {v.get('rule','')}", f"  • Suggested: {v.get('suggestion','')}"]
            ap.append(head + "\n" + "\n".join(body))
        else:
            body = []
            if v.get("source_url"):
                body.append(f"  • Cited: {v['source_url']}")
                body.append(f"  • Backing: \"{v.get('evidence_span','')}\"")
            elif v.get("checked_source_url"):
                body.append(f"  • Checked (no support found): {v['checked_source_url']}")
            if v.get("explanation"):
                body.append(f"  • {v['explanation']}")
            claims.append(head + ("\n" + "\n".join(body) if body else ""))
    out = []
    if claims:
        out.append("### Claim verdicts\n" + "\n\n".join(claims))
    if ap:
        out.append("### AP Style\n" + "\n\n".join(ap))
    return "\n\n".join(out)


def analyze(passage: str, do_retrieve: bool, k: int):
    if not passage.strip():
        return "Enter a passage.", ""
    sources, src_md = [], "_(retrieval off — AP-only, fast)_"
    if do_retrieve:
        backend, _ = _active_backend()
        sources = build_bundle(passage, k=int(k))
        src_md = ("\n".join(f"- [{s['url']}]({s['url']})" for s in sources)
                  if sources else f"_(no sources retrieved; backend: {backend})_")
    # shorter cap when not retrieving -> faster AP-only pass
    max_tok = 512 if do_retrieve else 256
    obj = _run(passage, sources, max_tok)
    return _fmt(obj), src_md


with gr.Blocks(title="Newsroom Verifier SLM (v6)") as app:
    gr.Markdown("# 📰 Cited Newsroom Verifier SLM (Qwen3-1.7B, v6)\n"
                "Paste a passage. The model flags **AP Style** issues *and* verifies **claims** "
                "against retrieved sources — only saying 'supported' when it can quote the evidence. "
                "It flags and suggests; it never rewrites your text.")
    inp = gr.Textbox(label="Passage", lines=3,
                     placeholder="Enrollment rose 12 percent, according to a district report.")
    with gr.Row():
        retrieve = gr.Checkbox(value=False, label="Retrieve sources & verify claims "
                               "(slower; uncheck for fast AP-only)")
        k = gr.Slider(2, 6, value=4, step=1, label="Sources")
    btn = gr.Button("Analyze", variant="primary")
    out = gr.Markdown()
    src = gr.Markdown()
    btn.click(analyze, [inp, retrieve, k], [out, src])
    gr.Examples([
        ["The meeting starts at 3:00 PM.", False, 4],
        ["Enrollment rose 12 percent this year.", False, 4],
        ["The James Webb Space Telescope launched on December 25, 2021 from French Guiana.", True, 4],
    ], [inp, retrieve, k])


if __name__ == "__main__":
    print("Loading UI... model loads on first analyze (~1 min), then stays warm.")
    app.launch(inbrowser=True)
