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
from ap_rules import ap_check  # deterministic AP checker (code, not model)

BASE = "Qwen/Qwen3-1.7B"
# Local adapter path if present, else fall back to the published HF Hub repo (Colab/remote).
_LOCAL_ADAPTER = "artifacts/qwen3-verifier-lora-v6"
ADAPTER = _LOCAL_ADAPTER if os.path.isdir(_LOCAL_ADAPTER) else \
    os.environ.get("HF_ADAPTER_REPO", "tiffuhknee/qwen3-1.7b-newsroom-verifier")
_MODEL = {"m": None, "tok": None, "dev": None}


def _load():
    if _MODEL["m"] is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if torch.cuda.is_available():            # Colab / GPU box
        dev, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev, dtype = "mps", torch.float16
    else:
        dev, dtype = "cpu", torch.float32
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype).to(dev)
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
    # Claim verdicts only — AP is handled deterministically by ap_rules, not the model.
    verdicts = [v for v in obj.get("verdicts", []) if v.get("verdict") != "ap_flag"]
    if not verdicts:
        return "### Claim verdicts\n✅ Nothing flagged against the retrieved sources."
    icons = {"supported": "✅ SUPPORTED", "unsupported": "❌ UNSUPPORTED",
             "misleading": "⚠️ MISLEADING"}
    claims = []
    for v in verdicts:
        vd = v.get("verdict", "")
        head = f"**{icons.get(vd, vd)}** — {v.get('span','')}"
        body = []
        if v.get("source_url"):
            body.append(f"  • Cited: {v['source_url']}")
            body.append(f"  • Backing: \"{v.get('evidence_span','')}\"")
        elif v.get("checked_source_url"):
            body.append(f"  • Checked (no support found): {v['checked_source_url']}")
        if v.get("explanation"):
            body.append(f"  • {v['explanation']}")
        claims.append(head + ("\n" + "\n".join(body) if body else ""))
    return "### Claim verdicts\n" + "\n\n".join(claims)


def _fmt_ap(hits: list[dict]) -> str:
    if not hits:
        return "✅ **AP Style:** no issues found."
    lines = ["### ⚑ AP Style (deterministic checker)"]
    for h in hits:
        lines.append(f"- **{h['span']}** — {h['rule']}\n  → suggested: {h['suggestion']}")
    return "\n".join(lines)


def analyze(passage: str, do_retrieve: bool, k: int):
    """AP style = deterministic code (instant, catches every violation).
    Claim verification = the SLM (only when sources are retrieved)."""
    if not passage.strip():
        return "Enter a passage.", ""
    # 1) AP: always, instant, complete.
    ap_md = _fmt_ap(ap_check(passage))
    # 2) Claim verification: only meaningful WITH sources (the SLM's job).
    if not do_retrieve:
        return ap_md + "\n\n_(claim verification skipped — turn on 'retrieve & verify' " \
               "to check claims against real sources)_", "_(retrieval off)_"
    backend, _ = _active_backend()
    sources = build_bundle(passage, k=int(k))
    src_md = ("\n".join(f"- [{s['url']}]({s['url']})" for s in sources)
              if sources else f"_(no sources retrieved; backend: {backend})_")
    if not sources:
        return ap_md + "\n\n_(no sources retrieved — can't verify claims)_", src_md
    obj = _run(passage, sources, 512)
    claim_md = _fmt(obj)
    return claim_md + "\n\n" + ap_md, src_md


NEWSPAPER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Spectral:ital,wght@0,400;0,600;1,400&family=Georgia&display=swap');

.gradio-container { background: #f4f1ea !important; max-width: 900px !important; margin: 0 auto !important; }
#masthead { text-align: center; padding: 32px 0 6px; }
#masthead .paper-name {
  font-family: 'Playfair Display', Georgia, serif; font-weight: 900;
  font-size: 3.4rem; letter-spacing: -0.5px; color: #1a1a1a; line-height: 1.1;
  margin: 22px 0 10px;
}
#masthead .kicker {
  font-family: 'Spectral', Georgia, serif; font-style: italic; color: #6b6257;
  font-size: 1.05rem; margin-top: 6px; line-height: 1.5;
}
#masthead .ruleline {
  border: none; border-top: 3px double #1a1a1a; margin: 18px auto 4px; width: 100%;
}
#masthead .dateline {
  font-family: 'Spectral', Georgia, serif; letter-spacing: 1px;
  font-size: 0.78rem; color: #6b6257; display: flex; justify-content: space-between;
  border-bottom: 1px solid #cfc7b8; padding: 2px 2px 10px;
}
.gr-box, .gr-form, textarea, .gr-input {
  font-family: 'Spectral', Georgia, serif !important; background: #fffdf8 !important;
  border-color: #d8d0c0 !important; color: #1a1a1a !important;
}
label span { font-family: 'Spectral', Georgia, serif !important; color: #3a352c !important; }
button.primary, #analyze-btn {
  background: #8b2635 !important; border: none !important; color: #fff !important;
  font-family: 'Spectral', Georgia, serif !important; font-weight: 600 !important;
  letter-spacing: 0.3px !important; font-size: 0.98rem !important;
}
button.primary:hover, #analyze-btn:hover { background: #6f1e2a !important; }
#result, #sources {
  font-family: 'Spectral', Georgia, serif !important; background: #fffdf8 !important;
  border: 1px solid #d8d0c0 !important; border-radius: 2px !important; padding: 16px 20px !important;
  color: #1a1a1a !important; line-height: 1.55 !important;
}
#result h3 {
  font-family: 'Playfair Display', Georgia, serif !important; font-size: 1.1rem !important;
  border-bottom: 1px solid #cfc7b8; padding-bottom: 4px; margin-top: 4px;
}
#sources { font-size: 0.85rem !important; color: #4a453c !important; }
.gr-examples, .gr-samples-table { font-family: 'Spectral', Georgia, serif !important; }
footer { display: none !important; }
"""

# Deterministic dateline built at import (Date.now unavailable in some sandboxes; use fixed label).
with gr.Blocks(title="The Verifier", css=NEWSPAPER_CSS, theme=gr.themes.Base(
        primary_hue="red", neutral_hue="stone",
        font=[gr.themes.GoogleFont("Spectral"), "Georgia", "serif"])) as app:
    gr.HTML("""
    <div id="masthead">
      <div class="dateline"><span>Vol. VI · Fine-Tuned Edition</span><span>Qwen3-1.7B · QLoRA</span></div>
      <h1 class="paper-name">The Verifier</h1>
      <div class="kicker">A cited, evidence-bounded newsroom copy desk &mdash;
        it checks AP style and verifies every claim against a real source, or says so plainly.</div>
      <hr class="ruleline"/>
    </div>
    """)
    inp = gr.Textbox(label="Copy to check", lines=3,
                     placeholder="Enrollment rose 12 percent, according to a district report.")
    with gr.Row():
        retrieve = gr.Checkbox(value=False,
                               label="Retrieve sources & verify claims (slower; off = AP style only)")
        k = gr.Slider(2, 6, value=4, step=1, label="Sources to pull")
    btn = gr.Button("Send to the copy desk", variant="primary", elem_id="analyze-btn")
    out = gr.Markdown(elem_id="result")
    src = gr.Markdown(elem_id="sources")
    btn.click(analyze, [inp, retrieve, k], [out, src])
    gr.Examples([
        ["The meeting has 5 members and starts at 3:00 PM on December 25.", False, 4],
        ["Enrollment rose 12 percent this year.", False, 4],
        ["The James Webb Space Telescope launched on December 25, 2021 from French Guiana.", True, 4],
    ], [inp, retrieve, k], label="Try a headline")


if __name__ == "__main__":
    print("Loading UI... model loads on first analyze (~1 min), then stays warm.")
    # SHARE=1 -> public gradio.live URL (Colab / remote). Else local browser.
    if os.environ.get("SHARE") == "1":
        app.launch(share=True, server_name="0.0.0.0")
    else:
        app.launch(inbrowser=True)
