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
import re

import gradio as gr

from eval import Scenario, build_messages
from retriever import build_bundle, bundle_from_links, extract_links, _active_backend
from ap_rules import ap_check  # deterministic AP checker (code, not model)

BASE = "Qwen/Qwen3-1.7B"
# Adapter is loaded from the HF Hub on the Space. Override with env HF_ADAPTER_REPO.
ADAPTER = os.environ.get("HF_ADAPTER_REPO", "REPLACE_WITH_YOUR_HF_USERNAME/qwen3-1.7b-newsroom-verifier")
_MODEL = {"m": None, "tok": None, "dev": None}


def _load():
    if _MODEL["m"] is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    # Spaces free tier is CPU; use GPU/MPS automatically when present.
    if torch.cuda.is_available():
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


def _fmt_links(statuses: list[dict]) -> str:
    """Liveness report for links found IN the passage (before content verification)."""
    if not statuses:
        return ""
    icons = {"alive": "🔗 ALIVE", "redirect": "↪️ REDIRECT", "dead": "🚫 DEAD"}
    lines = ["### 🔗 Links in your copy"]
    for s in statuses:
        head = f"- **{icons.get(s['status'], s['status'])}** — {s['url']}"
        if s.get("note"):
            head += f"\n  → {s['note']}"
        lines.append(head)
    return "\n".join(lines)


def analyze_core(passage: str, do_retrieve: bool, k: int) -> tuple[str, str]:
    """Run the full copy-desk pass on ONE passage. Returns (report_md, sources_md).

    Layers, cheapest first:
      1) AP style       — deterministic, instant, always.
      2) Link liveness  — probe every URL in the copy (dead/redirect/alive).
      3) Claim + link verification — the SLM, against BOTH linked pages and
         (optionally) retrieved sources.
    """
    if not passage.strip():
        return "Enter a passage.", ""
    # 1) AP: always, instant, complete.
    ap_md = _fmt_ap(ap_check(passage))

    # 2) Links already in the copy: liveness + fetch live pages as sources.
    link_sources, link_statuses = ([], [])
    if extract_links(passage):
        link_sources, link_statuses = bundle_from_links(passage)
    links_md = _fmt_links(link_statuses)

    # Verification runs if the author included links OR asked for web retrieval.
    if not do_retrieve and not link_sources:
        tail = "\n\n_(claim verification skipped — add a link, or turn on " \
               "'retrieve & verify', to check claims against real sources)_"
        report = "\n\n".join(x for x in (links_md, ap_md) if x) + tail
        return report, "_(retrieval off)_"

    # 3) Assemble sources: linked pages first, then retrieved (deduped by url).
    sources = list(link_sources)
    src_note = ""
    if do_retrieve:
        backend, _ = _active_backend()
        have = {s["url"] for s in sources}
        for s in build_bundle(passage, k=int(k)):
            if s["url"] not in have:
                sources.append(s); have.add(s["url"])
        if not sources:
            src_note = f"_(no sources retrieved; backend: {backend})_"

    src_md = ("\n".join(f"- [{s['url']}]({s['url']})" for s in sources)
              if sources else (src_note or "_(no sources)_"))
    if not sources:
        report = "\n\n".join(x for x in (links_md, ap_md) if x) \
            + "\n\n_(no sources — can't verify claims)_"
        return report, src_md

    obj = _run(passage, sources, 512)
    claim_md = _fmt(obj)
    report = "\n\n".join(x for x in (claim_md, links_md, ap_md) if x)
    return report, src_md


def analyze(passage: str, do_retrieve: bool, k: int):
    """Single-passage entry point for the paste box."""
    return analyze_core(passage, do_retrieve, k)


# ------------------------------------------------------------------------------
# Whole-document mode: read a file, chunk it, verify every chunk, roll up a report
# ------------------------------------------------------------------------------

def _read_document(path: str) -> str:
    """Best-effort plain-text extraction from .txt/.md/.pdf/.docx."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader  # older name; either works
            except Exception:
                raise RuntimeError("PDF support needs 'pypdf' (pip install pypdf). "
                                   "Or paste the text, or upload .txt/.md.")
        return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages)
    if ext == ".docx":
        try:
            import docx  # python-docx
        except Exception:
            raise RuntimeError("DOCX support needs 'python-docx' (pip install python-docx). "
                               "Or paste the text, or upload .txt/.md.")
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    # .txt / .md / anything else: read as UTF-8 text.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def chunk_document(text: str, max_chars: int = 600) -> list[str]:
    """Split into verifiable chunks on blank lines, packing short paragraphs
    together and hard-splitting any paragraph longer than max_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > max_chars and buf:  # keep document order: flush first
            chunks.append(buf); buf = ""
        while len(p) > max_chars:  # a single huge paragraph -> hard slices
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= max_chars:
            buf += "\n\n" + p
        else:
            chunks.append(buf); buf = p
    if buf:
        chunks.append(buf)
    return chunks


def analyze_document(file_obj, do_retrieve: bool, k: int):
    """Run the copy-desk pass across an uploaded document, chunk by chunk."""
    if file_obj is None:
        return "Upload a document (.txt, .md, .pdf, or .docx).", ""
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    try:
        text = _read_document(path)
    except Exception as e:
        return f"Could not read the document: {e}", ""
    chunks = chunk_document(text)
    if not chunks:
        return "The document appears to be empty.", ""

    sections, all_sources, risk = [], [], 0
    for i, ch in enumerate(chunks, 1):
        report, src_md = analyze_core(ch, do_retrieve, k)
        # Count libel-risk items = claims the model could not back with a source.
        risk += report.count("❌ UNSUPPORTED") + report.count("⚠️ MISLEADING")
        preview = ch if len(ch) <= 160 else ch[:157] + "…"
        sections.append(f"---\n\n#### § {i}\n> {preview}\n\n{report}")
        if src_md and src_md not in all_sources and not src_md.startswith("_("):
            all_sources.append(src_md)

    header = (f"## Document report — {len(chunks)} section(s)\n\n"
              f"**Potential legal-risk items** (unsupported or misleading factual "
              f"claims): **{risk}**\n")
    if risk:
        header += ("\n> ⚠️ Each item below marked UNSUPPORTED or MISLEADING is an "
                   "assertion the desk could not tie to a real source — the kind of "
                   "unbacked factual claim that carries defamation/libel exposure. "
                   "Add a citation or soften it.\n")
    body = "\n\n".join(sections)
    sources_md = "\n\n".join(all_sources) if all_sources else "_(no sources)_"
    return header + "\n" + body, sources_md


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
    with gr.Tabs():
        with gr.Tab("Paste copy"):
            inp = gr.Textbox(
                label="Copy to check", lines=4,
                placeholder="Paste a sentence or paragraph. Include links "
                            "(e.g. https://… ) and the desk will verify them too.")
            with gr.Row():
                retrieve = gr.Checkbox(
                    value=False,
                    label="Retrieve sources & verify claims (slower; off = AP style + links only)")
                k = gr.Slider(2, 6, value=4, step=1, label="Sources to pull")
            btn = gr.Button("Send to the copy desk", variant="primary", elem_id="analyze-btn")
            out = gr.Markdown(elem_id="result")
            src = gr.Markdown(elem_id="sources")
            btn.click(analyze, [inp, retrieve, k], [out, src])
            gr.Examples([
                ["The meeting has 5 members and starts at 3:00 PM on December 25.", False, 4],
                ["Enrollment rose 12 percent this year.", False, 4],
                ["The James Webb Space Telescope launched on December 25, 2021 from French Guiana.", True, 4],
                ["The report (https://en.wikipedia.org/wiki/James_Webb_Space_Telescope) "
                 "says the telescope launched in 2021.", False, 4],
            ], [inp, retrieve, k], label="Try a headline")

        with gr.Tab("Whole document"):
            gr.Markdown("Upload a **.txt, .md, .pdf, or .docx**. The desk splits it "
                        "into sections and checks every claim and link throughout — "
                        "flagging unbacked factual assertions (the libel-risk ones).")
            doc = gr.File(label="Document", file_types=[".txt", ".md", ".pdf", ".docx"])
            with gr.Row():
                doc_retrieve = gr.Checkbox(
                    value=False,
                    label="Retrieve sources & verify claims (much slower on long docs)")
                doc_k = gr.Slider(2, 6, value=4, step=1, label="Sources per section")
            doc_btn = gr.Button("Check the whole document", variant="primary", elem_id="analyze-btn")
            doc_out = gr.Markdown(elem_id="result")
            doc_src = gr.Markdown(elem_id="sources")
            doc_btn.click(analyze_document, [doc, doc_retrieve, doc_k], [doc_out, doc_src])


if __name__ == "__main__":
    # HF Spaces: bind 0.0.0.0:7860, no browser auto-open.
    app.launch(server_name="0.0.0.0", server_port=7860)
