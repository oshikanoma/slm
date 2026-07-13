// The Verifier — 100% in-browser (WebGPU). No server: YOUR fine-tuned model runs
// on the visitor's own GPU via Transformers.js. Weights (ONNX q4) stream from the
// HF Hub and cache locally after the first load.
import { AutoTokenizer, AutoModelForCausalLM } from
  "https://esm.run/@huggingface/transformers@4.2.0";
import { SYSTEM_PROMPT, MODEL_REPO, MODEL_DTYPE, MODEL_DEVICE } from "./config.js";

const $ = (id) => document.getElementById(id);
let tokenizer = null, model = null, loading = false;

// ---- WebGPU capability check ----------------------------------------------
function webgpuSupported() { return typeof navigator !== "undefined" && !!navigator.gpu; }

// ---- Lazy model load with progress ----------------------------------------
async function ensureModel() {
  if (model || loading) return model;
  loading = true;
  setStatus("Loading your fine-tuned model into the browser… first time downloads " +
            "~2&nbsp;GB (then it's cached and instant). Runs entirely on your device.");
  try {
    const progress = (p) => {
      if (p.status === "progress" && p.file && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        setStatus(`Downloading ${escapeHtml(p.file)} — ${pct}% (cached after first load)`);
      } else if (p.status === "ready") {
        setStatus("✅ Model ready — running on your GPU. Nothing you type leaves your device.");
      }
    };
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_REPO, { progress_callback: progress });
    model = await AutoModelForCausalLM.from_pretrained(MODEL_REPO, {
      dtype: MODEL_DTYPE, device: MODEL_DEVICE, progress_callback: progress,
    });
    setStatus("✅ Model ready — running on your GPU. Nothing you type leaves your device.");
  } catch (e) {
    setStatus("");
    loading = false;
    throw e;
  }
  loading = false;
  return model;
}

// ---- Prompt assembly: mirrors eval.py render_user_prompt exactly -----------
function renderUserPrompt(passage, sources) {
  const src = (sources || []).map(
    (s, i) => `[SOURCE ${i + 1}] url: ${s.url || ""}\n${s.text || ""}`
  ).join("\n\n");
  return `PASSAGE:\n${passage}\n\nSOURCES:\n${src}`;
}

async function verifyClaims(passage, sources) {
  await ensureModel();
  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: renderUserPrompt(passage, sources) },
  ];
  // Apply the model's own chat template (enable_thinking off — same as serving).
  // return_dict (default) gives { input_ids, attention_mask } tensors.
  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true, return_dict: true, enable_thinking: false,
  });
  const outputs = await model.generate({
    ...inputs, max_new_tokens: 640, do_sample: false,
  });
  // Keep only newly generated tokens (everything past the prompt length).
  const promptLen = inputs.input_ids.dims.at(-1);
  const gen = outputs.slice(null, [promptLen, null]);
  const text = tokenizer.batch_decode(gen, { skip_special_tokens: true })[0] ?? "";
  try { return JSON.parse(text); }
  catch { return { _raw: text }; }
}

// ---- Link handling ---------------------------------------------------------
// Extract URLs from the passage. NOTE: a browser page cannot fetch arbitrary
// other sites (CORS) or reliably probe liveness, so link *content* verification
// needs sources pasted alongside. We surface the links we found and let the
// model judge them against any SOURCES the user provides.
const MD_LINK = /\[[^\]]*\]\((https?:\/\/(?:[^\s()]|\([^\s()]*\))+)\)/g;
const BARE_URL = /https?:\/\/[^\s<>"']+/g;
const TRAIL = /[.,;:!?)\]}"']+$/;

function extractLinks(text) {
  const out = [], seen = new Set(), mdSpans = [];
  let m;
  MD_LINK.lastIndex = 0;
  while ((m = MD_LINK.exec(text))) { mdSpans.push([m.index, m.index + m[0].length]); push(m[1]); }
  BARE_URL.lastIndex = 0;
  while ((m = BARE_URL.exec(text))) {
    const at = m.index;
    if (mdSpans.some(([a, b]) => a <= at && at < b)) continue;
    push(m[0].replace(TRAIL, ""));
  }
  function push(u) { if (!seen.has(u)) { seen.add(u); out.push(u); } }
  return out;
}

// ---- Rendering -------------------------------------------------------------
const ICONS = { supported: "✅ SUPPORTED", unsupported: "❌ UNSUPPORTED", misleading: "⚠️ MISLEADING" };

function renderVerdicts(obj) {
  if (obj._raw !== undefined)
    return `<p><em>(model returned non-JSON)</em></p><pre>${escapeHtml(obj._raw.slice(0, 800))}</pre>`;
  const vs = (obj.verdicts || []).filter((v) => v.verdict !== "ap_flag");
  if (!vs.length) return `<h3>Claim verdicts</h3><p>✅ Nothing flagged against the provided sources.</p>`;
  const items = vs.map((v) => {
    let b = `<strong>${ICONS[v.verdict] || escapeHtml(v.verdict)}</strong> — ${escapeHtml(v.span || "")}`;
    if (v.source_url) {
      b += `<div class="sub">• Cited: ${linkify(v.source_url)}</div>`;
      b += `<div class="sub">• Backing: “${escapeHtml(v.evidence_span || "")}”</div>`;
    } else if (v.checked_source_url) {
      b += `<div class="sub">• Checked (no support found): ${linkify(v.checked_source_url)}</div>`;
    }
    if (v.explanation) b += `<div class="sub">• ${escapeHtml(v.explanation)}</div>`;
    return `<li>${b}</li>`;
  }).join("");
  return `<h3>Claim verdicts</h3><ul class="verdicts">${items}</ul>`;
}

function renderAP(hits) {
  if (!hits.length) return `<p>✅ <strong>AP Style:</strong> no issues found.</p>`;
  const items = hits.map((h) =>
    `<li><strong>${escapeHtml(h.span)}</strong> — ${escapeHtml(h.rule)}<div class="sub">→ suggested: ${escapeHtml(h.suggestion)}</div></li>`
  ).join("");
  return `<h3>⚑ AP Style (instant, in-browser)</h3><ul class="verdicts">${items}</ul>`;
}

function renderLinks(links) {
  if (!links.length) return "";
  const items = links.map((u) => `<li>🔗 ${linkify(u)}</li>`).join("");
  return `<h3>🔗 Links found in your copy</h3><ul class="verdicts">${items}</ul>` +
    `<p class="note">A browser page can't fetch other sites to auto-verify a link's ` +
    `content — paste the linked page's text into “Sources” and the model checks the claim against it.</p>`;
}

// ---- Main handler ----------------------------------------------------------
async function onAnalyze() {
  const passage = $("passage").value.trim();
  const sourcesRaw = $("sources").value.trim();
  if (!passage) { $("result").innerHTML = "<p>Enter a passage.</p>"; return; }

  // AP + links are instant and offline.
  const ap = window.apCheck(passage);
  const links = extractLinks(passage);
  let html = renderLinks(links) + renderAP(ap);

  // Claim verification needs sources + the model.
  const sources = parseSources(sourcesRaw);
  if (!sources.length) {
    $("result").innerHTML =
      `<p class="note">Add one or more sources below to verify the claim against them.</p>` + html;
    return;
  }
  $("analyze").disabled = true;
  $("result").innerHTML = `<p class="note">Verifying against ${sources.length} source(s)…</p>` + html;
  try {
    const obj = await verifyClaims(passage, sources);
    $("result").innerHTML = renderVerdicts(obj) + renderLinks(links) + renderAP(ap);
  } catch (e) {
    $("result").innerHTML = `<p class="err">Error: ${escapeHtml(String(e && e.message || e))}</p>` + html;
  } finally {
    $("analyze").disabled = false;
  }
}

// Sources box: one source per block, "URL <newline> text", blocks separated by blank lines.
function parseSources(raw) {
  if (!raw) return [];
  return raw.split(/\n\s*\n/).map((block) => {
    const lines = block.trim().split("\n");
    const first = lines[0].trim();
    if (/^https?:\/\//.test(first)) return { url: first, text: lines.slice(1).join("\n").trim() };
    return { url: "", text: block.trim() };
  }).filter((s) => s.text);
}

// ---- utils -----------------------------------------------------------------
function setStatus(html) { $("status").innerHTML = html; }
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function linkify(u) { const e = escapeHtml(u); return `<a href="${e}" target="_blank" rel="noopener">${e}</a>`; }

// ---- boot ------------------------------------------------------------------
function boot() {
  $("analyze").addEventListener("click", onAnalyze);
  if (!webgpuSupported()) {
    setStatus(`⚠️ This browser doesn't expose WebGPU, so the model can't run here. ` +
      `AP style + link detection still work. For live claim verification use Chrome or Edge (desktop). ` +
      `The model also lives at <a href="https://huggingface.co/tiffuhknee/qwen3-1.7b-newsroom-verifier" target="_blank" rel="noopener">HF Hub</a>.`);
    $("analyze").textContent = "Check AP style & links";
  }
}
document.addEventListener("DOMContentLoaded", boot);
