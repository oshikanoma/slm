// Browser-side source retrieval — the "paste a sentence, it looks up sources"
// behavior. Two backends, best first:
//   1. WHOLE-WEB via a Cloudflare Worker relay (config WORKER_URL) that holds the
//      Tavily key server-side and returns {url,text} sources — real open-web
//      search (news, any site), which a static page can't do on its own.
//   2. FALLBACK: free in-browser Wikipedia search (CORS-friendly, no key) when
//      no Worker is configured. Same as the original app with no Tavily key.

import { WORKER_URL } from "./config.js";

const WS = /\s+/g;
const QUOTED = /["“](.+?)["”]/;
const STOP = new Set(("the a an and or but of to in on for with at by from as is are " +
  "was were said that this his her their they it its he she we you will").split(" "));

// Distill a passage into a good search query — port of passage_to_query().
export function passageToQuery(passage, maxTerms = 12) {
  passage = (passage || "").replace(WS, " ").trim();
  const q = QUOTED.exec(passage);
  if (q && q[1].split(" ").length >= 3) return q[1].slice(0, 200);
  const words = passage.match(/[A-Za-z0-9$%.]+/g) || [];
  const salient = words.filter(
    (w) => !STOP.has(w.toLowerCase()) && (w.length > 2 || /\d/.test(w)));
  return salient.slice(0, maxTerms).join(" ") || passage.slice(0, 120);
}

const API = "https://en.wikipedia.org/w/api.php";
const qs = (params) =>
  API + "?" + new URLSearchParams({ ...params, format: "json", origin: "*" });

// Search -> ordered list of article titles.
async function searchTitles(query, k) {
  const r = await fetch(qs({ action: "query", list: "search", srsearch: query, srlimit: k }));
  if (!r.ok) throw new Error(`Wikipedia search HTTP ${r.status}`);
  const d = await r.json();
  return (d.query?.search || []).map((h) => h.title);
}

// Fetch the readable intro text for a title.
async function fetchExtract(title, maxChars) {
  const r = await fetch(qs({
    action: "query", prop: "extracts", titles: title,
    explaintext: "1", exintro: "1", redirects: "1",
  }));
  if (!r.ok) return null;
  const d = await r.json();
  const pages = d.query?.pages || {};
  const page = Object.values(pages)[0];
  if (!page || !page.extract) return null;
  const text = page.extract.replace(WS, " ").trim().slice(0, maxChars);
  const url = "https://en.wikipedia.org/wiki/" + encodeURIComponent(page.title.replace(/ /g, "_"));
  return { url, text, title: page.title };
}

// Whole-web search via the Cloudflare Worker relay.
// Tavily is a SEMANTIC engine — send the passage nearly verbatim (a natural
// sentence) rather than keyword-stripped, which scores far better.
async function searchWeb(passage, k) {
  const query = (passage || "").replace(WS, " ").trim().slice(0, 380);
  const r = await fetch(WORKER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });
  if (!r.ok) throw new Error(`search relay HTTP ${r.status}`);
  const d = await r.json();
  if (d.error) throw new Error(d.error);
  return { query: d.query || query, bundle: d.bundle || [], backend: "web" };
}

// Free Wikipedia fallback (no key).
async function searchWikipedia(passage, k, perSourceChars) {
  const query = passageToQuery(passage);
  const titles = await searchTitles(query, k + 2);
  const bundle = [];
  for (const t of titles) {
    if (bundle.length >= k) break;
    try {
      const src = await fetchExtract(t, perSourceChars);
      if (src && src.text.length >= 120) bundle.push(src);
    } catch { /* skip a source that fails to fetch */ }
  }
  return { query, bundle, backend: "wikipedia" };
}

// passage -> {query, bundle:[{url,text}], backend}. Prefers whole-web (Worker),
// falls back to Wikipedia if no Worker is set or the relay errors.
export async function buildBundle(passage, k = 4, perSourceChars = 1800) {
  if (WORKER_URL) {
    try { return await searchWeb(passage, k); }
    catch (e) { console.warn("web search relay failed, falling back to Wikipedia:", e); }
  }
  return await searchWikipedia(passage, k, perSourceChars);
}
