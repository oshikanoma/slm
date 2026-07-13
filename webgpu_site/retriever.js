// Browser-side source retrieval — the "paste a sentence, it looks up sources"
// behavior, running entirely client-side. Mirrors retriever.py's FREE default
// (Wikipedia), which is the only search backend that permits browser (CORS)
// access without a secret API key. passage -> query -> titles -> page text.
//
// (Open-web search — Tavily/Brave/Serper — needs a secret key a static site
// can't hold, so it isn't available here; same as the original app with no key.)

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

// passage -> [{url, text}] bundle the model verifies against.
// Over-fetches titles then keeps the first k with usable text.
export async function buildBundle(passage, k = 4, perSourceChars = 1800) {
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
  return { query, bundle };
}
