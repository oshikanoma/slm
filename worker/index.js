// Cloudflare Worker: whole-web source retrieval for The Verifier's browser app.
//
// The browser page can't hold a search API key (visible to everyone) or fetch
// arbitrary sites (CORS). This Worker does both server-side: it holds the secret
// TAVILY_API_KEY, runs a web search, and returns clean {url, text} sources with
// permissive CORS so the static page can call it.
//
// Secret is set out-of-band:  wrangler secret put TAVILY_API_KEY
// Deployed URL then goes into webgpu_site/config.js (WORKER_URL).

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
    if (request.method !== "POST") return json({ error: "POST only" }, 405);
    if (!env.TAVILY_API_KEY) return json({ error: "server missing TAVILY_API_KEY" }, 500);

    let body;
    try { body = await request.json(); }
    catch { return json({ error: "invalid JSON body" }, 400); }

    const query = (body.query || "").toString().slice(0, 400).trim();
    const k = Math.min(Math.max(parseInt(body.k, 10) || 4, 1), 6);
    if (!query) return json({ error: "empty query" }, 400);

    // Tavily search — returns web results WITH extracted page content, so we
    // don't need a separate fetch/parse step. Free tier, no card.
    let data;
    try {
      const r = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.TAVILY_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          max_results: k,
          search_depth: "advanced", // markedly better relevance than "basic"
          topic: "general",
          include_raw_content: false, // the shorter `content` snippet is enough + faster
        }),
      });
      if (!r.ok) {
        const detail = await r.text().catch(() => "");
        return json({ error: `search backend HTTP ${r.status}`, detail: detail.slice(0, 300) }, 502);
      }
      data = await r.json();
    } catch (e) {
      return json({ error: `search request failed: ${e.message || e}` }, 502);
    }

    const bundle = [];
    for (const item of data.results || []) {
      const text = (item.content || "").replace(/\s+/g, " ").trim().slice(0, 1800);
      if (item.url && text.length >= 80) bundle.push({ url: item.url, text });
      if (bundle.length >= k) break;
    }
    return json({ query, bundle });
  },
};
