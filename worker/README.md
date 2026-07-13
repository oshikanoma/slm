# Whole-web search for the live site — free, no credit card

The in-browser app can't hold a search API key (a static page is public) or fetch
arbitrary sites (CORS). This tiny **Cloudflare Worker** fixes both: it holds the
key server-side, runs the web search, and hands clean sources back to the page.

Cost: **$0**. Cards required: **none**. Both signups are card-free.

---

## Part A — What you do (two free signups, ~10 min)

### 1. Tavily (the web-search engine) — free, no card
1. Go to <https://tavily.com> → **Sign up** (Google/GitHub login is fine).
2. On the dashboard, copy your **API key** — it looks like `tvly-xxxxxxxx`.
3. Free tier = 1,000 searches/month, no card.

### 2. Cloudflare (hosts the Worker) — free, no card
1. Go to <https://dash.cloudflare.com/sign-up> → create an account (email + password).
2. Verify your email. That's all — you do **not** need to add a domain or a card.

### 3. Hand off to Claude
Tell Claude: **"Cloudflare + Tavily are set up"** and paste your **Tavily key**
(`tvly-...`). Claude does Part B. You'll also run ONE login command yourself
(browser auth — Claude can't click it):

```bash
# In this repo, run this yourself — it opens a browser to log into Cloudflare:
cd worker && npx wrangler login
```
When it says "Successfully logged in", tell Claude.

---

## Part B — What Claude does (automated)

```bash
cd worker
npx wrangler secret put TAVILY_API_KEY   # Claude pipes in your key
npx wrangler deploy                      # deploys the Worker, prints its URL
```

The deploy prints a URL like `https://verifier-search.<you>.workers.dev`.
Claude then puts that into `webgpu_site/config.js` (`WORKER_URL`) and redeploys the
site. From then on, pasting a sentence does a **whole-web** lookup.

---

## How it works / safety
- The browser page calls only your Worker, never Tavily directly — so the key
  stays secret (set via `wrangler secret`, never in the repo or the page).
- The Worker returns `{query, bundle:[{url,text}]}`; the page feeds those to the
  model exactly like pasted sources.
- If the Worker is ever down or `WORKER_URL` is blank, the app automatically
  falls back to free in-browser Wikipedia search — it never hard-fails.
