# AI & Ecommerce News Agent (v1)

- **Status**: In Progress
- **Created**: 2026-11-06
- **Last Updated**: 2026-11-07

## Goal
Deliver an automated, personalised digest of AI, agent and data-science news, filtered for an ecommerce and fashion data scientist at THE ICONIC.

- **Daily digest** (Mon–Fri, ~7–8am Sydney) to `anhnguyen.workmail@gmail.com`.
- **Weekly trends email** (Saturday ~8am Sydney).
- **👍/👎 feedback** links that adjust ranking weights over time.
- **$0 running cost**: Python on GitHub Actions, Gemini free tier, Gmail SMTP/IMAP.
- **Every summary links back to its source** (2026-11-07): news, papers, repos, blogs, models, posts, and every weekly trend, theme and idea.
- **Deep read** (2026-11-07): read the full text of the top papers, blogs and repos and summarise their key ideas in more detail.
- **Papers: rolling 30 days, industry first** (2026-11-07): for ecommerce DS papers, prioritise large-company research (Alibaba/Taobao, Amazon, Google, Netflix, Meta, …) and highly recommended papers (HF upvotes, citations, GitHub stars).

## Context & Constraints
- The repo is new: it holds only `.github/agents/` and the `.github/skills/crawl4ai` skill.
- Everything must be free: X/Twitter is excluded (paid API), and a personal Gemini key is used instead of company GCP.
- Runs as a personal project on GitHub Actions, not on the company n8n account.
- Some sources are unofficial or unreliable (the Good AI List API is undocumented; Reddit may block GitHub's servers), so each source must fail on its own without stopping the run.
- Gemini free-tier rate limits apply, so items are batched (about 5 calls a day).

## Approach

### Daily flow (Mon–Fri Sydney; cron `0 21 * * 0-4` UTC)
```
 1. Ingest feedback   Gmail IMAP: unread "[news-agent-feedback]" mails → data/feedback.json → update weights
 2. Collect           all sources in parallel, fail-soft  (window: 26h; Monday: 74h)
 3. Dedupe            data/seen.json (URL / arXiv id / repo full name)
 4. Prefilter         keyword × topic weight × source weight → ~80 candidates, per-source caps
 5. Rank (Gemini)     batches of 20 → JSON {id, score 0–10, section, topics[], tldr, why_iconic}
 6. Select            score ≥ 6, section caps, Top 5 across all
 7. Persist           data/items/YYYY-MM-DD.json (all scored items), update seen.json
 8. Render            HTML email + digests/daily/YYYY-MM-DD.md
 9. Send              Gmail SMTP (SSL 465)
10. Commit            digests/, data/ back to repo
```

### Weekly flow (Saturday Sydney; cron `0 21 * * 5` UTC)
1. Load `data/items/` for the last 7 days.
2. One Gemini call: themes and trends, notable shifts, "what to try at THE ICONIC".
3. Sections: trends summary · top 10 of the week · model releases · repos of the week · feedback stats.
4. Render HTML + `digests/weekly/YYYY-Www.md`, send, commit.

### Feedback loop
- Each item has two links: `mailto:<GMAIL_USER>?subject=[news-agent-feedback] up <item_id>` and `... down <item_id>`. Tapping one opens a pre-filled email; the user presses send.
- The daily run reads unread feedback mails over IMAP, then marks them read and archives them.
- It looks up the item in `data/items/` to get its `topics[]` and `source`.
- Weight update per topic and per source: `w ← clamp(w × (1 ± 0.05), 0.5×base, 2×base)`.
- Stored in `data/feedback.json` (vote log + current multipliers); used by the prefilter and passed to Gemini as context.

### Sources (`config/sources.yaml`)
| Section | Source | Method |
|---|---|---|
| Papers | arXiv cs.IR, cs.LG, cs.CL, cs.AI, cs.CV, stat.ML | arXiv API |
| Papers | HF Daily Papers | JSON API |
| Models | HF trending models | JSON API |
| Models / Industry | OpenAI, Anthropic, Google, DeepMind, Meta AI, Microsoft Research, NVIDIA, IBM Research, AWS ML, Amazon Science | RSS |
| Industry | TechCrunch AI, The Verge AI, VentureBeat AI | RSS |
| Industry | Import AI, Latent Space, The Batch | RSS |
| Eng blogs | Zalando, Shopify, Etsy, Instacart, Pinterest, Airbnb, Netflix, Spotify, Booking, DoorDash | RSS |
| Repos | Good AI List `https://goodailist.com/api/repos?sort=star_1d&limit=100` and `?is_new=1` | JSON |
| Community | Hacker News (Algolia; ≥50 points) | JSON API |
| Community | Reddit r/MachineLearning, r/LocalLLaMA, r/datascience, r/recommendersystems | `.rss` |
| Community | Bluesky public search | JSON API |

Good AI List rules: keep a repo if `is_new`, or `star_1d ≥ 300`, or `star_7d_pct ≥ 50`. Report a repo again only if its stars reach 2× the count when it was first reported.

### Topic profile (`config/interests.yaml`)
| Base weight | Topics |
|---|---|
| 3 | RecSys, search & ranking, personalisation, agentic commerce, LLM agents in production |
| 3 | Fashion: size & fit, visual search, outfit/complementary recs, return prediction |
| 2 | Forecasting, pricing/markdown, causal inference/uplift, experimentation, CLV/churn |
| 2 | GenAI for catalogue, multimodal, LLM/agent evaluation |
| 2 | New model releases, AI industry news |
| 1 | General ML/DS tooling, MLOps |

### Daily digest layout
1. 🔥 Top 5 must-reads
2. 🛍 Ecommerce & RecSys (≤6)
3. 📄 Papers (≤6)
4. 🤖 New models & releases (≤5)
5. 📰 AI industry news (≤5)
6. 🏢 Engineering blogs (≤5)
7. ⭐ Rising GitHub repos (≤8; stars today / this week, "worth trying?")
8. 💬 Community buzz (≤5)
9. Footer: run stats, failed sources, archive link

Each item: title (link) · source · score · TL;DR · why it matters · 👍 👎

### Repo layout
```
news_update/
├── plans/
├── pyproject.toml          # httpx, feedparser, pydantic, pyyaml, jinja2, google-genai, tenacity, pytest
├── .env.example            # GOOGLE_API_KEY, GMAIL_USER, GMAIL_APP_PASSWORD, DIGEST_TO
├── config/{sources.yaml, interests.yaml}
├── src/news_agent/
│   ├── main.py             # CLI: daily | weekly  [--since] [--dry-run] [--no-llm] [--force]
│   ├── models.py · store.py · prefilter.py · ranker.py
│   ├── feedback.py         # IMAP read + weight update
│   ├── weekly.py           # aggregation + trends
│   ├── digest.py · emailer.py
│   └── collectors/{arxiv, rss, hf, hackernews, reddit, bluesky, goodailist}.py
├── templates/{daily.html.j2, daily.md.j2, weekly.html.j2, weekly.md.j2}
├── data/{seen.json, feedback.json, items/}
├── digests/{daily/, weekly/}
├── tests/
└── .github/workflows/{daily.yml, weekly.yml}
```

### Key decisions and alternatives
| Decision | Choice | Alternatives considered |
|---|---|---|
| Scheduler | GitHub Actions (free, versioned, personal) | Company n8n: not personal, harder to version and test |
| LLM | `google-genai`, `gemini-2.5-flash` (configurable) | Vertex AI: supported via env vars only (`GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`), but billed |
| State | JSON files committed to the repo | Hosted DB: unnecessary cost |
| Email | Gmail SMTP send + IMAP feedback (one app password) | Slack: not requested |
| Feedback | `mailto` links | Web endpoint: needs a server |
| Weekends | Skipped; Monday covers 74h | Daily: arXiv is quiet at weekends |
| X/Twitter | Excluded; Bluesky + newsletters as proxy | Paid X API (~$200/month) |
| Failures | Each source fails on its own; with no LLM, send a keyword-only digest | — |

Sydney daylight saving moves delivery between ~7am (winter) and ~8am (summer).

## Implementation Steps
- [x] 0. Refine idea, choose sources, LLM, delivery, scheduler — `plans/`
- [x] 1. Scaffold project — `pyproject.toml`, `src/news_agent/`, `.gitignore`, `.env.example`
- [x] 2. Item model and dedupe store (prune >60 days, repo resurface rule) — `models.py`, `store.py`
- [x] 3. Source and interest configs — `config/sources.yaml`, `config/interests.yaml`
- [x] 4. Collectors: arxiv, rss, hf (papers + models), hackernews, reddit, bluesky, goodailist — `collectors/`
- [x] 5. Keyword prefilter with weights and per-source caps — `prefilter.py`
- [x] 6. Gemini ranker: batched JSON scoring, retries, API key / Vertex — `ranker.py`
- [x] 7. Scored item history — `data/items/YYYY-MM-DD.json`
- [x] 8. Daily digest templates with 👍/👎 links — `digest.py`, `templates/daily.*.j2`
- [x] 9. Email sender with `--dry-run` — `emailer.py`
- [x] 10. Feedback loop: IMAP read, weight multipliers — `feedback.py`, `data/feedback.json`
- [x] 11. Weekly trends email — `weekly.py`, `templates/weekly.*.j2`
- [x] 12. CLI (`daily` / `weekly`), logging, run stats — `main.py`
- [x] 13. Weekday guard and Monday 74h window — `main.py`
- [x] 14. GitHub Actions workflows with commit-back — `.github/workflows/daily.yml`, `weekly.yml`
- [x] 15. Tests — `tests/`
- [x] 16. README with setup and tuning guide — `README.md`
- [x] 18. v1.1 R1: link rows for every item, weekly citations with links — `digest.py`, `templates/*`, `weekly.py`
- [x] 19. v1.1 R2: HF weekly rolling papers, OpenAlex collector + affiliation enrichment — `collectors/hf.py`, `collectors/openalex.py`
- [x] 20. v1.1 R2: company detection, popularity boost, paper merge, show-only seen marking, LLM cache — `papers.py`, `store.py`, `prefilter.py`, `ranker.py`, `main.py`
- [x] 21. v1.1 R3: longer TL;DRs, deep read of top items (full-text fetch + Gemini/extractive summary), email "Deep read" block — `content.py`, `deepread.py`, `ranker.py`, `main.py`, `templates/*`
- [x] 22. v1.1 R4: "Today in brief", long-read HTML redesign, figure images, HTML-only output — `brief.py`, `content.py`, `digest.py`, `main.py`, `templates/*.html.j2`
- [x] 23. v1.1 R5: semantic ranking (fastembed) against THE ICONIC problems, Gemini summaries only, arXiv paper pool — `semantic.py`, `ranker.py`, `summarize.py`, `store.py`, `main.py`, `config/*`, workflow cache
- [~] 17. End-to-end verification (see Testing & Validation)
  - [x] 81 unit tests pass (`.venv/bin/python -m pytest -q`)
  - [x] Live `daily --dry-run --no-llm`: 994 items collected from 8 collectors, 0 failures, 80 candidates, ~26 picks, ~9s
  - [x] Stateful run in a temp root: seen.json/items/digest written; re-run → 1 new item (dedupe works); weekly digest built from history
  - [x] Code review pass; 3 issues fixed (see Decision Log)
  - [x] v1.1: 111 tests pass. Live dry run: 1,895 items, 140 papers given affiliations by OpenAlex, 23 picks, all 23 with a 🔗 Source row; industry papers (Alibaba, Amazon, Pinterest, Kuaishou, Walmart, Meituan…) lead the paper candidates
  - [x] v1.1 code review: 4 issues fixed (see Decision Log)
  - [x] R3: 124 tests pass. Live dry run read 8 items in full (7 from arXiv HTML, 1 ACM paper as abstract only because of a 403); email 58KB (under Gmail's 102KB clip)
  - [ ] R3: check the quality of Gemini deep reads once `GOOGLE_API_KEY` is set
  - [x] R4: 135 tests pass. Live dry run checked visually in a browser: brief → In depth (6/8 with paper figures) → More to read; 81–85KB HTML
  - [x] R4: Gemini-written "Today in brief" checked in the browser (cites DuoLM, CommerceGen-Studio, Pinterest, arXiv Taobao JBM paper)
  - [ ] R4: check the render in Gmail web + mobile
  - [x] R5: 126 tests pass (fake embedder, no model download). Live run with Gemini: about 2 min (was over 6), 1,444 items ranked, 31 picks (29 after the size guard), arXiv papers in Ecommerce & RecSys, no crashes
  - [ ] Real Gemini-ranked run (needs `GOOGLE_API_KEY`)
  - [ ] One real email (needs Gmail app password)
  - [ ] Feedback round trip (👍 → multiplier changes in `data/feedback.json`)
  - [x] Pushed to GitHub (2026-11-07, commit 13f2a92); both workflows active
  - [ ] Add repo secrets (GOOGLE_API_KEY, GMAIL_USER, GMAIL_APP_PASSWORD, DIGEST_TO), then a manual `workflow_dispatch` run of both workflows

## v1.1 Requirements (2026-11-07)

### R1. Link back for every summary
- Every daily item keeps its linked title, plus an explicit link row by kind:
  - paper: arXiv · PDF · HF · Code · DOI
  - repo: GitHub
  - community: post + discussion
  - news/blog: article
  - model: model card
- Items without a valid http(s) URL are dropped before ranking.
- Weekly: each trend bullet and "Try at THE ICONIC" idea cites item ids, rendered as numbered links. Theme items resolve against all of the week's items (not just top/models/repos).

### R2. Papers: rolling 30 days, industry and popularity first
- **Sources**:
  - HF Papers weekly lists (`?week=YYYY-Www`, the last ~5 ISO weeks): upvotes, organisation, GitHub stars.
  - OpenAlex search over the last 30 days for ecommerce DS queries (affiliations, citations).
  - OpenAlex DOI lookup to add affiliations to arXiv papers.
  - Semantic Scholar is skipped (429 without a key); an arXiv 30-day query is skipped (strict rate limit).
- **Merge**: the same paper from arXiv, HF and OpenAlex is merged on `arxiv:<id>` (union of affiliations, links and tags; max of metrics).
- **Industry detection**: `config/interests.yaml → papers.companies` (canonical name → aliases), matched against affiliations and the title/abstract (e.g. "deployed in Taobao").
- **Priority**: a prefilter bonus for industry and popularity; after the LLM, `score = min(10, llm_score + boost)`, and the reasons are shown on the card (🏢 Alibaba · ▲ 143 upvotes).
- **Rolling window**: papers are marked seen only when *shown*, so an unshown paper stays eligible for 30 days and can surface later as it gains upvotes or citations. LLM results are cached from `data/items/` (last 30 days), so re-considered papers cost no Gemini calls.

### R3. Longer summaries + deep read of top items (2026-11-07)
- **Longer TL;DRs:** every item gets a 2–4 sentence TL;DR (what it is, the key idea/method, the main result) and a 1–2 sentence "why it matters".
- **Deep read:** after selection, the top `deep_read.max_items` (default 8: Top 5 first, then the best papers/blogs/repos from the sections) are fetched in full and summarised:
  - paper: arXiv HTML full text (references removed); if no HTML exists, Gemini reads the PDF directly
  - blog/news: article text via trafilatura
  - repo: GitHub README
  - if fetching fails (403/paywall): the abstract or feed text, labelled "based on abstract"
- **Output per item** (one Gemini call each, JSON): a 4–6 sentence summary · 3–5 key ideas · results (with numbers) · 1–3 "how to apply at THE ICONIC" bullets · the text source used.
- **No LLM:** the extractive fallback picks key sentences (propose/introduce/results/outperform/deployed/%) from the full text; no "apply" bullets are invented.
- Deep summaries are stored in history and feed the weekly trends prompt.

### R4. "Today in brief" + long-read HTML redesign (2026-11-07)
- **Today in brief** at the top, before the details: a 2–3 sentence overview plus 4–6 highlight bullets, each citing and linking its items. Gemini writes it in one call; without Gemini, the highlights are the opening sentence of each deep-read item.
- **Long-read layout:** the deep-read items become full article blocks:
  - kicker (section · source · company), serif headline, figure image with caption, summary paragraph
  - numbered key ideas, a Results callout and a "How to apply at THE ICONIC" callout
  - source links and 👍/👎 buttons
  - everything else goes under "More to read" as compact cards
- No source-type label in the deep read (removed the "full text (arXiv HTML) · key sentences extracted" line).
- **Images (only raster, since Gmail can't show SVG):**
  - papers: a main-body arXiv figure (overview/framework/architecture/pipeline caption first), else the HF paper card
  - blog/news: `og:image`
  - repos: GitHub's preview card
- **HTML only:** no `.md` output. The archive is `digests/{daily,weekly}/*.html`; the email's plain-text part comes from the HTML.

### R5. Semantic ranking; Gemini only for summaries; arXiv papers first-class (2026-11-07)
- **Ranking without an LLM:** a local embedding model (fastembed `BAAI/bge-small-en-v1.5`, ONNX, about 67MB, no API, about 20s per run) scores every item against **THE ICONIC problem statements** in `config/interests.yaml`:
  - recsys/personalisation, search & query understanding, size & fit/returns, visual search
  - outfits/complete-the-look, catalogue GenAI, forecasting/pricing, marketing/CLV, experimentation/causal, shopping agents
  - plus general interests (model releases, AI industry, agents, MLOps) at lower weight
  - score = calibrated cosine relevance × topic weight × feedback multipliers, plus the paper industry/popularity boost
  - each item shows the THE ICONIC problem it matches (🎯)
- **Gemini only summarises:**
  - one batched call for the TL;DR and why-it-matters of the selected items
  - deep reads (one call each) and Today in brief
  - thinking is off for the batch summaries and small for deep reads, to keep runs fast
- **arXiv papers:** the daily arXiv feed (cs.IR/LG/CL/AI/CV, stat.ML/ME) competes on relevance, not popularity. A **paper pool** (`data/paper_pool.json`) keeps relevant unshown papers for 30 days so arXiv gets a real rolling window, like HF Papers.
- Removed: the keyword prefilter and Gemini batch ranking.

## Risks & Open Questions
| Risk | Mitigation |
|---|---|
| Good AI List API undocumented | Skip the section with a warning |
| Reddit blocks GitHub's servers | Skip it; Hacker News and Bluesky still cover community news |
| Gemini free-tier limits change | Batching, tenacity backoff, keyword-only fallback |
| Noisy early digests | Feedback loop; tune `interests.yaml` and the threshold after week 1 |
| Repo grows from `data/items` | Prune items >90 days, `seen.json` entries >60 days |

Open questions:
- Section caps and the score threshold (6): keep, or shorten to a Top 10 only?
- Repo visibility: public (unlimited Actions minutes, public archive) or private (~90 of 2,000 free minutes/month)?

User setup required:
1. Gemini API key: https://aistudio.google.com/apikey
2. Gmail: turn on 2-Step Verification, create an app password at https://myaccount.google.com/apppasswords, and enable IMAP.
3. GitHub secrets: `GOOGLE_API_KEY`, `GMAIL_USER`, `GMAIL_APP_PASSWORD`, `DIGEST_TO`.
4. Repo → Settings → Actions → Workflow permissions → **Read and write**.
5. Local: copy `.env.example` to `.env` (git-ignored).

## Testing & Validation
- Unit tests (pytest, no network): collector parsers on saved sample responses, dedupe, prefilter, ranker JSON parsing (mocked Gemini), feedback weight update, digest rendering.
- Local `daily --dry-run` with real sources; inspect the HTML output.
- One real daily email.
- Feedback round trip: send 👍 on an item, confirm weights change on the next run.
- `weekly --force` run.
- Manual GitHub Actions run (`workflow_dispatch`) for both workflows.

## Build Notes (2026-11-06)
- Built by an orchestrator plus 3 parallel sub-agents (collectors · ranking/feedback/weekly · digest/email), each owning separate files against the shared contract in `models.py`, `config.py`, `store.py`, `collectors/__init__.py`.
- Feed status after live verification: 26 RSS feeds working (openai/anthropic go through a community-maintained copy of the feed, `fallback_urls`); `venturebeat_ai` (429) and `the_batch` (404) disabled.
- arXiv: 24h grace on `since` (papers appear after the daily announcement), `max_results: 1000`.
- Reddit: one combined `/r/a+b+c/top/.rss` request (rate limits); `.json` returns 403. Bluesky: `api.bsky.app` (public.api returns 403 for search).

## Decision Log
- 2026-11-06: Scheduled pipeline (not a chat agent) delivering email digests; chat layer deferred to v2.
- 2026-11-06: Gemini via `google-genai` with a personal API key; Vertex supported through env vars. Avoid company GCP/n8n for a personal tool.
- 2026-11-06: GitHub Actions + Gmail SMTP chosen to keep running cost at $0. X/Twitter excluded (paid).
- 2026-11-06: Added Good AI List rising repos via its `/api/repos` JSON endpoint.
- 2026-11-06: Added new model releases and AI industry news topics.
- 2026-11-06: Pulled weekly trends email, 👍/👎 feedback and weekend skip into v1.
- 2026-11-06: Implementation done. Keyword fallback score uses a saturating curve `3 + 7·(1−e^(−p/8))` so scores spread out instead of all hitting 10.
- 2026-11-06: Replaced the generic `agent` keyword with specific phrases (it was swamping the prefilter).
- 2026-11-06: In-run dedupe also by canonical URL and normalised title; the primary source wins over community reposts.
- 2026-11-06: Review fixes: commit step runs `always()` so feedback votes taken from Gmail are never lost; keyword-fallback items are kept out of Top 5 in partly failed LLM runs; a missing email config fails the run (exit 1) instead of silently marking items as seen.
- 2026-11-07: v1.1 R1: every daily item gets a 🔗 Source row (primary link + labelled extras such as PDF/HF/Code/DOI/HN discussion); items without an http(s) URL are dropped; weekly trends, themes and ideas must cite item ids (uncited LLM statements are dropped) and render as links.
- 2026-11-07: v1.1 R2: papers come from HF Papers weekly lists (30 days, upvotes + org) and OpenAlex search (30 days, affiliations + citations; Zenodo/Figshare DOIs excluded), with OpenAlex enrichment for arXiv papers ≥2 days old. Semantic Scholar is skipped (429 without a key); an arXiv 30-day query is skipped (rate limit).
- 2026-11-07: Company detection uses affiliations, or a title/abstract sentence with a deployment cue (so "evaluated on Amazon reviews" doesn't count). Aliases are whole-word, and hyphens count as part of the word (so "meta-learning" ≠ Meta).
- 2026-11-07: Papers are marked seen only once shown; LLM judgements are cached from history (`split_cached`). Cached scores alone don't switch a run to Gemini mode. `first_seen`/`shown_on` stop rolling papers from repeating in every weekly digest.
- 2026-11-07: R3: one Gemini call per deep-read item (8/day, within the free tier); papers use arXiv HTML text with references removed (head 75% + tail 25% within 40k chars), or the PDF sent directly to Gemini when no HTML exists. The deep summary replaces the TL;DR on the card. Without an LLM the fallback is extract-only: strong idea cues first, sentences citing other work dropped, results must contain a quantity, repeat introductions of the same name removed, and no "apply" bullets.
- 2026-11-07: Added `trafilatura` for article/HTML text extraction. Added a Gmail clip guard (`render_fitted`) that drops the lowest-ranked section items if the HTML exceeds 100KB.
- 2026-11-07: R4: removed Markdown output (daily/weekly `.md` templates); the archive is HTML and the email text part comes from `html_to_text`. Deep-read items render as long-read articles; the source-type label is hidden. Topic chips were dropped; in keyword mode the "Matched: topics" text shows as a muted Topics line instead of a "Why it matters" box.
- 2026-11-07: Images: only relevant raster figures (main-body arXiv figure with an overview/architecture caption first, else the first main-body figure, else an appendix overview figure; `og:image` for articles; GitHub card for repos). The HF paper card (a first-page screenshot) was rejected as low value. MathML `<annotation>` LaTeX is stripped from captions and text.
- 2026-11-07: Brief: Gemini writes the overview + highlights (uncited highlights dropped); the fallback uses each deep-read item's opening sentence (results first).
- 2026-11-07: R5: ranking moved from Gemini to local semantic search (fastembed bge-small, ONNX): faster (~30s for ~1,400 items), free, no quota. Calibrated on live data: paper sims p50 0.72 / p99 0.82 → floor 0.70, ceil 0.82; short texts get per-kind calibration. Papers are scored on THE ICONIC problems only (general interests ×0.6). Topics generic outside retail (`context: true`) are scaled by a retail-context similarity (own scale 0.48–0.60).
- 2026-11-07: Gemini is only used for summaries: one batched call for the picks (thinking off), deep reads (6, thinking 512) and the brief. Gemini client pinned to API v1beta (JSON mode is unavailable on v1). Removed the keyword prefilter, LLM batch ranking and score cache.
- 2026-11-07: arXiv: `data/paper_pool.json` keeps relevant unshown papers for 30 days; paper sections cap a single source at 50% while alternatives qualify; Top allows at most 3 per source and per section. Popularity boost reduced to 0.6.
- 2026-11-07: HTML weight: fonts set once on the container (inherited), section caps trimmed to about 33 picks; the Gmail size guard still trims the lowest items if needed.
- 2026-11-07: LLM cost log: `data/llm_usage.csv`, one row per run (datetime, input tokens, output tokens, estimated USD), written in a `finally` so failed runs are logged too; committed by both workflows. Prices from the Gemini pricing page (gemini-2.5-flash $0.30/$2.50 per 1M, thinking billed as output). First live run: 42,178 in / 10,574 out ≈ $0.039. Note: gemini-2.5-flash is deprecated and from 1 Dec 2026 is billed as gemini-3.5-flash-lite (same $0.30/$2.50); consider switching `llm.model` before then.
- 2026-11-07: Switched `llm.model` to `gemini-3.5-flash-lite` (gemini-2.5-flash is deprecated and re-routed from 1 Dec 2026). Gemini 3.x rejects `thinking_budget` (400), so the config takes `thinking` / `deep_thinking` as a level string (`minimal` / `low`) or an int budget for 2.5 models. Live run: deep reads 25s (was 45s), 42,113 in / 6,417 out ≈ $0.029 (was $0.039), with similar summary quality.
- Later (v2+): VS Code chat agent / MCP over the digest archive; X/Twitter if a free option appears; embedding-based prefilter.
