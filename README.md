# AI × Ecommerce News Agent

A personal, free, automated digest of **AI, agents and data-science news, tailored to ecommerce and fashion** (built for a data scientist at THE ICONIC).

- **Daily digest** (Mon–Fri, ~7–8am Sydney): papers, model releases, AI industry news, engineering blogs, rising GitHub repos and community buzz, **ranked semantically against THE ICONIC's problems** (recommendations, search, size & fit, visual search, catalogue GenAI, shopping agents, forecasting, marketing, experimentation) and **summarised by Gemini**.
- **Weekly trends email** (Saturday ~8am Sydney): themes of the week, top 10, models, repos, and "what to try at THE ICONIC".
- **👍 / 👎 feedback**: every item has two buttons that open a pre-filled email to yourself. Hit send, and the next run adjusts topic and source weights.
- **Every summary links back to its source**: each item has a 🔗 *Source* row (e.g. arXiv · PDF · HF · Code, GitHub, article, HN discussion), and every weekly trend and idea cites the items behind it.
- **Long-read email**: it opens with **Today in brief** (an overview and 4–6 linked highlights), then **In depth** articles with a relevant figure from the paper or post, key ideas, results and how to apply them, then **More to read** as compact cards.
- **Deep read of the top items**: the top ~8 papers, blogs and repos are read in full (arXiv HTML or PDF, article text, GitHub README) and summarised in more detail: a 4–6 sentence summary, key ideas, results with numbers, and how to apply it at THE ICONIC. Other items get a 2–4 sentence TL;DR.
- **Papers: rolling 30 days, industry first**: ecommerce DS papers from large companies (Alibaba/Taobao, Amazon, Google, Netflix, Meta, …) and highly recommended papers (HF upvotes, citations, code stars) are boosted. Unshown papers stay eligible for 30 days, so a paper that takes off later still reaches you.
- **$0 to run**: GitHub Actions + Gemini free tier + Gmail.

See [plans/2026-11-06-ai-ecommerce-news-agent.md](plans/2026-11-06-ai-ecommerce-news-agent.md) for the full design.

## How it works

```
feedback (Gmail IMAP) → collect (arXiv, HF Papers 30d, OpenAlex 30d, RSS, Good AI List, HN, Reddit, Bluesky)
  → merge same paper across sources → dedupe (data/seen.json) → OpenAlex affiliations
  → + paper pool (relevant unshown arXiv/HF papers, 30 days)
  → semantic ranking: local embeddings (fastembed bge-small, no API) vs THE ICONIC problem statements
    (× retail-context factor × feedback weights + industry/popularity boost) → select (score ≥ 5, caps, source mix, Top 5)
  → Gemini summaries for the picks (one batched call)
  → deep read of the top ~8 (full text → Gemini summary, or key-sentence extraction without an LLM)
  → Today in brief (overview + highlights) → long-read HTML email + digests/daily/YYYY-MM-DD.html → commit state back to the repo
```

Each source fails on its own: if one source breaks, the run continues and the failure is listed in the email footer. Without a Gemini key, ranking still works (it never uses Gemini) and summaries fall back to key sentences extracted from the text. A typical run takes about 2 minutes (collect ~30s, rank ~30s, Gemini ~1 min).

## Setup

### 1. Gemini API key (free)
Create a key at <https://aistudio.google.com/apikey>.

To use Vertex AI instead (billed to a GCP project), leave `GOOGLE_API_KEY` empty and set `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` and `GOOGLE_APPLICATION_CREDENTIALS`. No code changes are needed.

### 2. Gmail app password
1. Turn on 2-Step Verification for the Google account.
2. Create an app password at <https://myaccount.google.com/apppasswords>.
3. Gmail → Settings → *Forwarding and POP/IMAP* → **Enable IMAP** (needed for 👍/👎 feedback).

### 3. GitHub repository
1. Push this repo to GitHub.
2. *Settings → Secrets and variables → Actions* → add the secrets:
   `GOOGLE_API_KEY`, `GMAIL_USER`, `GMAIL_APP_PASSWORD`, `DIGEST_TO`.
3. *Settings → Actions → General → Workflow permissions* → **Read and write permissions**.
4. *Actions* tab → **Daily digest** → *Run workflow* to test it now.

Schedules (in UTC, so the Sydney delivery time moves by an hour with daylight saving):

| Workflow | Cron | Sydney time |
|---|---|---|
| `daily.yml` | `0 21 * * 0-4` | Mon–Fri ~7am (AEST) / ~8am (AEDT) |
| `weekly.yml` | `0 21 * * 5` | Sat ~7am / ~8am |

The code also checks the Sydney weekday, so a late scheduled run never sends on the wrong day. Monday's digest covers 74 hours (since Friday).

## Run locally

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]" --python .venv/bin/python
cp .env.example .env   # fill in keys (git-ignored)

.venv/bin/python -m news_agent daily --dry-run --force         # writes out/daily-YYYY-MM-DD.html, no email/state
.venv/bin/python -m news_agent daily --dry-run --force --no-llm  # no Gemini calls (extractive summaries)
.venv/bin/python -m news_agent daily --only arxiv goodailist --dry-run --force
.venv/bin/python -m news_agent daily --force                   # real run: email + state update
.venv/bin/python -m news_agent weekly --dry-run --force
.venv/bin/python -m pytest -q
```

| Flag | Effect |
|---|---|
| `--dry-run` | Write the HTML preview to `out/`; no email, no IMAP, no state changes |
| `--no-llm` | Keyword-only ranking |
| `--force` | Ignore the Sydney weekday guard |
| `--since N` | Look-back window in hours (daily) |
| `--only a b` | Run only these collectors (daily) |
| `--no-feedback` | Skip Gmail feedback ingestion (daily) |
| `--no-deep-read` | Skip full-text reading of the top items (daily) |
| `--today YYYY-MM-DD` | Override the local date (testing) |

## Tuning

- **THE ICONIC problems:** `config/interests.yaml` → `topics`. Each topic's `query` is the problem statement items are compared against; `iconic: true` marks business problems (shown as 🎯), `weight` sets importance and `context: true` requires a retail/e-commerce context. `semantic` holds the similarity calibration (`floor`/`ceil`, per-kind overrides). `profile` is the reader description sent to Gemini.
- **Source mix:** `ranking.max_source_share` (one source fills at most half of a paper section while others qualify) and `max_top_per_source` keep arXiv papers in the digest alongside HF Daily Papers.
- **Digest size:** `config/interests.yaml` → `ranking.score_threshold`, `top_n` and `section_caps`.
- **Sources:** `config/sources.yaml`. Add an RSS feed with `{id, name, url, kind}` or set `enabled: false`. Good AI List thresholds are under `goodailist`.
- **Model:** `config/interests.yaml` → `llm.model`.
- **Deep read:** `config/interests.yaml` → `deep_read` (`max_items`, `kinds`, `max_chars`, `max_pdf_mb`). Each item is one Gemini call. If a page can't be fetched (paywall or 403), the summary is based on the abstract or feed text and labelled as such. The email is kept under Gmail's ~102KB clipping limit by dropping the lowest-ranked section items if needed.
- **Paper priority:** `config/interests.yaml` → `papers` sets `rolling_days`, the industry and popularity boosts, and the `companies` list (canonical name → aliases). A company counts when it appears in the author affiliations, or in a title/abstract sentence that also says the work was deployed (so "evaluated on Amazon reviews" doesn't count).
- **LLM cost log:** every run appends one row to `data/llm_usage.csv` (`datetime, input_tokens, output_tokens, estimated_cost_usd`; Sydney time). Output tokens include Gemini "thinking" tokens. The cost uses paid-tier prices from `llm.pricing` ($0.30 / $2.50 per 1M tokens for `gemini-3.5-flash-lite`), so on the free tier it is what the run *would* cost. A typical daily run is about 42k input + 6k output tokens, roughly $0.03.
- **Model:** `llm.model` (default `gemini-3.5-flash-lite`). `llm.thinking` / `llm.deep_thinking` take a Gemini 3.x level (`minimal`/`low`/`medium`/`high`) or, for Gemini 2.5 models, an int token budget.
- **Feedback strength:** `config/interests.yaml` → `feedback.step` (default ±5% per vote, clamped to 0.5×–2×). The current multipliers are in `data/feedback.json`.

## Layout

```
config/            sources.yaml, interests.yaml
src/news_agent/    main.py (CLI), collectors/, semantic.py, ranker.py, summarize.py, llm.py, deepread.py, brief.py, feedback.py, weekly.py,
                   digest.py, emailer.py, store.py, models.py, config.py
templates/         daily/weekly × html/md Jinja2 templates
data/              seen.json (dedupe), feedback.json (votes + weights), items/ (scored history, 90 days)
digests/           daily/ and weekly/ HTML archive
.github/workflows/ daily.yml, weekly.yml
plans/             design plan and index
```

## Notes
- X/Twitter is not included (its API is paid); Bluesky and newsletters stand in for it.
- Reddit often blocks GitHub-hosted runners; when that happens it shows as a failed source and the run continues.
- Good AI List is read from its undocumented `/api/repos` endpoint, so it may need fixing if the site changes.
