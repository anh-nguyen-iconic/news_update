"""CLI entry point: `python -m news_agent daily|weekly`."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .collectors import run_all
from .brief import build_brief
from .deepread import deep_read
from .config import Settings, load_settings
from .digest import render_daily, render_weekly
from .emailer import send_email, write_preview
from .feedback import FeedbackStore, ingest_from_gmail
from .llm import USAGE, append_usage_log
from .models import DailyDigest, RunStats, ScoredItem
from .ranker import rank, select
from .summarize import summarize
from .store import History, PaperPool, SeenStore, merge_duplicates
from .weekly import build_weekly

log = logging.getLogger("news_agent")

# Local time of the reader: digest dates, weekday guard and the cost log use it. Override with NEWS_AGENT_TZ.
TZ = ZoneInfo(os.getenv("NEWS_AGENT_TZ") or "Asia/Ho_Chi_Minh")
DAILY_WEEKDAYS = {0, 1, 2, 3, 4}  # Mon-Fri (local)
WEEKLY_WEEKDAY = 5  # Saturday (local)


def local_today(override: str | None = None) -> date:
    return date.fromisoformat(override) if override else datetime.now(TZ).date()


def default_window_hours(today: date) -> int:
    # Monday covers Friday morning -> Monday morning.
    return 74 if today.weekday() == 0 else 26


MAX_WINDOW_HOURS = 168


def _last_run_path(settings: Settings):
    return settings.data_dir / "last_run.json"


def window_hours(today: date, settings: Settings, now: datetime | None = None) -> int:
    """Look-back window: at least the default, and back to the last successful run (+2h overlap) if that
    was longer ago (e.g. a dropped scheduled run), capped at a week. Overlaps are removed by seen.json."""
    default = default_window_hours(today)
    path = _last_run_path(settings)
    try:
        last = datetime.fromisoformat(json.loads(path.read_text())["daily"])
    except (OSError, ValueError, KeyError, TypeError):
        return default
    now = now or datetime.now(timezone.utc)
    since_last = (now - last).total_seconds() / 3600 + 2
    return int(min(MAX_WINDOW_HOURS, max(default, math.ceil(since_last))))


def _record_run(settings: Settings, now: datetime) -> None:
    path = _last_run_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"daily": now.isoformat(timespec="seconds")}) + "\n", encoding="utf-8")


def _feedback_store(settings: Settings) -> FeedbackStore:
    return FeedbackStore(settings.data_dir / "feedback.json", settings.interests.get("feedback", {}))


def _enrich_affiliations(items: list, settings: Settings) -> None:
    """Best-effort OpenAlex lookup of author affiliations for papers old enough to be indexed (~2+ days)."""
    cfg = settings.sources.get("openalex") or {}
    if not cfg.get("enabled"):
        return
    try:
        from .collectors import openalex
        from .collectors.base import make_client
    except ImportError:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    todo = [
        it for it in items
        if it.kind == "paper" and it.key.startswith("arxiv:") and not it.affiliations
        and it.published is not None and it.published < cutoff
    ]
    if not todo:
        return
    try:
        with make_client(settings.sources.get("http")) as client:
            n = openalex.enrich(todo, client, cfg)
        log.info("openalex: enriched %d/%d papers with affiliations", n, len(todo))
    except Exception as exc:  # noqa: BLE001 - enrichment is optional
        log.warning("openalex enrichment failed: %s", exc)


GMAIL_CLIP_BYTES = 100_000  # Gmail clips messages above ~102KB ("[Message clipped]")


def render_fitted(digest: DailyDigest, settings: Settings) -> tuple[str, str, str]:
    """Render the daily digest, dropping the lowest-ranked section items until the HTML fits Gmail."""
    subject, html, text = render_daily(digest, settings)
    dropped = 0
    while len(html.encode("utf-8")) > GMAIL_CLIP_BYTES:
        candidates = [(k, v) for k, v in digest.sections.items() if v]
        if not candidates:
            break
        key, items = max(candidates, key=lambda kv: (len(kv[1]), -min(i.score for i in kv[1])))
        items.remove(min(items, key=lambda i: (i.score, i.key)))
        if not items:
            del digest.sections[key]
        dropped += 1
        subject, html, text = render_daily(digest, settings)
    if dropped:
        digest.stats.selected -= dropped
        log.warning("email too large for Gmail: dropped %d lowest-ranked section items", dropped)
    return subject, html, text


def run_daily(settings: Settings, args: argparse.Namespace) -> int:
    today = local_today(args.today)
    if today.weekday() not in DAILY_WEEKDAYS and not args.force:
        log.info("%s is a weekend (%s) - skipping daily run (use --force to override)", today, TZ.key)
        return 0

    archive = settings.digests_dir / "daily" / f"{today.isoformat()}.html"
    if archive.exists() and not args.force and not args.dry_run:
        log.info("today's digest (%s) was already sent - skipping (use --force to resend)", archive.name)
        return 0

    window = args.since or window_hours(today, settings)
    since = datetime.now(timezone.utc) - timedelta(hours=window)
    stats = RunStats(window_hours=window)
    history = History(settings.data_dir / "items")
    feedback = _feedback_store(settings)
    seen = SeenStore(
        settings.data_dir / "seen.json",
        resurface_multiplier=settings.sources.get("goodailist", {}).get("resurface_multiplier", 2.0),
    )

    if not args.dry_run and not args.no_feedback:
        stats.feedback_ingested = ingest_from_gmail(settings, feedback, history)
        feedback.save()
        log.info("feedback votes ingested: %d", stats.feedback_ingested)

    items, failures = run_all(settings.sources, since, only=args.only)
    stats.collected = len(items)
    stats.failed_sources = failures
    for it in items:
        stats.per_source[it.source] = stats.per_source.get(it.source, 0) + 1

    # Every summary must link back to its source: drop anything without a usable URL.
    items = [it for it in merge_duplicates(items) if it.url.startswith(("http://", "https://"))]
    # Rolling 30-day paper window: relevant arXiv/HF papers not shown yet stay in the pool.
    pool = PaperPool(settings.data_dir / "paper_pool.json", int(settings.papers.get("rolling_days", 30)))
    items = pool.merge_into(items)
    fresh = seen.filter_new(items)
    stats.after_dedupe = len(fresh)
    _enrich_affiliations(fresh, settings)

    # Ranking is semantic (local embeddings vs THE ICONIC problems); Gemini only summarises.
    ranked = rank(fresh, settings, feedback)
    threshold = float(settings.ranking.get("score_threshold", 5))
    stats.ranked = len(ranked)
    stats.candidates = sum(1 for it in ranked if it.score >= threshold)
    log.info("collected=%d new=%d (pool %d) relevant=%d", stats.collected, stats.after_dedupe, len(pool.data),
             stats.candidates)
    top, sections = select(ranked, settings)
    shown = {it.key for it in top} | {it.key for lst in sections.values() for it in lst}
    for it in ranked:
        it.first_seen = it.first_seen or today
        if it.key in shown:
            it.shown_on = today

    if args.no_llm:
        settings.google_api_key, settings.use_vertex = None, False
    if not args.no_deep_read:
        deep_read(top, sections, settings)
    selected = [*top, *(i for lst in sections.values() for i in lst)]
    stats.llm_used = summarize(selected, settings) > 0 or any(i.deep_llm for i in selected)
    stats.selected = len(selected)
    log.info("ranked=%d selected=%d gemini=%s", stats.ranked, stats.selected, stats.llm_used)

    digest = DailyDigest(date=today, top=top, sections=sections, stats=stats)
    build_brief(digest, settings)
    subject, html, text = render_fitted(digest, settings)

    if args.dry_run:
        path = write_preview(settings.root / "out" / f"daily-{today.isoformat()}.html", html)
        log.info("dry run: preview written to %s (subject: %s)", path, subject)
        return 0

    write_preview(settings.digests_dir / "daily" / f"{today.isoformat()}.html", html)

    # Persist relevant/shown items before sending so feedback links and the weekly digest resolve.
    keep_min = float(settings.papers.get("pool_min_score", 4))
    history.save(today, [it for it in ranked if it.key in shown or it.score >= keep_min])
    history.prune(90, today)

    if not args.dry_run and not settings.email_available:
        log.error("email not configured (GMAIL_USER / GMAIL_APP_PASSWORD / DIGEST_TO); not marking items as seen")
        return 1

    send_email(settings, subject, html, text)
    log.info("email sent to %s", settings.digest_to)

    # Mark only after a successful send so a failed run is retried with the same items.
    # Papers are marked only once shown, so unshown papers stay eligible for the rolling window.
    seen.mark([it for it in fresh if it.kind != "paper" or it.key in shown])
    seen.prune(60)
    seen.save()
    pool.update(ranked, shown, today, float(settings.papers.get("pool_min_score", 4)))
    pool.save()
    _record_run(settings, datetime.now(timezone.utc))
    return 0


def run_weekly(settings: Settings, args: argparse.Namespace) -> int:
    today = local_today(args.today)
    if today.weekday() != WEEKLY_WEEKDAY and not args.force:
        log.info("%s is not Saturday (%s) - skipping weekly run (use --force to override)", today, TZ.key)
        return 0
    iso = today.isocalendar()
    archive = settings.digests_dir / "weekly" / f"{iso[0]}-W{iso[1]:02d}.html"
    if archive.exists() and not args.force and not args.dry_run:
        log.info("this week's digest (%s) was already sent - skipping (use --force to resend)", archive.name)
        return 0
    if args.no_llm:
        settings.google_api_key, settings.use_vertex = None, False

    history = History(settings.data_dir / "items")
    weekly = build_weekly(history, _feedback_store(settings), settings, today)
    subject, html, text = render_weekly(weekly, settings)

    if args.dry_run:
        path = write_preview(settings.root / "out" / f"weekly-{weekly.week}.html", html)
        log.info("dry run: preview written to %s (subject: %s)", path, subject)
        return 0

    write_preview(settings.digests_dir / "weekly" / f"{weekly.week}.html", html)
    if not settings.email_available:
        log.error("email not configured (GMAIL_USER / GMAIL_APP_PASSWORD / DIGEST_TO)")
        return 1
    send_email(settings, subject, html, text)
    log.info("weekly email sent to %s", settings.digest_to)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="news-agent", description="AI x Ecommerce news digest")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in (("daily", "collect, rank and email the daily digest"), ("weekly", "email the weekly trends digest")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--dry-run", action="store_true", help="write HTML preview to out/, no email, no state changes")
        sp.add_argument("--no-llm", action="store_true", help="keyword-only ranking (no Gemini calls)")
        sp.add_argument("--force", action="store_true", help="ignore the weekday schedule guard (local time)")
        sp.add_argument("--today", help="override local date (YYYY-MM-DD), for testing")
        sp.add_argument("-v", "--verbose", action="store_true")
        if name == "daily":
            sp.add_argument("--since", type=int, help="look-back window in hours (default 26, Monday 74)")
            sp.add_argument("--only", nargs="+", help="run only these collectors (e.g. arxiv rss)")
            sp.add_argument("--no-feedback", action="store_true", help="skip Gmail IMAP feedback ingestion")
            sp.add_argument("--no-deep-read", action="store_true", help="skip full-text reading of the top items")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = load_settings()
    USAGE.reset()
    try:
        if args.command == "daily":
            return run_daily(settings, args)
        return run_weekly(settings, args)
    finally:
        # One row per run in data/llm_usage.csv (also when the run fails: tokens were still spent).
        row = append_usage_log(settings.data_dir / "llm_usage.csv", datetime.now(TZ), pricing=settings.llm.get("pricing"))
        if row:
            log.info("Gemini usage: %s input + %s output tokens, est. $%s",
                     row["input_tokens"], row["output_tokens"], row["estimated_cost_usd"])


if __name__ == "__main__":
    sys.exit(main())
