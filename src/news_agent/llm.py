"""Gemini helpers (summaries only): client creation and JSON-mode calls with retries."""
from __future__ import annotations

import csv
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

TLDR_CHARS = 450

# USD per 1M tokens (Gemini Developer API, paid tier; output includes thinking tokens).
# The free tier is $0, so this is what the run *would* cost on the paid tier. Override in llm.pricing.
DEFAULT_PRICING = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
}
USAGE_LOG_FIELDS = ["datetime", "input_tokens", "output_tokens", "estimated_cost_usd"]


class UsageTracker:
    """Accumulates Gemini token usage for one run (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.by_model: dict[str, dict[str, int]] = {}

    def add(self, model: str, usage: Any) -> None:
        if usage is None:
            return
        inp = int(getattr(usage, "prompt_token_count", 0) or 0)
        out = int(getattr(usage, "candidates_token_count", 0) or 0) + int(getattr(usage, "thoughts_token_count", 0) or 0)
        with self._lock:
            m = self.by_model.setdefault(model, {"input": 0, "output": 0})
            m["input"] += inp
            m["output"] += out

    @property
    def input_tokens(self) -> int:
        return sum(m["input"] for m in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(m["output"] for m in self.by_model.values())

    def cost(self, pricing: dict[str, dict[str, float]] | None = None) -> float:
        prices = {**DEFAULT_PRICING, **(pricing or {})}
        total = 0.0
        for model, m in self.by_model.items():
            p = prices.get(model) or prices["gemini-2.5-flash"]
            total += m["input"] / 1e6 * float(p["input"]) + m["output"] / 1e6 * float(p["output"])
        return total


USAGE = UsageTracker()


def append_usage_log(path: Path, when: datetime, tracker: UsageTracker = USAGE,
                     pricing: dict[str, dict[str, float]] | None = None) -> dict[str, Any] | None:
    """Append one row (datetime, input tokens, output tokens, estimated USD cost) to the CSV log."""
    if tracker.input_tokens == 0 and tracker.output_tokens == 0:
        return None
    row = {
        "datetime": when.isoformat(timespec="seconds"),
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "estimated_cost_usd": f"{tracker.cost(pricing):.6f}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=USAGE_LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return row


def get_client(settings: "Settings") -> Any | None:
    """Create a google-genai client, or None when no credentials / SDK problems."""
    if not settings.llm_available:
        return None
    try:
        from google import genai
        from google.genai import types

        if settings.use_vertex:
            return genai.Client()  # GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION read from env
        # JSON mode (response_schema) is only available on v1beta of the Gemini Developer API.
        api_version = settings.llm.get("api_version", "v1beta")
        return genai.Client(api_key=settings.google_api_key, http_options=types.HttpOptions(api_version=api_version))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not create Gemini client: %s", exc)
        return None


def _is_retryable(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
        return False  # bad request / auth errors won't fix themselves
    return isinstance(exc, Exception)


def _parse_json_text(text: str | None) -> Any:
    if not text:
        raise ValueError("empty LLM response")
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    return json.loads(text)


def gemini_json(
    client: Any, model: str, contents: Any, schema: Any, settings: "Settings", thinking: int | str | None = None
) -> Any:
    """Call Gemini with a JSON response schema, retrying with exponential backoff. Returns parsed JSON.

    ``thinking`` controls Gemini "thinking": a string is a Gemini 3.x ``thinking_level``
    ("minimal", "low", "medium", "high"); an int is a Gemini 2.5 ``thinking_budget`` in tokens (0 = off).
    Defaults to ``llm.thinking``.
    """
    from google.genai import types

    llm = settings.llm
    value = llm.get("thinking", llm.get("thinking_budget")) if thinking is None else thinking
    kwargs: dict[str, Any] = {}
    if isinstance(value, str) and value.strip():
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=value.strip().lower())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(value))
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=float(llm.get("temperature", 0.2)),
        **kwargs,
    )
    min_wait = float(llm.get("retry_min_wait_s", 2))
    retrying = Retrying(
        stop=stop_after_attempt(max(1, int(llm.get("max_retries", 4)))),
        wait=wait_exponential(multiplier=min_wait, min=min_wait, max=max(min_wait, 60)),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        before_sleep=lambda rs: log.warning("Gemini call failed (%s); retry %d in %.0fs",
                                            str(rs.outcome.exception())[:120], rs.attempt_number,
                                            rs.next_action.sleep if rs.next_action else 0),
    )
    for attempt in retrying:
        with attempt:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            USAGE.add(model, getattr(resp, "usage_metadata", None))  # failed attempts that returned are billed too
            return _parse_json_text(getattr(resp, "text", None))
    raise RuntimeError("unreachable")  # pragma: no cover


def deep_thinking(settings: "Settings") -> int | str | None:
    """Thinking setting for deep reads / brief / weekly (``llm.deep_thinking``; int budget or str level)."""
    return settings.llm.get("deep_thinking", settings.llm.get("deep_thinking_budget"))


def first_sentences(text: str, limit: int = TLDR_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sentences:
        cand = f"{out} {s}".strip()
        if len(cand) > limit:
            break
        out = cand
    if not out:
        out = text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return out
