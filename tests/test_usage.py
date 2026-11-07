import csv
from datetime import datetime
from types import SimpleNamespace

from news_agent.llm import UsageTracker, append_usage_log, gemini_json
from news_agent.config import load_settings


def usage(i, o, t=0):
    return SimpleNamespace(prompt_token_count=i, candidates_token_count=o, thoughts_token_count=t)


def test_tracker_sums_and_prices_thinking_as_output():
    tr = UsageTracker()
    tr.add("gemini-2.5-flash", usage(1_000_000, 100_000, 300_000))
    tr.add("gemini-2.5-flash", usage(500_000, 100_000, None))
    tr.add("unknown-model", None)
    assert tr.input_tokens == 1_500_000 and tr.output_tokens == 500_000
    assert abs(tr.cost() - (1.5 * 0.30 + 0.5 * 2.50)) < 1e-9
    assert abs(tr.cost({"gemini-2.5-flash": {"input": 1, "output": 1}}) - 2.0) < 1e-9


def test_append_usage_log_one_row_per_run_with_header(tmp_path):
    path = tmp_path / "data" / "llm_usage.csv"
    tr = UsageTracker()
    assert append_usage_log(path, datetime(2026, 11, 7, 7, 0), tr) is None and not path.exists()  # no tokens, no row
    tr.add("gemini-2.5-flash", usage(10_000, 2_000))
    append_usage_log(path, datetime(2026, 11, 7, 7, 0), tr)
    append_usage_log(path, datetime(2026, 11, 8, 7, 0), tr)
    rows = list(csv.DictReader(open(path)))
    assert list(rows[0]) == ["datetime", "input_tokens", "output_tokens", "estimated_cost_usd"]
    assert rows[0] == {"datetime": "2026-11-07T07:00:00", "input_tokens": "10000", "output_tokens": "2000",
                       "estimated_cost_usd": "0.008000"}
    assert len(rows) == 2


def test_gemini_json_records_usage(monkeypatch):
    import news_agent.llm as llm

    tr = UsageTracker()
    monkeypatch.setattr(llm, "USAGE", tr)

    class Client:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                return SimpleNamespace(text='{"ok": true}', usage_metadata=usage(120, 30, 10))

    s = load_settings()
    assert gemini_json(Client, "gemini-2.5-flash", "hi", None, s) == {"ok": True}
    assert tr.input_tokens == 120 and tr.output_tokens == 40


def test_gemini_json_thinking_level_vs_budget(monkeypatch):
    seen = []

    class Client:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                seen.append(config.thinking_config)
                return SimpleNamespace(text="{}", usage_metadata=None)

    s = load_settings()
    gemini_json(Client, "gemini-3.5-flash-lite", "hi", None, s, thinking="Low")
    gemini_json(Client, "gemini-2.5-flash", "hi", None, s, thinking=0)
    gemini_json(Client, "gemini-3.5-flash-lite", "hi", None, s)  # default from config (llm.thinking)
    assert seen[0].thinking_level.value.lower() == "low" and seen[0].thinking_budget is None
    assert seen[1].thinking_budget == 0 and seen[1].thinking_level is None
    assert seen[2].thinking_level.value.lower() == "minimal"
