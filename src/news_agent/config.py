"""Configuration and environment loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def project_root() -> Path:
    env = os.getenv("NEWS_AGENT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Settings:
    root: Path
    sources: dict[str, Any]
    interests: dict[str, Any]
    google_api_key: str | None = None
    use_vertex: bool = False
    gmail_user: str | None = None
    gmail_app_password: str | None = None
    digest_to: str | None = None
    archive_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def digests_dir(self) -> Path:
        return self.root / "digests"

    @property
    def templates_dir(self) -> Path:
        return self.root / "templates"

    @property
    def ranking(self) -> dict[str, Any]:
        return self.interests.get("ranking", {})

    @property
    def llm(self) -> dict[str, Any]:
        return self.interests.get("llm", {})

    @property
    def topics(self) -> list[dict[str, Any]]:
        return self.interests.get("topics", [])

    @property
    def papers(self) -> dict[str, Any]:
        return self.interests.get("papers", {})

    @property
    def llm_available(self) -> bool:
        return bool(self.google_api_key) or self.use_vertex

    @property
    def email_available(self) -> bool:
        return bool(self.gmail_user and self.gmail_app_password and self.digest_to)


def load_settings(root: Path | None = None) -> Settings:
    root = root or project_root()
    if load_dotenv is not None:
        load_dotenv(root / ".env", override=False)
    return Settings(
        root=root,
        sources=load_yaml(root / "config" / "sources.yaml"),
        interests=load_yaml(root / "config" / "interests.yaml"),
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None,
        use_vertex=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes"),
        gmail_user=os.getenv("GMAIL_USER") or None,
        gmail_app_password=os.getenv("GMAIL_APP_PASSWORD") or None,
        digest_to=os.getenv("DIGEST_TO") or os.getenv("GMAIL_USER") or None,
        archive_url=os.getenv("ARCHIVE_URL") or None,
    )
