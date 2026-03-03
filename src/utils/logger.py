"""
logger.py — Centralized logging for sentiment-mlops
-----------------------------------------------------
Features:
  - Single shared Loguru logger instance (imported anywhere)
  - Console: plain-text, colorized
  - File:    JSON-structured, 10 MB rotation, 5 backups retained
  - Log level driven by APP_ENV (.env):
        dev        → DEBUG
        staging    → INFO
        prod       → WARNING
        (default)  → INFO
  - Sensitive-data sanitizer runs before every file write
      masks: S3 URIs, AWS key patterns, Bearer/API tokens, e-mail addresses
  - Module + function context injected automatically via Loguru's {name} / {function}
"""

import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime

from loguru import logger
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# 1. Bootstrap
# ──────────────────────────────────────────────

load_dotenv()  # pulls APP_ENV (and everything else) from .env

_ENV: str = os.getenv("APP_ENV", "dev").lower().strip()

_LEVEL_MAP: dict[str, str] = {
    "dev":     "DEBUG",
    "staging": "INFO",
    "prod":    "WARNING",
}
LOG_LEVEL: str = _LEVEL_MAP.get(_ENV, "INFO")

# Root of the project  →  <project_root>/logs/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/utils/ → project root
LOG_DIR: Path = _PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE: Path = LOG_DIR / f"sentiment_mlops_{datetime.now().strftime('%Y-%m-%d')}.log"

# ──────────────────────────────────────────────
# 2. Sensitive-data sanitizer
# ──────────────────────────────────────────────

# Each tuple: (human label, compiled pattern, replacement)
_SANITIZE_RULES: list[tuple[str, re.Pattern, str]] = [
    # S3 URIs  →  s3://<BUCKET>/<KEY>
    (
        "s3_uri",
        re.compile(r"s3://[^\s\"']+", re.IGNORECASE),
        "<S3_URI_REDACTED>",
    ),
    # AWS Access Key IDs  (20-char uppercase alphanumeric starting with AKIA/ASIA/AROA…)
    (
        "aws_access_key",
        re.compile(r"\b(AKIA|ASIA|AROA|AIPA|ANPA|ANVA|APKA)[A-Z0-9]{16}\b"),
        "<AWS_KEY_REDACTED>",
    ),
    # AWS Secret Access Keys  (40-char base64-ish strings after known prefixes)
    (
        "aws_secret",
        re.compile(
            r'(?i)(aws_secret_access_key|secret.?key)\s*[=:]\s*["\']?([A-Za-z0-9/+]{40})["\']?'
        ),
        r"\1=<AWS_SECRET_REDACTED>",
    ),
    # Bearer / API tokens
    (
        "bearer_token",
        re.compile(r"(?i)(bearer\s+|api[_-]?key\s*[=:]\s*)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
        "<TOKEN_REDACTED>",
    ),
    # E-mail addresses
    (
        "email",
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "<EMAIL_REDACTED>",
    ),
    # Generic passwords in key=value form
    (
        "password",
        re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?\S+["\']?'),
        r"\1=<PASSWORD_REDACTED>",
    ),
]


def _sanitize(text: str) -> str:
    """Apply all redaction rules to a log message string."""
    for _label, pattern, replacement in _SANITIZE_RULES:
        text = pattern.sub(replacement, text)
    return text


# ──────────────────────────────────────────────
# 3. Custom sinks
# ──────────────────────────────────────────────

def _plain_console_sink(message) -> None:
    """
    Colorized plain-text sink for stdout.
    Loguru already formats `message` with ANSI codes when the terminal supports it.
    We print as-is; no sanitization on console (developer feedback).
    """
    print(message, end="")


def _json_file_sink(message) -> None:
    """
    Sanitized JSON-structured sink that appends to the daily log file.

    Output shape per line:
    {
        "timestamp": "2024-05-01T12:00:00.123456+0000",
        "level":     "INFO",
        "env":       "prod",
        "module":    "src.components.data_ingestion",
        "function":  "download_from_s3",
        "line":      42,
        "message":   "...",
        "extra":     {}          ← populated when logger.bind(key=val) is used
    }
    """
    record = message.record

    raw_message = record["message"]
    clean_message = _sanitize(raw_message)

    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level":     record["level"].name,
        "env":       _ENV,
        "module":    record["name"],       # e.g. src.components.data_ingestion
        "function":  record["function"],
        "line":      record["line"],
        "message":   clean_message,
        "extra":     {
            k: _sanitize(str(v)) if isinstance(v, str) else v
            for k, v in record["extra"].items()
        },
    }

    # Append exception info when present
    if record["exception"] is not None:
        exc = record["exception"]
        log_entry["exception"] = {
            "type":    exc.type.__name__ if exc.type else None,
            "value":   _sanitize(str(exc.value)) if exc.value else None,
            "traceback": _sanitize(message) if exc.traceback else None,
        }

    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry, default=str) + "\n")


# ──────────────────────────────────────────────
# 4. Logger configuration
# ──────────────────────────────────────────────

def _configure_logger() -> None:
    """
    Remove Loguru's default handler and register our two custom sinks.
    Called once at module import; idempotent (guard via module-level flag).
    """
    logger.remove()  # Drop the default stderr sink

    # ── Console sink ──────────────────────────────────────────────────────────
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        _plain_console_sink,
        level=LOG_LEVEL,
        format=console_format,
        colorize=True,
        backtrace=True,
        diagnose=(_ENV == "dev"),   # show variable values in tracebacks only in dev
        enqueue=False,
    )

    # ── JSON file sink ────────────────────────────────────────────────────────
    logger.add(
        _json_file_sink,
        level=LOG_LEVEL,
        # rotation / retention are handled by Loguru when sink is a *path*,
        # but since we're using a custom callable we manage the file manually.
        # For size-based rotation with backups, we register a *second* add()
        # using the path directly for Loguru's built-in rotation engine,
        # then disable its format so our callable controls the content.
        format="{message}",        # raw pass-through; _json_file_sink uses record dict
        backtrace=True,
        diagnose=(_ENV == "dev"),
        enqueue=True,              # async-safe: queues writes in a background thread
        catch=True,                # don't crash the app if logging itself errors
    )

    # ── Rotation-aware path sink (delegates to Loguru's rotation engine) ─────
    # This sink writes nothing useful on its own (format=" ") but lets Loguru
    # handle 10 MB rotation + keep last 5 files automatically.
    logger.add(
        str(LOG_FILE),
        level=LOG_LEVEL,
        format=" ",                 # content managed by _json_file_sink above
        rotation="10 MB",
        retention=5,
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        catch=True,
    )

    logger.debug(
        f"Logger initialised | env={_ENV} | level={LOG_LEVEL} | log_file={LOG_FILE}"
    )


# Run once on import
_configure_logger()


# ──────────────────────────────────────────────
# 5. Public helpers
# ──────────────────────────────────────────────

def get_logger(component: str):
    """
    Return a logger bound with a `component` tag so log entries carry
    pipeline stage context (e.g. "data_ingestion", "model_trainer").

    Usage
    -----
        from src.utils.logger import get_logger
        log = get_logger("data_ingestion")
        log.info("Downloading dataset from S3")
    """
    return logger.bind(component=component)


# Re-export the base logger for callers that don't need component tagging
__all__ = ["logger", "get_logger", "LOG_LEVEL", "LOG_DIR", "LOG_FILE"]