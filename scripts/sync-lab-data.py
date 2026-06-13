#!/usr/bin/env python3
"""Sync the public lab snapshot from the local experiment registry.

This script reads the SQLite registry and writes a sanitized JSON snapshot
to public/data/lab-snapshot.json. It is intentionally stdlib-only.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path("/Users/vukrosic/my-life/experiment-registry/registry/experiments.sqlite")
OUTPUT_PATH = REPO_ROOT / "public" / "data" / "lab-snapshot.json"

HOME_PATH_RE = re.compile(
    r"(?i)(?:file://)?(?:/Users/[^/\s'\"`<>]+|/home/[^/\s'\"`<>]+|/root(?:/[^/\s'\"`<>]+)?)"
)
SECRET_ASSIGN_RE = re.compile(
    r"(?i)(?P<prefix>(?:--?(?:api[-_]?key|secret|token|password|passwd|bearer|session|cookie|auth|access[-_]?token)\s*(?:=|\s+)))(?P<value>[^\s'\"`]+)"
)
TOKEN_RE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")
HEX_RE = re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_snapshot() -> dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "summary": {
            "threads": 0,
            "runs": 0,
            "queue_items": 0,
            "decisions": 0,
            "ideas": 0,
        },
        "threads": [],
        "runs": [],
        "decisions": [],
    }


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def redact_sensitive(value: str) -> str:
    text = value
    text = HOME_PATH_RE.sub("[redacted-path]", text)
    text = SECRET_ASSIGN_RE.sub(lambda m: f"{m.group('prefix')}[redacted]", text)
    text = TOKEN_RE.sub("[redacted-token]", text)
    text = HEX_RE.sub("[redacted-token]", text)
    return text


def clean_text(value: Any, *, limit: int | None = None) -> str:
    if value is None:
        return ""
    text = redact_sensitive(str(value))
    text = normalize_whitespace(text)
    if limit is not None and len(text) > limit:
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return text


def date_only(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return text.split("T", 1)[0]


def metric_payload(row: sqlite3.Row) -> dict[str, str] | None:
    candidates: list[tuple[str, Any, str]] = [
        ("final val loss", row["final_val_loss"], "loss"),
        ("final train loss", row["final_train_loss"], "loss"),
        ("final val accuracy", row["final_val_accuracy"], "accuracy"),
        ("steps", row["actual_steps"], "int"),
        ("tokens seen", row["tokens_seen"], "int"),
    ]
    for label, value, kind in candidates:
        if value is None:
            continue
        if kind == "int":
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            return {"label": label, "value": f"{number:,}"}
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        precision = 4
        if kind == "accuracy":
            precision = 4
        return {"label": label, "value": f"{number:.{precision}f}"}
    return None


def command_summary(command: Any, name: str) -> str:
    text = clean_text(command, limit=180)
    if not text:
        text = clean_text(name, limit=80)
    return text


def connect_db() -> sqlite3.Connection | None:
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def fetch_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"]) if row else 0
    except sqlite3.Error:
        return 0


def fetch_threads(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT name, status, created_at, summary, hypothesis
        FROM threads
        ORDER BY created_at DESC, name ASC
        """
    ).fetchall()
    threads: list[dict[str, Any]] = []
    for row in rows:
        title = clean_text(row["summary"] or row["hypothesis"] or row["name"], limit=140)
        if not title:
            title = clean_text(row["name"], limit=80)
        threads.append(
            {
                "name": clean_text(row["name"], limit=80),
                "title": title,
                "status": clean_text(row["status"], limit=40),
                "created": date_only(row["created_at"]),
            }
        )
    return threads


def fetch_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT name, command, status, final_val_loss, final_train_loss, final_val_accuracy,
               actual_steps, tokens_seen, created_at, finished_at
        FROM runs
        ORDER BY COALESCE(finished_at, created_at) DESC, created_at DESC, name ASC
        LIMIT 20
        """
    ).fetchall()
    runs: list[dict[str, Any]] = []
    for row in rows:
        runs.append(
            {
                "name": clean_text(row["name"], limit=80),
                "command_summary": command_summary(row["command"], row["name"]),
                "status": clean_text(row["status"], limit=40),
                "metric": metric_payload(row),
                "created": date_only(row["finished_at"] or row["created_at"]),
            }
        )
    return runs


def fetch_decisions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT decision, reason, decided_at
        FROM decisions
        ORDER BY decided_at DESC, id DESC
        """
    ).fetchall()
    decisions: list[dict[str, Any]] = []
    for row in rows:
        text = clean_text(row["decision"], limit=140)
        reason = clean_text(row["reason"], limit=200)
        if reason:
            text = f"{text}: {reason}" if text else reason
        decisions.append(
            {
                "decision": text,
                "created": date_only(row["decided_at"]),
            }
        )
    return decisions


def build_snapshot() -> dict[str, Any]:
    conn = connect_db()
    if conn is None:
        return empty_snapshot()

    try:
        snapshot = {
            "generated_at": now_iso(),
            "summary": {
                "threads": fetch_count(conn, "threads"),
                "runs": fetch_count(conn, "runs"),
                "queue_items": fetch_count(conn, "queue_items"),
                "decisions": fetch_count(conn, "decisions"),
                "ideas": fetch_count(conn, "ideas"),
            },
            "threads": fetch_threads(conn),
            "runs": fetch_runs(conn),
            "decisions": fetch_decisions(conn),
        }
    except sqlite3.Error:
        snapshot = empty_snapshot()
    finally:
        conn.close()
    return snapshot


def write_snapshot(snapshot: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    snapshot = build_snapshot()
    write_snapshot(snapshot)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
