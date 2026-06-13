#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


WEBSITE_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = Path("/Users/vukrosic/my-life/llm-research-kit-scaling/queue")
RESULTS_ROOT = Path("/Users/vukrosic/my-life/llm-research-kit-scaling/results")
SNAPSHOT_PATH = WEBSITE_ROOT / "public" / "data" / "queue-snapshot.json"

EMPTY_SNAPSHOT = {
    "generated_at": "",
    "specs": [],
    "results": [],
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def to_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def get_requirement(spec: dict[str, Any], key: str) -> int | float | None:
    requires = spec.get("requires")
    if isinstance(requires, dict):
        value = to_number(requires.get(key))
        if value is not None:
            return value
    value = to_number(spec.get(key))
    if value is not None:
        return value
    return None


def normalize_status(value: str) -> str:
    status = value.strip().lower()
    if any(token in status for token in ("done", "complete", "finished", "success", "failed", "out-of-time")):
        return "done"
    if any(token in status for token in ("claim", "leased", "running", "active", "in_progress", "in-progress", "claimed")):
        return "claimed"
    return "queued"


def resolve_spec_status(spec: dict[str, Any], has_results: bool) -> str:
    raw_status = spec.get("status") or spec.get("state") or spec.get("queue_status")
    if isinstance(raw_status, str) and raw_status.strip():
        return raw_status.strip()
    claimed_hint = spec.get("claimed_by") or spec.get("claimed_worker") or spec.get("worker")
    if isinstance(claimed_hint, str) and claimed_hint.strip():
        return "claimed"
    if has_results:
        return "done"
    return "queued"


def format_metrics(metrics: Any) -> dict[str, Any]:
    if isinstance(metrics, dict):
        return metrics
    return {}


def build_snapshot() -> dict[str, Any]:
    if not QUEUE_ROOT.exists() or not RESULTS_ROOT.exists():
        return {
            "generated_at": now_utc(),
            "specs": [],
            "results": [],
        }

    specs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    results_by_spec: dict[str, list[dict[str, Any]]] = {}

    for result_path in sorted(RESULTS_ROOT.glob("**/result.json")):
        try:
            result = load_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        spec_id = str(result.get("spec_id") or result_path.parent.parent.name)
        row = {
            "spec_id": spec_id,
            "worker": str(result.get("worker") or ""),
            "finished": str(result.get("finished") or ""),
            "exit_status": str(result.get("exit_status") or ""),
            "metrics": format_metrics(result.get("metrics")),
        }
        results.append(row)
        results_by_spec.setdefault(spec_id, []).append(row)

    results.sort(key=lambda row: (row["finished"], row["spec_id"], row["worker"]), reverse=True)

    for spec_path in sorted(QUEUE_ROOT.glob("*.yaml")):
        try:
            spec = load_yaml(spec_path)
        except OSError:
            continue
        spec_id = str(spec.get("id") or spec_path.stem)
        title = str(spec.get("title") or spec_id)
        has_results = spec_id in results_by_spec
        specs.append(
            {
                "id": spec_id,
                "title": title,
                "status": resolve_spec_status(spec, has_results),
                "gpu_vram_gb": get_requirement(spec, "gpu_vram_gb"),
                "hours": get_requirement(spec, "hours"),
            }
        )

    return {
        "generated_at": now_utc(),
        "specs": specs,
        "results": results,
    }


def main() -> int:
    snapshot = build_snapshot()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
