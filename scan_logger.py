from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

import pandas as pd


SCAN_DIR = Path("data") / "scans"
SCAN_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip())
    return cleaned or "scan"


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for column in safe.columns:
        if safe[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            safe[column] = safe[column].map(lambda value: json.dumps(value, default=str) if isinstance(value, (dict, list, tuple)) else value)
    return safe


def save_scan_results(df: pd.DataFrame, summary: dict[str, Any] | None = None, name: str = "full_watchlist_scan") -> Path:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    path = SCAN_DIR / f"{_timestamp()}_{_safe_name(name)}.csv"
    _csv_safe(df).to_csv(path, index=False)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "summary": summary or {},
        "csv_file": path.name,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def list_scan_files() -> list[Path]:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(SCAN_DIR.glob("*.csv"), reverse=True)


def load_scan(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_latest_scan() -> pd.DataFrame:
    files = list_scan_files()
    if not files:
        return pd.DataFrame()
    return load_scan(files[0])
