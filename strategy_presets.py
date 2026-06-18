from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json


PRESET_FILE = Path("data") / "strategy_presets.json"
PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_presets() -> dict[str, dict[str, Any]]:
    if not PRESET_FILE.exists():
        return {}
    try:
        data = json.loads(PRESET_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_preset(name: str, config: dict[str, Any]) -> None:
    presets = load_presets()
    presets[name.strip() or "Untitled preset"] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
    }
    PRESET_FILE.write_text(json.dumps(presets, indent=2, default=str), encoding="utf-8")


def delete_preset(name: str) -> None:
    presets = load_presets()
    presets.pop(name, None)
    PRESET_FILE.write_text(json.dumps(presets, indent=2, default=str), encoding="utf-8")
