from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LOBEvent:
    ts: int
    mid: float
    imb: float
    y: int = 0


def _require(fields: Dict[str, Any], key: str) -> str:
    if key not in fields:
        raise ValueError(f"Schema violation: missing field '{key}'")
    return fields[key]


def parse_event(fields: Dict[str, Any]) -> Optional[LOBEvent]:
    try:
        ts_raw = _require(fields, "ts")
        mid_raw = _require(fields, "mid")
        imb_raw = _require(fields, "imb")

        ts = int(ts_raw)
        mid = float(mid_raw)
        imb = float(imb_raw)
        y = int(fields.get("y", "0"))

        if mid <= 0:
            return None

        return LOBEvent(ts=ts, mid=mid, imb=imb, y=y)
    except Exception:
        return None
