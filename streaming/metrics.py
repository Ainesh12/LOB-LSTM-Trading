from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Metrics:
    start_ts: float = field(default_factory=time.time)
    last_report_ts: float = field(default_factory=time.time)

    events_ok: int = 0
    events_bad: int = 0
    batches: int = 0

    ready_sequences: int = 0
    last_latency_ms: float = 0.0

    def bump_ok(self, n: int = 1) -> None:
        self.events_ok += n

    def bump_bad(self, n: int = 1) -> None:
        self.events_bad += n

    def bump_batch(self) -> None:
        self.batches += 1

    def bump_ready(self, n: int = 1) -> None:
        self.ready_sequences += n

    def set_latency_ms(self, ms: float) -> None:
        self.last_latency_ms = ms

    def snapshot(self) -> Dict[str, str]:
        now = time.time()
        elapsed = max(now - self.start_ts, 1e-9)
        eps = self.events_ok / elapsed 

        return {
            "uptime_s": f"{elapsed:.2f}",
            "events_ok": str(self.events_ok),
            "events_bad": str(self.events_bad),
            "batches": str(self.batches),
            "ready_sequences": str(self.ready_sequences),
            "events_per_sec": f"{eps:.2f}",
            "last_latency_ms": f"{self.last_latency_ms:.3f}",
        }

    def should_report(self, every_s: float) -> bool:
        now = time.time()
        if now - self.last_report_ts >= every_s:
            self.last_report_ts = now
            return True
        return False

