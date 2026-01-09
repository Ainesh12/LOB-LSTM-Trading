from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ParquetSink:
    out_dir: str = "data/predictions_parquet"
    flush_every: int = 1000
    run_id: str = field(default_factory=lambda: str(int(time.time())))
    _buf: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, row: Dict[str, Any]) -> None:
        self._buf.append(row)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> Optional[str]:
        if not self._buf:
            return None

        os.makedirs(self.out_dir, exist_ok=True)
        df = pd.DataFrame(self._buf)
        self._buf.clear()
        path = os.path.join(self.out_dir, f"predictions_run={self.run_id}_{int(time.time()*1000)}.parquet")
        df.to_parquet(path, index=False)
        return path
