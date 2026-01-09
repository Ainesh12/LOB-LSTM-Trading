from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class LOBEvent:
    ts: int       
    mid: float   
    imbalance: float 
    y: int 

    def to_redis_fields(self) -> Dict[str, str]:
        return {
            "ts": str(self.ts),
            "mid": f"{self.mid:.6f}",
            "imb": f"{self.imbalance:.8f}",
            "y": str(self.y),
        }
