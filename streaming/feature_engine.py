from collections import deque
import numpy as np

class FeatureEngine:

    def __init__(self, seq_len: int = 100):
        self.seq_len = seq_len
        self.imb_hist = deque(maxlen=6)
        self.mid_hist = deque(maxlen=6)
        self.buf = deque(maxlen=seq_len)

    def update(self, imb: float, mid: float):
        self.imb_hist.append(float(imb))
        self.mid_hist.append(float(mid))

        I = self.imb_hist[-1]
        if len(self.imb_hist) >= 2:
            dI = self.imb_hist[-1] - self.imb_hist[-2]
        else:
            dI = 0.0

        if len(self.mid_hist) >= 2 and self.mid_hist[-2] != 0:
            r1 = (self.mid_hist[-1] - self.mid_hist[-2]) / self.mid_hist[-2]
        else:
            r1 = 0.0
        if len(self.mid_hist) >= 6 and self.mid_hist[-6] != 0:
            r5 = (self.mid_hist[-1] - self.mid_hist[-6]) / self.mid_hist[-6]
        else:
            r5 = 0.0

        self.buf.append([I, dI, r1, r5])

    def ready(self) -> bool:
        return len(self.buf) == self.seq_len

    def snapshot(self) -> np.ndarray:
        arr = np.array(self.buf, dtype=np.float32)
        return arr[None, :, :]
