import argparse
import json
import time
from typing import Dict, Any

import numpy as np
import redis
import torch
from src.model import LOBLSTM

from streaming.feature_engine import FeatureEngine
from streaming.parquet_sink import ParquetSink

STREAM_KEY = "lob:stream"
PRED_STREAM_KEY = "lob:predictions"
GROUP = "lob_group"
CONSUMER = "c1"


def ensure_group(r: redis.Redis, stream: str, group: str):
    try:
        r.xgroup_create(stream, group, id="0-0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


class InferenceModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = LOBLSTM(input_dim=4)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, x_seq: np.ndarray) -> Dict[str, Any]:
        x = torch.from_numpy(x_seq).to(self.device)
        logits = self.model(x)            # (1,2)
        probs = torch.softmax(logits, dim=-1)[0]
        p_up = float(probs[1].item())
        pred = int(p_up >= 0.5)
        return {"p_up": p_up, "pred": pred}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--stream", default=STREAM_KEY)
    parser.add_argument("--pred_stream", default=PRED_STREAM_KEY)
    parser.add_argument("--group", default=GROUP)
    parser.add_argument("--consumer", default=CONSUMER)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--model_path", default="artifacts/model_ts.pt")
    parser.add_argument("--parquet_dir", default="data/predictions_parquet")
    parser.add_argument("--flush_every", type=int, default=1000)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--block_ms", type=int, default=1000)
    args = parser.parse_args()

    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
    ensure_group(r, args.stream, args.group)

    fe = FeatureEngine(seq_len=args.seq_len)
    model = InferenceModel(args.model_path, device="cpu")
    sink = ParquetSink(out_dir=args.parquet_dir, flush_every=args.flush_every)

    print(f"[consumer] stream={args.stream} group={args.group} consumer={args.consumer}")
    print(f"[infer] model={args.model_path}")
    print(f"[sink] parquet_dir={args.parquet_dir} flush_every={args.flush_every}")
    print(f"[serve] pred_stream={args.pred_stream}")

    events_ok = 0
    events_bad = 0
    ready_sequences = 0
    last_metrics_t = time.time()
    start_t = last_metrics_t

    last_id = ">"

    try:
        while True:
            resp = r.xreadgroup(
                groupname=args.group,
                consumername=args.consumer,
                streams={args.stream: last_id},
                count=args.count,
                block=args.block_ms,
            )
            if not resp:
                sink.flush()
                continue

            for _, messages in resp:
                for msg_id, fields in messages:
                    try:
                        ts = int(fields.get("ts", "0"))
                        imb = float(fields["imb"])
                        mid = float(fields["mid"])
                        y = int(fields.get("y", "0"))

                        fe.update(imb=imb, mid=mid)
                        events_ok += 1

                        if fe.ready():
                            x_seq = fe.snapshot() 
                            out = model.predict(x_seq)
                            ready_sequences += 1

                            row = {
                                "ts": ts,
                                "mid": mid,
                                "imb": imb,
                                "y": y,
                                "p_up": out["p_up"],
                                "pred": out["pred"],
                                "seq_len": args.seq_len,
                                "feat_dim": int(x_seq.shape[-1]),
                            }

                            r.xadd(args.pred_stream, row, maxlen=200_000, approximate=True)

                            sink.add(row)
                        r.xack(args.stream, args.group, msg_id)

                    except Exception:
                        events_bad += 1
                        r.xack(args.stream, args.group, msg_id)
            now = time.time()
            if now - last_metrics_t >= 5.0:
                dt = now - last_metrics_t
                rate = events_ok / max(now - start_t, 1e-9)
                msg = {
                    "uptime_s": round(now - start_t, 2),
                    "events_ok": events_ok,
                    "events_bad": events_bad,
                    "ready_sequences": ready_sequences,
                    "events_per_sec": round(rate, 2),
                    "flush_every": args.flush_every,
                }
                print("[METRICS]", json.dumps(msg))
                last_metrics_t = now

    except KeyboardInterrupt:
        path = sink.flush()
        print("\nStopping inference consumer.")
        if path:
            print(f"Flushed Parquet: {path}")


if __name__ == "__main__":
    main()
