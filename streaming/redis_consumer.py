import argparse
import os
import time
import redis

from streaming.feature_engine import FeatureEngine
from streaming.schema import parse_event
from streaming.metrics import Metrics

STREAM_KEY = "lob:stream"
GROUP = "lob_group"
CONSUMER = "c1"

METRICS_STREAM = "lob:metrics"
SILVER_PATH = os.path.join("outputs", "silver_events.csv")


def ensure_group(r: redis.Redis, stream: str, group: str):
    try:
        r.xgroup_create(stream, group, id="0-0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def append_silver_rows(path: str, rows):
    ensure_dir(path)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        if write_header:
            f.write("ts,mid,imb,y\n")
        for r in rows:
            f.write(f"{r['ts']},{r['mid']},{r['imb']},{r['y']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--stream", default=STREAM_KEY)
    parser.add_argument("--group", default=GROUP)
    parser.add_argument("--consumer", default=CONSUMER)
    parser.add_argument("--seq_len", type=int, default=100)

    parser.add_argument("--report_every_s", type=float, default=2.0)
    parser.add_argument("--silver_path", default=SILVER_PATH)
    parser.add_argument("--silver_batch", type=int, default=500)
    args = parser.parse_args()

    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
    ensure_group(r, args.stream, args.group)

    fe = FeatureEngine(seq_len=args.seq_len)
    metrics = Metrics()
    last_id = ">"

    silver_buffer = []

    print(f"Consuming from stream={args.stream} group={args.group} consumer={args.consumer}")

    try:
        while True:
            t0 = time.time()
            resp = r.xreadgroup(
                groupname=args.group,
                consumername=args.consumer,
                streams={args.stream: last_id},
                count=200,
                block=1000,
            )
            latency_ms = (time.time() - t0) * 1000.0
            metrics.set_latency_ms(latency_ms)

            if not resp:
                if metrics.should_report(args.report_every_s):
                    snap = metrics.snapshot()
                    r.xadd(METRICS_STREAM, snap)
                    print(f"[METRICS] {snap}")
                continue

            metrics.bump_batch()

            for _, messages in resp:
                for msg_id, fields in messages:
                    evt = parse_event(fields)
                    if evt is None:
                        metrics.bump_bad()
                        r.xack(args.stream, args.group, msg_id)
                        continue

                    metrics.bump_ok()
                    fe.update(imb=evt.imb, mid=evt.mid)

                    silver_buffer.append({"ts": evt.ts, "mid": evt.mid, "imb": evt.imb, "y": evt.y})
                    if len(silver_buffer) >= args.silver_batch:
                        append_silver_rows(args.silver_path, silver_buffer)
                        silver_buffer.clear()
                    if fe.ready():
                        metrics.bump_ready()
                        x_seq = fe.snapshot()
                        print(f"READY seq={x_seq.shape} last_y={evt.y} last_mid={evt.mid:.4f}")
                    r.xack(args.stream, args.group, msg_id)
            if metrics.should_report(args.report_every_s):
                snap = metrics.snapshot()
                r.xadd(METRICS_STREAM, snap)
                print(f"[METRICS] {snap}")

    except KeyboardInterrupt:
        if silver_buffer:
            append_silver_rows(args.silver_path, silver_buffer)
        print("\nStopping Redis Consumer (Ctrl+C)")


if __name__ == "__main__":
    main()
