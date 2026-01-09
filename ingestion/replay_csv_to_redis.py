import time
import argparse
import redis
import numpy as np

from src.data_loader import load_lobster_data
from src.features import (
    compute_imbalance_and_midprice,
    filter_executions,
    build_predictors_and_response,
)
from ingestion.schemas import LOBEvent

STREAM_KEY = "lob:stream"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--stream", default=STREAM_KEY)
    parser.add_argument("--max_events", type=int, default=0, help="0 = all")
    parser.add_argument("--sleep_ms", type=int, default=0, help="simulate real-time pacing")
    parser.add_argument("--trim", type=int, default=200000, help="approx max stream length")
    args = parser.parse_args()

    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)

    df = load_lobster_data()
    I, mid = compute_imbalance_and_midprice(df)
    I_exec, mid_exec, t_exec = filter_executions(df, I, mid)

    X, y, mid_for_y = build_predictors_and_response(I_exec, mid_exec)

    n = len(y)
    if args.max_events and args.max_events > 0:
        n = min(n, args.max_events)

    print(f"Publishing {n} events to Redis stream {args.stream}...")

    for i in range(n):
        evt = LOBEvent(
            ts=int(i),
            mid=float(mid_for_y[i]),
            imbalance=float(X[i, 0]),
            y=int(y[i]),
        )

        r.xadd(
            args.stream,
            fields=evt.to_redis_fields(),
            maxlen=args.trim,
            approximate=True
        )

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print("Done.")
    print("Tip: redis-cli XRANGE lob:stream - + COUNT 3")

if __name__ == "__main__":
    main()
