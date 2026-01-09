import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_all_parquet(parquet_dir: str) -> pd.DataFrame:
    pattern = os.path.join(parquet_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found at: {pattern}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    required = {"y", "pred", "p_up"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    return df


def confidence_from_row(df: pd.DataFrame) -> pd.Series:
    p = df["p_up"].astype(float)
    pred = df["pred"].astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)
    return pd.Series(conf, index=df.index, name="conf")


def accuracy_by_confidence_bucket(df: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    conf = confidence_from_row(df)
    edges = np.linspace(0.0, 1.0, buckets + 1)
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(buckets)]
    bucket = pd.cut(conf, bins=edges, labels=labels, include_lowest=True)

    correct = (df["pred"].astype(int) == df["y"].astype(int)).astype(int)

    out = (
        pd.DataFrame({"bucket": bucket, "correct": correct, "conf": conf})
        .groupby("bucket", dropna=False)
        .agg(n=("correct", "size"), acc=("correct", "mean"), avg_conf=("conf", "mean"))
        .reset_index()
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", default="data/predictions_parquet")
    parser.add_argument("--buckets", type=int, default=10)
    args = parser.parse_args()

    df = load_all_parquet(args.parquet_dir)
    y = df["y"].astype(int)
    pred = df["pred"].astype(int)
    acc = (pred == y).mean()

    conf = confidence_from_row(df)
    avg_conf = conf.mean()
    print("=== Streaming Predictions Evaluation ===")
    print(f"rows: {len(df)}")
    print(f"accuracy: {acc:.4f}")
    print(f"avg_confidence(pred-class): {avg_conf:.4f}")
    bucket_tbl = accuracy_by_confidence_bucket(df, buckets=args.buckets)
    print("\n=== Accuracy vs Confidence Buckets ===")
    print(bucket_tbl.to_string(index=False))
    out_csv = os.path.join(args.parquet_dir, "eval_summary.csv")
    bucket_tbl.to_csv(out_csv, index=False)
    print(f"\nSaved bucket table to: {out_csv}")


if __name__ == "__main__":
    main()
