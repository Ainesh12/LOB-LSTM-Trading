import numpy as np
import pandas as pd
from .config import LVL, HORIZON, SEQ_LEN


def compute_imbalance_and_midprice(df: pd.DataFrame):
    best_ask = df["AskPrice1"].to_numpy(dtype=float)
    best_bid = df["BidPrice1"].to_numpy(dtype=float)
    mid = (best_ask + best_bid) / 2.0

    bid_vol = np.zeros(len(df), dtype=float)
    ask_vol = np.zeros(len(df), dtype=float)
    for lvl in range(1, LVL + 1):
        bid_vol += df[f"BidSize{lvl}"].to_numpy(dtype=float)
        ask_vol += df[f"AskSize{lvl}"].to_numpy(dtype=float)

    denom = bid_vol + ask_vol
    denom[denom == 0] = 1.0
    I = (bid_vol - ask_vol) / denom

    return I, mid


def filter_executions(df: pd.DataFrame, I: np.ndarray, mid: np.ndarray):
    typ = df["Type"].to_numpy()
    mask = np.isin(typ, [4, 5])

    I_exec = I[mask]
    mid_exec = mid[mask]
    t_exec = df["Time"].to_numpy()[mask]

    return I_exec, mid_exec, t_exec


def build_predictors_and_response(I_exec: np.ndarray, mid_exec: np.ndarray):
    I = pd.Series(I_exec)
    mid = pd.Series(mid_exec)

    dI = I.diff().fillna(0.0)
    r1 = (mid.diff().fillna(0.0)) / mid.shift(1).replace(0, np.nan)
    r1 = r1.fillna(0.0)

    r5 = (mid - mid.shift(5)) / mid.shift(5)
    r5 = r5.fillna(0.0)

    X = np.vstack([I.values, dI.values, r1.values, r5.values]).T  # (N, 4)

    future_mid = mid.shift(-HORIZON)
    delta = (future_mid - mid).to_numpy()

    mask = delta != 0
    X = X[mask]
    mid_for_y = mid.values[mask]
    delta = delta[mask]

    y = (delta > 0).astype(np.int64)

    return X.astype(np.float32), y, mid_for_y.astype(np.float32)


def build_sequences(X: np.ndarray, y: np.ndarray, mid_for_y: np.ndarray):
    N, n_feat = X.shape
    if N <= SEQ_LEN:
        raise ValueError(f"Not enough observations ({N}) for SEQ_LEN={SEQ_LEN}")

    seqs = []
    labels = []
    mids = []
    for t in range(SEQ_LEN - 1, N):
        seqs.append(X[t - SEQ_LEN + 1 : t + 1])
        labels.append(y[t])
        mids.append(mid_for_y[t])

    X_seq = np.stack(seqs, axis=0).astype(np.float32)
    y_seq = np.array(labels, dtype=np.int64)
    mid_seq = np.array(mids, dtype=np.float32)

    return X_seq, y_seq, mid_seq
