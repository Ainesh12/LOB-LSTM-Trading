import numpy as np
import torch
from torch.utils.data import Dataset
from .config import BACKTEST_FRACTION, VAL_FRACTION_TV

class LOBSequenceDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y_seq).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def time_series_split(X_seq, y_seq, mid_seq):
    n = len(y_seq)
    n_bt = int(n * BACKTEST_FRACTION)
    if n_bt <= 0 or n_bt >= n:
        raise ValueError("Bad BACKTEST_FRACTION; leads to empty splits")

    X_tv, y_tv, mid_tv = X_seq[:-n_bt], y_seq[:-n_bt], mid_seq[:-n_bt]
    X_bt, y_bt, mid_bt = X_seq[-n_bt:], y_seq[-n_bt:], mid_seq[-n_bt:]

    n_tv = len(y_tv)
    n_val = int(n_tv * VAL_FRACTION_TV)
    n_train = n_tv - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError("Bad VAL_FRACTION_TV; leads to empty train/val")

    X_train, y_train = X_tv[:n_train], y_tv[:n_train]
    X_val, y_val = X_tv[n_train:], y_tv[n_train:]

    return (X_train, y_train), (X_val, y_val), (X_bt, y_bt, mid_bt)
