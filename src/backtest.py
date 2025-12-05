import numpy as np
import torch


def backtest_model(model, X_bt, y_bt, mid_bt, device="cpu", prob_threshold=0.55):
    model.eval()
    X_bt_t = torch.from_numpy(X_bt).to(device)
    with torch.no_grad():
        logits = model(X_bt_t)
        probs = torch.softmax(logits, dim=1)[:, 1]  
        probs = probs.cpu().numpy()

    long_mask = probs > prob_threshold
    short_mask = probs < (1.0 - prob_threshold)
    position = np.zeros_like(probs, dtype=float)
    position[long_mask] = 1.0
    position[short_mask] = -1.0

    mid = np.array(mid_bt, dtype=float)
    dmid = np.diff(mid, prepend=mid[0])

    pnl = position * dmid
    equity = 1.0 + np.cumsum(pnl / mid[0])

    return equity, position, probs


def backtest_perfect_foresight(y_bt, mid_bt):
    y = np.array(y_bt, dtype=int)
    mid = np.array(mid_bt, dtype=float)
    dmid = np.diff(mid, prepend=mid[0])

    position = np.where(y == 1, 1.0, -1.0)
    pnl = position * dmid
    equity = 1.0 + np.cumsum(pnl / mid[0])

    return equity, position
