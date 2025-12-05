from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def ensure_outdir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curves(history, out_path="outputs/training_curves.png"):
    out_path = ensure_outdir(out_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_name("loss_curves.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["val_acc"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_name("accuracy_curves.png"))
    plt.close()


def plot_equity_curves(eq_model, eq_pf, out_path="outputs/equity_curves.png"):
    out_path = ensure_outdir(out_path)
    steps = np.arange(len(eq_model))

    plt.figure()
    plt.plot(steps, eq_model, label="Model equity")
    plt.plot(steps, eq_pf, label="Perfect foresight", linestyle="--")
    plt.xlabel("Backtest step")
    plt.ylabel("Equity")
    plt.title("Equity Curve: Model vs Perfect Foresight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_signals_and_pnl(mid_bt, position, equity, out_path="outputs/signals_pnl.png"):
    out_path = ensure_outdir(out_path)

    mid_bt = np.asarray(mid_bt, dtype=float)
    position = np.asarray(position, dtype=float)
    equity = np.asarray(equity, dtype=float)

    steps = np.arange(len(mid_bt))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(steps, mid_bt, label="Midprice")
    ax1.set_xlabel("Backtest step")
    ax1.set_ylabel("Midprice")
    ax1.set_title("Signals and P&L over Backtest")

    long_idx = np.where(position > 0)[0]
    short_idx = np.where(position < 0)[0]

    ax1.scatter(long_idx, mid_bt[long_idx], marker="^", label="Long", alpha=0.7)
    ax1.scatter(short_idx, mid_bt[short_idx], marker="v", label="Short", alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(steps, equity, label="Equity", linestyle="--")
    ax2.set_ylabel("Equity")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)