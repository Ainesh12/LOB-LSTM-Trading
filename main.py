import torch
from src.data_loader import load_lobster_data
from src.features import (
    compute_imbalance_and_midprice,
    filter_executions,
    build_predictors_and_response,
    build_sequences,
)
from src.dataset import time_series_split
from src.train import train_model
from src.backtest import backtest_model, backtest_perfect_foresight
from src.plotting import plot_training_curves, plot_equity_curves, plot_signals_and_pnl

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_lobster_data()
    print("Loaded data:", df.shape)

    I, mid = compute_imbalance_and_midprice(df)
    I_exec, mid_exec, t_exec = filter_executions(df, I, mid)
    print(f"Execution events: {len(I_exec)}")

    X, y, mid_for_y = build_predictors_and_response(I_exec, mid_exec)
    print("Flat predictors shape:", X.shape, "labels shape:", y.shape)

    X_seq, y_seq, mid_seq = build_sequences(X, y, mid_for_y)
    print("Sequence predictors shape:", X_seq.shape, "labels shape:", y_seq.shape)

    (X_train, y_train), (X_val, y_val), (X_bt, y_bt, mid_bt) = time_series_split(
        X_seq, y_seq, mid_seq
    )
    print("Train/Val/Backtest sizes:", len(y_train), len(y_val), len(y_bt))

    model, history = train_model(X_train, y_train, X_val, y_val, device=device)

    equity_model, position_model, preds = backtest_model(
        model, X_bt, y_bt, mid_bt, device=device
    )
    equity_perfect, position_pf = backtest_perfect_foresight(y_bt, mid_bt)

    print("Final equity (model):", float(equity_model[-1]))
    print("Final equity (perfect foresight):", float(equity_perfect[-1]))


    plot_training_curves(history, "outputs/training_curves.png")
    plot_equity_curves(equity_model, equity_perfect, "outputs/equity_curves.png")
    plot_signals_and_pnl(mid_bt, position_model, equity_model, "outputs/signals_pnl.png")
    print("Plots saved to outputs/")

if __name__ == "__main__":
    main()
