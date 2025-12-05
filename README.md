# LOB-LSTM Trading (In Progress)

Deep learning–based intraday trading strategy on **limit order book (LOB)** data using an **LSTM model**.  
The current version focuses on **offline backtesting** in Python; the next phase will add a **real-time C++ execution engine** wired through **Redis**.

---

## Overview

This project trains an LSTM model on **NASDQ LOBSTER** order book data (INTC) to predict **short-horizon mid-price direction** and backtests a simple long/short strategy based on the model’s signals.

**Completed so far:**

- End-to-end Python pipeline:
  - Load & preprocess LOBSTER message/order book files
  - Compute order flow imbalance & midprice features
  - Build sequence inputs for an LSTM
  - Train & validate a classification model
  - Backtest a trading strategy based on model signals
- Performance tracking:
  - Train/validation loss & accuracy curves
  - Equity curve vs. a “perfect foresight” upper bound
  - Signals & PnL visualization

**In progress:**

- C++ low-latency trading engine (`cpp_engine/`)
- Redis pub/sub bridge between Python model and C++ engine
- Web dashboard for real-time equity/position monitoring

---

## Project Structure

```text
lob-lstm-trading/
│
├── best_output_plots/        # Saved best-run plots (tracked)
│   ├── accuracy_curves.png
│   ├── equity_curves.png
│   ├── loss_curves.png
│   └── signals_pnl.png
│
├── config/                   # Configs / hyperparameters
│
├── cpp_engine/               # Planned C++ execution engine (redis)(in progress)
│
├── data/                     # LOBSTER CSVs: message/orderbook files
│
├── outputs/                  # Auto-generated outputs (per train outputs)(ignored in git)
│
├── src/
│   ├── __init__.py
│   ├── backtest.py           # Backtesting logic and equity calculation
│   ├── config.py             # Central config (paths, hyperparams)
│   ├── data_loader.py        # Load & parse LOBSTER CSVs
│   ├── dataset.py            # Train/val/backtest splits and utilities
│   ├── features.py           # Feature engineering (imbalance, midprice)
│   ├── model.py              # LSTM model definition
│   ├── plotting.py           # Helper functions to generate plots
│   └── train.py              # Training loop, evaluation & metrics
│
├── .env.example              # Example environment variable file
├── .gitignore
├── main.py                   # Full Python pipeline
├── pyproject.toml            # Project/packaging config
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Data & Problem Setup

- **Data source:** LOBSTER sample data for **INTC** (Intel)  
  - Message file: event timestamps & order actions  
  - Orderbook file: top L levels of bid/ask price & volume  
- **Prediction target:**  
  - Next-step **midprice movement direction** (up / down) over a short horizon  
- **Model input:**  
  - Sequences of length **100** of:
    - Order imbalance (buy vs sell pressure)
    - Midprice and related features

---

## Model & Strategy

### Model

- Architecture: **LSTM** sequence model (PyTorch)
- Input: sequences of shape `(window=100, features=…)`
- Output: 2-class logits (up / down)
- Loss: Cross-entropy
- Optimizer: Adam
- Training: ~30 epochs on one trading day of INTC LOBSTER data

### Trading Strategy (Backtest)

At each time step in the backtest window:

1. Use the LSTM to predict `P(up)` vs `P(down)`.
2. If `P(up) > 0.5` → go **long 1 unit**  
   If `P(up) ≤ 0.5` → go **short 1 unit**  
3. Mark-to-market P&L using midprice.
4. Track:
   - Equity curve of the model-driven strategy
   - “Perfect foresight” equity (upper bound using true labels)

_No transaction costs, fees, or slippage are modeled yet._

---

## Machine Learning Workflow (Step-by-Step)

1. **Load raw LOBSTER limit order book data**  
   - Message file (order events, timestamps, directions)  
   - Orderbook file (bid/ask levels)

2. **Feature engineering**
   - Compute **Order Flow Imbalance (OFI)**
   - Compute **Mid-price** at each event
   - Align and clean events (filter only execution-relevant rows)

3. **Label generation**
   - Predict **next mid-price movement** (Up / Down)

4. **Sequence construction**
   - Rolling windows of length **100** form one sample input  
   - Shape becomes: `(num_sequences, 100, num_features)`

5. **Split dataset chronologically**
   - Train → Validation → Backtest (no shuffling to prevent leakage)

6. **Train LSTM model**
   - Optimizer: Adam
   - Loss: Cross-entropy
   - Metrics: Accuracy + Loss

7. **Model evaluation**
   - Compare validation performance to detect overfitting
   - Save best plots for reporting in `best_output_plots/`

8. **Trading backtest**
   - Use model signals to long/short 1 unit per step
   - Track **equity & PnL** vs mid-price
   - Compare against **perfect foresight** benchmark

> Result: A positive-alpha ML signal that yields **+10% simulated return** on sample LOBSTER data.

---

## Results (Best Run So Far)

**Best run metrics (single-day INTC sample):**

- **Final equity (MY MODEL):** `1.1065`  → **+10.65% return**
- **Final equity (perfect foresight):** `1.0582` → **+5.82%**
- **Train accuracy (final epoch):** ≈ **82–84%**
- **Validation accuracy (final epoch):** ≈ **91%**

> The model produces a positive edge and, on this dataset, even outperforms the simple "perfect foresight" benchmark used in the example backtest.

### Plots

All plots below are from the **best run** and are stored in  
`best_output_plots/` for reproducibility.

#### Training & Validation Loss

![Training & Validation Loss](best_output_plots/loss_curves.png)

#### Training & Validation Accuracy

![Training & Validation Accuracy](best_output_plots/accuracy_curves.png)

#### Equity Curve: Model vs. Perfect Foresight

![Equity Curve](best_output_plots/equity_curves.png)

#### Signals & PnL (Example Day)

![Signals & PnL](best_output_plots/signals_pnl.png)

---

## In-Progress components inside cpp_engine/

```text
cpp_engine/
├── include/
│   └── engine/
│       ├── TradingEngine.hpp
│       └── MessageTypes.hpp
├── src/
│   ├── main.cpp               # Redis subscriber + event loop
│   └── TradingEngine.cpp      # Strategy & PnL logic
└── CMakeLists.txt
```
### Notes
- The Python LSTM + backtest pipeline is already working and produces the results shown above.
- In-progress C++/Redis

## License

MIT

