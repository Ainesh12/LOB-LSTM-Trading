import pandas as pd
from .config import DATA_DIR, MSG_FILENAME, LOB_FILENAME

def load_lobster_data():
    msg_path = DATA_DIR / MSG_FILENAME
    lob_path = DATA_DIR / LOB_FILENAME

    msg_cols = ["Time", "Type", "OrderID", "Size", "Price", "Direction"]
    msg = pd.read_csv(msg_path, header=None, names=msg_cols)

    lob_cols = []
    for lvl in range(1, 11):
        lob_cols += [
            f"AskPrice{lvl}", f"AskSize{lvl}",
            f"BidPrice{lvl}", f"BidSize{lvl}",
        ]
    lob = pd.read_csv(lob_path, header=None, names=lob_cols)

    msg = msg.reset_index(drop=True)
    lob = lob.reset_index(drop=True)

    n = min(len(msg), len(lob))
    msg = msg.iloc[:n]
    lob = lob.iloc[:n]

    df = pd.concat([msg[["Time", "Type"]], lob], axis=1)

    df["Time"] = df["Time"].astype(float)
    df["Type"] = df["Type"].astype(int)

    return df
