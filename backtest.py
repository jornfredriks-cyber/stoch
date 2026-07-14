import pandas as pd


def simulate_trades(
    weekly: pd.DataFrame,
    entry_level: float = 32.0,
    exit_level: float = 80.0,
) -> list[dict]:
    """
    Pure state-machine over a precomputed weekly close/k/d frame (as
    produced by stoch_scan.compute_kd). No I/O.

    Entry: first week %K crosses above entry_level (prior week %K <
    entry_level, current week %K >= entry_level). Fill at that week's
    close. %D is not checked at entry.

    Exit: first week after entry where %K < %D AND %K < exit_level are
    both true simultaneously. This holds through %K >= exit_level even
    if %K has already crossed below %D ("ride it"), and holds below
    exit_level as long as %K remains above %D. Fill at that week's close.

    A ticker can produce multiple sequential trades (Flat -> Long -> Flat
    -> ...). If still in a position when the data ends, the trade is
    recorded as "open", marked-to-market at the last available close.
    Weeks where %K or %D is NaN (rolling windows not yet full) are
    skipped entirely.
    """
    weekly = weekly.dropna(subset=["k", "d"])

    trades: list[dict] = []
    in_position = False
    entry: dict | None = None
    prev_k = None

    for dt, row in weekly.iterrows():
        k, d, close = row["k"], row["d"], row["close"]

        if not in_position:
            if prev_k is not None and prev_k < entry_level <= k:
                in_position = True
                entry = {"entry_date": dt, "entry_price": close, "entry_k": round(k, 2)}
        else:
            if k < d and k < exit_level:
                trades.append({
                    "entry_date": entry["entry_date"],
                    "entry_price": entry["entry_price"],
                    "entry_k": entry["entry_k"],
                    "exit_date": dt,
                    "exit_price": close,
                    "exit_k": round(k, 2),
                    "exit_d": round(d, 2),
                    "status": "closed",
                    "return_pct": round((close / entry["entry_price"] - 1) * 100, 2),
                    "holding_weeks": weekly.index.get_loc(dt) - weekly.index.get_loc(entry["entry_date"]),
                })
                in_position = False
                entry = None

        prev_k = k

    if in_position:
        last_date = weekly.index[-1]
        last_close = weekly["close"].iloc[-1]
        trades.append({
            "entry_date": entry["entry_date"],
            "entry_price": entry["entry_price"],
            "entry_k": entry["entry_k"],
            "exit_date": None,
            "exit_price": None,
            "exit_k": None,
            "exit_d": None,
            "status": "open",
            "return_pct": round((last_close / entry["entry_price"] - 1) * 100, 2),
            "holding_weeks": weekly.index.get_loc(last_date) - weekly.index.get_loc(entry["entry_date"]),
        })

    return trades
