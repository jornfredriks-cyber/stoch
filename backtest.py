import os
import sys
from datetime import date

import pandas as pd

from scan_utils import _Tee, fetch_ohlc_bulk, find_ticker_list
from stoch_scan import (
    STOCH_D_SMOOTH,
    STOCH_K_MAX,
    STOCH_K_MIN,
    STOCH_K_SMOOTH,
    STOCH_LENGTH,
    compute_kd,
)


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


def run_backtest(folder: str):
    input_dir = os.path.join(folder, "INPUT")
    output_dir = os.path.join(folder, "OUTPUT")
    screener_dir = os.path.join(folder, "OUTPUT", "Screener")

    tickers = find_ticker_list(input_dir, screener_dir)
    tickers = list(dict.fromkeys(tickers))
    print(f"Tickers loaded : {len(tickers)}")
    print(f"Weekly Stoch   : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  Entry K>={STOCH_K_MIN}  |  Exit K<D and K<{STOCH_K_MAX}\n")

    yf_symbols = [t.replace(".", "-").replace("/", "-") for t in tickers]
    ohlc = fetch_ohlc_bulk(yf_symbols, period="max")

    all_trades = []
    errors = 0
    for i, (sym, yf_sym) in enumerate(zip(tickers, yf_symbols), 1):
        try:
            df = ohlc.get(yf_sym)
            if df is None:
                print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  — no data")
                continue

            kd = compute_kd(df, STOCH_LENGTH, STOCH_K_SMOOTH, STOCH_D_SMOOTH)
            trades = simulate_trades(kd, entry_level=STOCH_K_MIN, exit_level=STOCH_K_MAX)
            for t in trades:
                t["ticker"] = sym
            all_trades.extend(trades)

            closed = sum(1 for t in trades if t["status"] == "closed")
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  {len(trades)} trades ({closed} closed)")

        except Exception as exc:
            errors += 1
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  ERROR: {exc}")

    out_name = f"Stoch_Backtest_Trades_{date.today()}.csv"
    out_path = os.path.join(output_dir, out_name)
    columns = ["ticker", "entry_date", "entry_price", "entry_k",
               "exit_date", "exit_price", "exit_k", "exit_d",
               "status", "return_pct", "holding_weeks"]
    pd.DataFrame(all_trades, columns=columns).to_csv(out_path, index=False)

    closed_trades = [t for t in all_trades if t["status"] == "closed"]
    open_trades   = [t for t in all_trades if t["status"] == "open"]
    wins   = [t for t in closed_trades if t["return_pct"] > 0]
    losses = [t for t in closed_trades if t["return_pct"] <= 0]

    print(f"\n{'=' * 55}")
    print(f"Total trades  : {len(all_trades)}")
    print(f"Errors        : {errors}")
    print(f"Closed / Open : {len(closed_trades)} / {len(open_trades)}")
    if closed_trades:
        print(f"Win rate      : {len(wins) / len(closed_trades) * 100:.1f}%  ({len(wins)}/{len(closed_trades)})")
    if wins:
        print(f"Avg win       : {sum(t['return_pct'] for t in wins) / len(wins):.2f}%")
    if losses:
        print(f"Avg loss      : {sum(t['return_pct'] for t in losses) / len(losses):.2f}%")
    if closed_trades:
        avg_hold = sum(t["holding_weeks"] for t in closed_trades) / len(closed_trades)
        print(f"Avg hold (wk) : {avg_hold:.1f}")
    print(f"Saved → OUTPUT/{out_name}")


def main():
    folder     = os.path.dirname(os.path.abspath(__file__))
    input_dir  = os.path.join(folder, "INPUT")
    output_dir = os.path.join(folder, "OUTPUT")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    for d in (input_dir, output_dir):
        gitkeep = os.path.join(d, ".gitkeep")
        if not os.path.exists(gitkeep):
            open(gitkeep, "a").close()

    log_path = os.path.join(output_dir, f"stoch_backtest_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        run_backtest(folder)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/{os.path.basename(log_path)}")


if __name__ == "__main__":
    main()
