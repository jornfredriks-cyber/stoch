import configparser
import os
import sys
from datetime import date

import pandas as pd

from scan_utils import _Tee, _parse_watchlist_line, fetch_ohlc_bulk, find_latest_csv
from stoch_scan import (
    STOCH_D_SMOOTH,
    STOCH_K_MAX,
    STOCH_K_MIN,
    STOCH_K_SMOOTH,
    STOCH_LENGTH,
    compute_kd,
)

DARVAS_SCREENER_DIR = "/Users/jamesblond/Documents/1-Projects/AI Trade/Darvas Phase3 Scan"

_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_bs = _cfg["backtest_signal"] if "backtest_signal" in _cfg else {}

TRADE_VALUE_USD = float(_bs.get("trade_value_usd", 1000.0))


def simulate_trades_signal(
    weekly: pd.DataFrame,
    entry_level: float = 32.0,
    exit_level: float = 80.0,
    trade_value_usd: float = TRADE_VALUE_USD,
) -> list[dict]:
    """
    Pure state-machine over a precomputed weekly close/k/d frame (as
    produced by stoch_scan.compute_kd). No I/O.

    Implements the exact entry/exit conditions of the "Stochastic Weekly
    Signal" Pine indicator (buySignal/sellSignal), not the broader
    state-based rule used by backtest.py's simulate_trades():

    Entry (green circle): %K crosses STRICTLY above entry_level (prior
    week %K < entry_level, current week %K > entry_level) AND %K > %D on
    the current week. Fill at that week's close.

    Exit (red circle): %K crosses STRICTLY below exit_level (prior week
    %K > exit_level, current week %K < exit_level) AND %K < %D on the
    current week. Fill at that week's close.

    Quirk (expected, not a bug): because exit requires an actual
    crossunder-of-exit_level event, a trade entered when %K never climbs
    back above exit_level again will never generate an exit signal under
    this literal rule -- it stays open until the ticker's data ends.

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
            if prev_k is not None and prev_k < entry_level and k > entry_level and k > d:
                in_position = True
                entry = {"entry_date": dt, "entry_price": close, "entry_k": round(k, 2)}
        else:
            if prev_k is not None and prev_k > exit_level and k < exit_level and k < d:
                dollar_pnl = round(trade_value_usd * (close / entry["entry_price"] - 1), 2)
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
                    "dollar_pnl": dollar_pnl,
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
            "dollar_pnl": round(trade_value_usd * (last_close / entry["entry_price"] - 1), 2),
            "holding_weeks": weekly.index.get_loc(last_date) - weekly.index.get_loc(entry["entry_date"]),
        })

    return trades


def load_darvas_raw_tickers() -> list[str]:
    """
    Resolves the latest "DARVAS Raw Screener_*.csv" snapshot from the
    Darvas Phase3 Scan project and returns its bare ticker list. Same
    two-line read as find_ticker_list's own CSV fallback branch, just
    pointed at the Darvas screener instead of a watchlist/Stoch source.
    """
    csv_path = find_latest_csv(DARVAS_SCREENER_DIR)
    df = pd.read_csv(csv_path)
    return df["Symbol"].dropna().tolist()


def _newest_input_file(input_dir: str) -> str | None:
    if not os.path.isdir(input_dir):
        return None
    files = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if not name.startswith(".") and os.path.isfile(os.path.join(input_dir, name))
    ]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_backtest_signal_tickers(folder: str) -> tuple[list[str], str]:
    """
    Resolves the ticker universe for run_backtest_signal(): if INPUT/
    contains any non-hidden file, use the newest one -- no filename or
    format requirement (unlike scan_utils.find_ticker_list's Watchlist/
    Tradingview-only glob), just whichever file was dropped in most
    recently. Parsed the same way as a watchlist (one ticker per line, or
    comma-separated EXCHANGE:TICKER -- _parse_watchlist_line handles both).
    Falls back to load_darvas_raw_tickers() if INPUT/ has no files, same
    as this function's behavior before INPUT/ support was added.

    Returns (tickers, source_label) so the caller can log which universe
    was actually used.
    """
    input_dir = os.path.join(folder, "INPUT")
    newest = _newest_input_file(input_dir)
    if newest is None:
        return load_darvas_raw_tickers(), "Darvas Raw screener"

    with open(newest) as f:
        content = f.read()
    tickers = []
    for line in content.splitlines():
        tickers.extend(_parse_watchlist_line(line))
    return tickers, f"INPUT/{os.path.basename(newest)}"


def run_backtest_signal(folder: str):
    output_dir = os.path.join(folder, "OUTPUT")

    tickers, source = load_backtest_signal_tickers(folder)
    tickers = list(dict.fromkeys(tickers))
    print(f"Tickers loaded : {len(tickers)}  ({source})")
    print(f"Weekly Stoch   : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  Entry: K crosses >{STOCH_K_MIN} & K>D  |  Exit: K crosses <{STOCH_K_MAX} & K<D")
    print(f"Trade value    : ${TRADE_VALUE_USD:.2f} per trade, no stop-loss/take-profit\n")

    yf_symbols = [t.replace(".", "-").replace("/", "-") for t in tickers]
    # auto_adjust=False: split-adjusted only, not dividend-adjusted -- matches
    # TradingView's default unadjusted chart much more closely than the fully
    # adjusted prices backtest.py/stoch_scan.py use. See scan_utils.fetch_ohlc_bulk.
    ohlc = fetch_ohlc_bulk(yf_symbols, period="max", auto_adjust=False)

    all_trades = []
    errors = 0
    for i, (sym, yf_sym) in enumerate(zip(tickers, yf_symbols), 1):
        try:
            df = ohlc.get(yf_sym)
            if df is None:
                print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  — no data")
                continue

            kd = compute_kd(df, STOCH_LENGTH, STOCH_K_SMOOTH, STOCH_D_SMOOTH)
            trades = simulate_trades_signal(kd, entry_level=STOCH_K_MIN, exit_level=STOCH_K_MAX)
            for t in trades:
                t["ticker"] = sym
            all_trades.extend(trades)

            closed = sum(1 for t in trades if t["status"] == "closed")
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  {len(trades)} trades ({closed} closed)")

        except Exception as exc:
            errors += 1
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  ERROR: {exc}")

    out_name = f"StochSignal_Backtest_Trades_{date.today()}.csv"
    out_path = os.path.join(output_dir, out_name)
    columns = ["ticker", "entry_date", "entry_price", "entry_k",
               "exit_date", "exit_price", "exit_k", "exit_d",
               "status", "return_pct", "dollar_pnl", "holding_weeks"]
    pd.DataFrame(all_trades, columns=columns).to_csv(out_path, index=False)

    closed_trades = [t for t in all_trades if t["status"] == "closed"]
    open_trades   = [t for t in all_trades if t["status"] == "open"]
    wins   = [t for t in closed_trades if t["return_pct"] > 0]
    losses = [t for t in closed_trades if t["return_pct"] <= 0]

    print(f"\n{'=' * 55}")
    print(f"Total trades  : {len(all_trades)}")
    print(f"Errors        : {errors}")
    print(f"Closed / Open : {len(closed_trades)} / {len(open_trades)}"
          f"  (open = never crossed back above {STOCH_K_MAX} after entry, expected under this rule)")
    if closed_trades:
        print(f"Win rate      : {len(wins) / len(closed_trades) * 100:.1f}%  ({len(wins)}/{len(closed_trades)})")
    if wins:
        print(f"Avg win       : {sum(t['return_pct'] for t in wins) / len(wins):.2f}%"
              f"   (${sum(t['dollar_pnl'] for t in wins) / len(wins):.2f})")
    if losses:
        print(f"Avg loss      : {sum(t['return_pct'] for t in losses) / len(losses):.2f}%"
              f"   (${sum(t['dollar_pnl'] for t in losses) / len(losses):.2f})")
    if closed_trades:
        avg_hold = sum(t["holding_weeks"] for t in closed_trades) / len(closed_trades)
        print(f"Avg hold (wk) : {avg_hold:.1f}")
        total_pnl = sum(t["dollar_pnl"] for t in closed_trades)
        print(f"Total $ P&L   : ${total_pnl:.2f}  (closed trades only, ${TRADE_VALUE_USD:.0f}/trade)")
    print(f"Saved → OUTPUT/{out_name}")


def main():
    folder     = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(folder, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)
    gitkeep = os.path.join(output_dir, ".gitkeep")
    if not os.path.exists(gitkeep):
        open(gitkeep, "a").close()

    log_path = os.path.join(output_dir, f"stoch_signal_backtest_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        run_backtest_signal(folder)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/{os.path.basename(log_path)}")


if __name__ == "__main__":
    main()
