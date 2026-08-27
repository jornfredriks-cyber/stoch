import configparser
import os
import sys
from datetime import date

import pandas as pd

from backtest_signal import _newest_input_file
from scan_utils import _Tee, _parse_watchlist_line, fetch_ohlc_bulk
from stoch_scan import STOCH_D_SMOOTH, STOCH_K_SMOOTH, STOCH_LENGTH, compute_kd

_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_bt = _cfg["backtest_basic_trail"] if "backtest_basic_trail" in _cfg else {}

ENTRY_LEVEL = float(_bt.get("entry_level", 32.0))
TRAIL_PCT   = float(_bt.get("trail_pct",   20.0))


def simulate_trades_basic_trail(
    weekly: pd.DataFrame,
    entry_level: float = ENTRY_LEVEL,
    trail_pct: float = TRAIL_PCT,
) -> list[dict]:
    """
    Pure state-machine over a precomputed weekly close/k/d frame (as
    produced by stoch_scan.compute_kd). No I/O.

    Entry: identical to backtest_signal.simulate_trades_signal()'s buy leg
    -- %K crosses STRICTLY above entry_level (prior week %K < entry_level,
    current week %K > entry_level) AND %K > %D on the current week. Fill at
    that week's close.

    Exit: the ONLY exit is a trail_pct% trailing stop off the highest
    weekly CLOSE seen since entry -- no signal exit, no take-profit, no
    time limit (per explicit user decision, 2026-08-27). Initial stop =
    entry_price * (1 - trail_pct/100); each subsequent week the stop is
    recalculated off the new highest close and only ever moves up. Checked
    at each weekly close, not daily intraweek highs/lows -- a position can
    therefore fill BELOW the stop level itself if the close gaps past it
    (see stop_price vs exit_price in the returned dict).

    A ticker can produce multiple sequential trades. If still in a
    position when the data ends, the trade is recorded as "open", marked-
    to-market at the last available close. Weeks where %K or %D is NaN
    (rolling windows not yet full) are skipped entirely.
    """
    weekly = weekly.dropna(subset=["k", "d"])

    trades: list[dict] = []
    in_position = False
    entry: dict | None = None
    highest_close = None
    stop = None
    prev_k = None

    for dt, row in weekly.iterrows():
        k, d, close = row["k"], row["d"], row["close"]

        if not in_position:
            if prev_k is not None and prev_k < entry_level and k > entry_level and k > d:
                in_position = True
                entry = {"entry_date": dt, "entry_price": close, "entry_k": round(k, 2)}
                highest_close = close
                stop = close * (1 - trail_pct / 100)
        else:
            highest_close = max(highest_close, close)
            stop = max(stop, highest_close * (1 - trail_pct / 100))
            if close <= stop:
                trades.append({
                    "entry_date": entry["entry_date"],
                    "entry_price": entry["entry_price"],
                    "entry_k": entry["entry_k"],
                    "exit_date": dt,
                    "exit_price": close,
                    "stop_price": round(stop, 4),
                    "status": "closed",
                    "return_pct": round((close / entry["entry_price"] - 1) * 100, 2),
                    "holding_weeks": weekly.index.get_loc(dt) - weekly.index.get_loc(entry["entry_date"]),
                })
                in_position = False
                entry = None
                highest_close = None
                stop = None

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
            "stop_price": round(stop, 4),
            "status": "open",
            "return_pct": round((last_close / entry["entry_price"] - 1) * 100, 2),
            "holding_weeks": weekly.index.get_loc(last_date) - weekly.index.get_loc(entry["entry_date"]),
        })

    return trades


def load_basic_trail_tickers(folder: str) -> tuple[list[str], str]:
    """
    Resolves the ticker universe: newest file in INPUT/ if present (any
    name/format, same resolver as backtest_signal.load_backtest_signal_tickers).
    If INPUT/ is empty, falls back to the newest file in OUTPUT/BasicScan/
    (the Basic Scanner's own last output) -- NOT the Darvas Raw screener,
    since silently testing against a completely different, uptrend-biased
    universe would defeat the point of "based on this scanner" (explicit
    user decision, 2026-08-27). Raises if neither has anything.
    """
    input_dir = os.path.join(folder, "INPUT")
    newest = _newest_input_file(input_dir)
    if newest is not None:
        source_label = f"INPUT/{os.path.basename(newest)}"
    else:
        basic_scan_dir = os.path.join(folder, "OUTPUT", "BasicScan")
        newest = _newest_input_file(basic_scan_dir)
        if newest is None:
            raise FileNotFoundError(
                "No ticker source found: INPUT/ is empty and OUTPUT/BasicScan/ has "
                "no scan output yet. Run stoch_basic_screener.py && stoch_basic_scan.py "
                "first, or drop a ticker list into INPUT/."
            )
        source_label = f"OUTPUT/BasicScan/{os.path.basename(newest)}"

    with open(newest) as f:
        content = f.read()
    tickers = []
    for line in content.splitlines():
        tickers.extend(_parse_watchlist_line(line))
    return tickers, source_label


def run_backtest_basic_trail(folder: str):
    output_dir = os.path.join(folder, "OUTPUT")

    tickers, source = load_basic_trail_tickers(folder)
    tickers = list(dict.fromkeys(tickers))
    print(f"Tickers loaded : {len(tickers)}  ({source})")
    print(f"Weekly Stoch   : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  Entry: K crosses >{ENTRY_LEVEL} & K>D  |  Exit: {TRAIL_PCT:.0f}% trailing stop only\n")

    yf_symbols = [t.replace(".", "-").replace("/", "-") for t in tickers]
    # auto_adjust=False: same rationale as backtest_signal.py -- this reuses
    # that exact entry rule, so the same dividend-adjustment timing
    # distortion applies (see backtest_signal.py's docstring/CLAUDE.md).
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
            trades = simulate_trades_basic_trail(kd, entry_level=ENTRY_LEVEL, trail_pct=TRAIL_PCT)
            for t in trades:
                t["ticker"] = sym
            all_trades.extend(trades)

            closed = sum(1 for t in trades if t["status"] == "closed")
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  {len(trades)} trades ({closed} closed)")

        except Exception as exc:
            errors += 1
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  ERROR: {exc}")

    out_name = f"BasicTrail_Backtest_Trades_{date.today()}.csv"
    out_path = os.path.join(output_dir, out_name)
    columns = ["ticker", "entry_date", "entry_price", "entry_k",
               "exit_date", "exit_price", "stop_price",
               "status", "return_pct", "holding_weeks"]
    pd.DataFrame(all_trades, columns=columns).to_csv(out_path, index=False)

    closed_trades = [t for t in all_trades if t["status"] == "closed"]
    open_trades   = [t for t in all_trades if t["status"] == "open"]
    wins   = [t for t in closed_trades if t["return_pct"] > 0]
    losses = [t for t in closed_trades if t["return_pct"] <= 0]

    print(f"\n{'=' * 55}")
    print(f"Total trades   : {len(all_trades)}")
    print(f"Errors         : {errors}")
    print(f"Closed / Open  : {len(closed_trades)} / {len(open_trades)}")
    if closed_trades:
        print(f"Stopped out    : {len(losses)}   |   Won : {len(wins)}"
              f"   (win rate {len(wins) / len(closed_trades) * 100:.1f}%)")
    if wins:
        win_returns = [t["return_pct"] for t in wins]
        print(f"Won   % above entry  — min: {min(win_returns):.2f}%"
              f"  max: {max(win_returns):.2f}%"
              f"  avg: {sum(win_returns) / len(win_returns):.2f}%")
    if losses:
        loss_returns = [t["return_pct"] for t in losses]
        print(f"Lost  % vs entry     — min: {min(loss_returns):.2f}%"
              f"  max: {max(loss_returns):.2f}%"
              f"  avg: {sum(loss_returns) / len(loss_returns):.2f}%")
    if closed_trades:
        avg_hold = sum(t["holding_weeks"] for t in closed_trades) / len(closed_trades)
        print(f"Avg hold (wk)  : {avg_hold:.1f}")
    print(f"Saved → OUTPUT/{out_name}")


def main():
    folder     = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(folder, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)
    gitkeep = os.path.join(output_dir, ".gitkeep")
    if not os.path.exists(gitkeep):
        open(gitkeep, "a").close()

    log_path = os.path.join(output_dir, f"backtest_basic_trail_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        run_backtest_basic_trail(folder)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/{os.path.basename(log_path)}")


if __name__ == "__main__":
    main()
