import configparser
import os
import sys
from datetime import date

import pandas as pd

from backtest_signal import _newest_input_file
from scan_utils import _Tee, _parse_watchlist_line, fetch_ohlc_bulk
from stoch_scan import STOCH_D_SMOOTH, STOCH_K_SMOOTH, STOCH_LENGTH, compute_kd, compute_kd_daily

_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_bt = _cfg["backtest_basic_trail"] if "backtest_basic_trail" in _cfg else {}

ENTRY_LEVEL         = float(_bt.get("entry_level",         32.0))
TRAIL_PCT           = float(_bt.get("trail_pct",           20.0))
CONFIRM_DAYS        = int(  _bt.get("confirm_days",         0))
DAILY_STOCH_LENGTH  = int(  _bt.get("daily_stoch_length",  14))
DAILY_STOCH_K_SMOOTH = int( _bt.get("daily_stoch_k_smooth", 3))
DAILY_STOCH_D_SMOOTH = int( _bt.get("daily_stoch_d_smooth", 3))


def _find_price_touch(daily_slice: pd.DataFrame, price: float):
    """
    Scans daily bars in chronological order for the first day whose low
    breaches `price`. Fills at that price itself, not the day's actual low
    -- same idealized-fill assumption backtest_signal_sltp_sweep.py's
    SL/TP touch-check uses (raw OHLC can't tell us the true intraday
    sequencing or gap depth). Dual-purpose: used both for the exit
    trailing-stop touch (a sell-stop) and the entry limit-order touch (a
    buy-limit) -- mechanically identical check, "did price trade down to
    this level," just interpreted differently by the caller. Returns
    (touch_date, price), or None if the slice has no touch.
    """
    for dt, day in daily_slice.iterrows():
        if day["low"] <= price:
            return dt, price
    return None


def _daily_kd_at_confirm_day(daily_kd: pd.DataFrame, signal_date, confirm_days: int):
    """
    Returns (day_N_date, k, d) for the confirm_days-th trading day strictly
    after signal_date, or None if fewer than confirm_days trading days
    remain in the data, or that day's daily %K/%D isn't available yet
    (rolling window not full -- signal too close to the start of history).
    """
    future = daily_kd.loc[daily_kd.index > signal_date]
    if len(future) < confirm_days:
        return None
    k_n = future["k"].iloc[confirm_days - 1]
    d_n = future["d"].iloc[confirm_days - 1]
    if pd.isna(k_n) or pd.isna(d_n):
        return None
    return future.index[confirm_days - 1], k_n, d_n


def _next_day_open(daily: pd.DataFrame, day_n_date):
    """
    Returns (date, open_price) for the single trading day immediately
    after day_n_date, or None if none exists (day_n_date is the last bar
    in the data).
    """
    future = daily.loc[daily.index > day_n_date]
    if future.empty:
        return None
    return future.index[0], float(future["open"].iloc[0])


def simulate_trades_basic_trail(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    entry_level: float = ENTRY_LEVEL,
    trail_pct: float = TRAIL_PCT,
    confirm_days: int = CONFIRM_DAYS,
    daily_kd: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Pure state-machine. `weekly` is a precomputed close/k/d frame (as
    produced by stoch_scan.compute_kd); `daily` is the raw OHLC frame
    (needs "open", "close", "low" columns) covering the same period;
    `daily_kd` is a precomputed daily close/k/d frame (as produced by
    stoch_scan.compute_kd_daily) -- only required when confirm_days > 0.
    No I/O.

    Signal: identical to backtest_signal.simulate_trades_signal()'s buy leg
    -- %K crosses STRICTLY above entry_level (prior week %K < entry_level,
    current week %K > entry_level) AND %K > %D on the current week.

    Entry, if confirm_days == 0 (default): immediate, at the signal week's
    close -- original behavior, unchanged.

    Entry, if confirm_days > 0 (added 2026-08-27, user request -- filter
    out signals that immediately reverse; revised same day from an earlier
    price-close version that barely discriminated in practice, to a
    momentum-based one that keeps the same wait length): a two-stage
    confirm-then-fill process:
      1. CONFIRM: look confirm_days TRADING days past the signal date
         ("day N"). If day N's DAILY %K is above day N's DAILY %D (an
         independent, faster oscillator from the weekly one used for the
         signal itself -- see daily_kd), confirmation passes. If not (or
         day N's daily %K/%D isn't available yet, or fewer than
         confirm_days trading days remain in the data), the signal is
         skipped entirely -- no retry.
      2. FILL: a plain market buy at day N+1's OPEN -- the earliest
         realistic fill after a decision made off day N's close. Always
         fills if day N+1 exists in the data (no rejection case here,
         unlike the earlier limit-order version -- there's no price level
         that can fail to be touched). If day N+1 doesn't exist (signal
         too close to the end of history), the signal is skipped.
    entry_k always records the signal week's %K for context, regardless
    of how much later the actual fill happens. Deliberately no volume
    gate (2-3 days is too small/noisy a sample to trust one). Only
    validated for small confirm_days values landing within the single
    following calendar week.

    Exit: the ONLY exit is a trail_pct% trailing stop off the highest
    weekly CLOSE seen since the ACTUAL entry (not the signal date) -- no
    signal exit, no take-profit, no time limit. Initial stop =
    entry_price * (1 - trail_pct/100).

    Checked against each DAILY LOW within the week (2026-08-27 revision --
    the original weekly-close-only check let a single bad week's close
    land arbitrarily far below the nominal stop, and separately could MISS
    a real intraweek touch entirely if price recovered by that Friday's
    close). Fills at the stop price itself the moment a day's low touches
    it, same technique as backtest_signal_sltp_sweep.py's SL/TP check.

    Sequencing, to avoid look-ahead: each week's daily lows are checked
    against the stop as it stood at the END OF THE PRIOR week -- never a
    stop this week's own (not-yet-known-until-Friday) close would newly
    justify raising it to. Only after a week survives with no touch does
    its close get folded into the ratchet for the following week's check.

    A ticker can produce multiple sequential trades. If still in a
    position when the data ends, the trade is recorded as "open", marked-
    to-market at the last available DAILY close (not weekly). Weeks where
    %K or %D is NaN (rolling windows not yet full) are skipped from
    iteration, but their daily bars are still covered by the touch-check
    window (date-range based, not row-count based) -- no blind spot.
    holding_days (not holding_weeks) since exits can now land mid-week.
    """
    weekly = weekly.dropna(subset=["k", "d"])

    trades: list[dict] = []
    in_position = False
    entry: dict | None = None
    highest_close = None
    stop = None
    prev_k = None
    prev_week_end = None

    for dt, row in weekly.iterrows():
        k, d, close = row["k"], row["d"], row["close"]
        next_prev_week_end = dt  # default; overridden below on a confirmed deferred entry

        if not in_position:
            if prev_k is not None and prev_k < entry_level and k > entry_level and k > d:
                if confirm_days > 0:
                    result = _daily_kd_at_confirm_day(daily_kd, dt, confirm_days)
                    if result is not None:
                        day_n_date, k_n, d_n = result
                        if k_n > d_n:
                            # Confirmed -> a plain market buy at the next
                            # trading day's open, the earliest realistic
                            # fill after a decision made off day N's close.
                            fill = _next_day_open(daily, day_n_date)
                            if fill is not None:
                                fill_date, fill_price = fill
                                in_position = True
                                entry = {"entry_date": fill_date, "entry_price": fill_price, "entry_k": round(k, 2)}
                                highest_close = fill_price
                                stop = fill_price * (1 - trail_pct / 100)
                                next_prev_week_end = fill_date
                            # else: no next trading day -- skip, no retry
                        # else: daily %K <= %D on day N -- not confirmed, skip
                    # else: not enough future data to even reach day N, or its daily k/d isn't available -- skip
                else:
                    in_position = True
                    entry = {"entry_date": dt, "entry_price": close, "entry_k": round(k, 2)}
                    highest_close = close
                    stop = close * (1 - trail_pct / 100)
        else:
            window = (
                daily.loc[(daily.index > prev_week_end) & (daily.index <= dt)]
                if prev_week_end is not None
                else daily.iloc[0:0]
            )
            touch = _find_price_touch(window, stop)
            if touch is not None:
                exit_date, exit_price = touch
                trades.append({
                    "entry_date": entry["entry_date"],
                    "entry_price": entry["entry_price"],
                    "entry_k": entry["entry_k"],
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "stop_price": round(stop, 4),
                    "status": "closed",
                    "return_pct": round((exit_price / entry["entry_price"] - 1) * 100, 2),
                    "holding_days": (pd.Timestamp(exit_date) - pd.Timestamp(entry["entry_date"])).days,
                })
                in_position = False
                entry = None
                highest_close = None
                stop = None
            else:
                highest_close = max(highest_close, close)
                stop = max(stop, highest_close * (1 - trail_pct / 100))

        prev_k = k
        prev_week_end = next_prev_week_end

    if in_position:
        last_date = daily.index[-1]
        last_close = float(daily["close"].iloc[-1])
        trades.append({
            "entry_date": entry["entry_date"],
            "entry_price": entry["entry_price"],
            "entry_k": entry["entry_k"],
            "exit_date": None,
            "exit_price": None,
            "stop_price": round(stop, 4),
            "status": "open",
            "return_pct": round((last_close / entry["entry_price"] - 1) * 100, 2),
            "holding_days": (pd.Timestamp(last_date) - pd.Timestamp(entry["entry_date"])).days,
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
          f"  |  Entry: K crosses >{ENTRY_LEVEL} & K>D"
          + (f", confirmed {CONFIRM_DAYS}d later (daily K>D), filled at next open" if CONFIRM_DAYS > 0 else ", immediate")
          + f"\n{'':17s}|  Exit: {TRAIL_PCT:.0f}% trailing stop only, checked intraweek (daily lows)\n")

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
            kd_daily = (
                compute_kd_daily(df, DAILY_STOCH_LENGTH, DAILY_STOCH_K_SMOOTH, DAILY_STOCH_D_SMOOTH)
                if CONFIRM_DAYS > 0 else None
            )
            trades = simulate_trades_basic_trail(
                df, kd, entry_level=ENTRY_LEVEL, trail_pct=TRAIL_PCT,
                confirm_days=CONFIRM_DAYS, daily_kd=kd_daily,
            )
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
               "status", "return_pct", "holding_days"]
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
        avg_hold = sum(t["holding_days"] for t in closed_trades) / len(closed_trades)
        print(f"Avg hold (days): {avg_hold:.1f}")
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
