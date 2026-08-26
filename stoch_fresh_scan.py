import configparser
import os
import sys
import time
from datetime import date

import pandas as pd

from scan_utils import _Tee, fetch_ohlc_bulk, fetch_ohlc_single, find_latest_csv
from stoch_scan import (
    STOCH_D_SMOOTH,
    STOCH_K_MIN,
    STOCH_K_SMOOTH,
    STOCH_LENGTH,
    compute_kd,
)

# ── Config ────────────────────────────────────────────────────────────────────
_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_f = _cfg["stoch_fresh_scan"] if "stoch_fresh_scan" in _cfg else {}

LOOKBACK_WEEKS = int(  _f.get("lookback_weeks", 2))
ENTRY_LEVEL    = float(_f.get("entry_level",    STOCH_K_MIN))
FETCH_CHUNK_SIZE = 50


def is_fresh_stoch_buy_within_lookback(
    df: pd.DataFrame,
    stoch_length: int = STOCH_LENGTH,
    k_smooth: int = STOCH_K_SMOOTH,
    d_smooth: int = STOCH_D_SMOOTH,
    entry_level: float = ENTRY_LEVEL,
    lookback_weeks: int = LOOKBACK_WEEKS,
) -> tuple:
    """
    Checks whether a stock's Weekly Stochastic %K crossed STRICTLY above
    entry_level (prior completed week's %K < entry_level, that week's %K >
    entry_level, %K > %D on that week) at any point within the last
    `lookback_weeks` FULLY COMPLETED weekly bars. The current, possibly
    still-forming week is always dropped before any check runs -- never
    eligible, unlike stoch_signal_scan.is_fresh_stoch_signal_buy(), which
    deliberately does use it.

    The most-recently-completed qualifying week is returned if more than
    one week in the window qualifies.

    Returns:
        (True,  k_value, d_value, entry_date, entry_price, weeks_ago)
        (False, None,    None,    None,       None,        None)
    """
    kd = compute_kd(df, stoch_length, k_smooth, d_smooth)

    # Drop the current (possibly still-forming) week FIRST, from the raw kd
    # frame -- before dropna(). Dropping after dropna() would be wrong
    # whenever the live week's k/d happens to be NaN (e.g. thin history):
    # .iloc[:-1] on a NaN-dropped frame would then strip a genuinely
    # completed week instead of the live one. Slicing first guarantees
    # exactly one bar (the live week) is excluded, always.
    completed = kd.iloc[:-1].dropna(subset=["k", "d"])

    if len(completed) < lookback_weeks + 1:
        return False, None, None, None, None, None

    for weeks_ago in range(1, lookback_weeks + 1):
        k_now  = completed["k"].iloc[-weeks_ago]
        d_now  = completed["d"].iloc[-weeks_ago]
        k_prev = completed["k"].iloc[-weeks_ago - 1]

        if (k_prev < entry_level) and (k_now > entry_level) and (k_now > d_now):
            entry_date  = completed.index[-weeks_ago]
            entry_price = float(completed["close"].iloc[-weeks_ago])
            return True, round(float(k_now), 2), round(float(d_now), 2), entry_date, entry_price, weeks_ago

    return False, None, None, None, None, None


def main():
    folder     = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(folder, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"stoch_fresh_scan_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    screener_dir = os.path.join(folder, "OUTPUT", "Screener")

    try:
        _run(screener_dir, output_dir)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/{os.path.basename(log_path)}")


def _run(screener_dir: str, output_dir: str):
    csv_path = find_latest_csv(screener_dir)
    print(f"Screener file : {os.path.basename(csv_path)}")

    tv      = pd.read_csv(csv_path)
    symbols = tv["Symbol"].dropna().tolist()
    print(f"Tickers loaded: {len(symbols)}")
    print(f"Weekly Stoch  : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  Fresh buy: %K crosses strictly above {ENTRY_LEVEL} within the last"
          f" {LOOKBACK_WEEKS} completed weekly bar(s), %K>%D on that bar\n")

    symbol_pairs = [
        (sym, sym.replace(".", "-").replace("/", "-"))
        for sym in symbols if isinstance(sym, str)
    ]

    # auto_adjust=False: yfinance's dividend-adjustment (the auto_adjust=True
    # default) retroactively rescales the whole price history, which shifts
    # the weekly Stochastic's rolling min/max window enough to smear a real
    # crossover across two weekly bars for dividend payers (confirmed on
    # EQNR, 2026-07-30 -- see Open Threads) -- same fix already applied in
    # stoch_signal_scan.py/backtest_signal.py, needed here independently of
    # any EMA-gate rationale since this scan's whole purpose is precise
    # crossover-timing detection.
    ohlc = fetch_ohlc_bulk(
        [yf for _, yf in symbol_pairs],
        chunk_size=FETCH_CHUNK_SIZE,
        auto_adjust=False,
    )

    failed = [yf for _, yf in symbol_pairs if yf not in ohlc]
    if failed:
        print(f"Retrying {len(failed)} failed symbols one-by-one…")
        recovered = 0
        for yf_sym in failed:
            time.sleep(0.5)
            df = fetch_ohlc_single(yf_sym)
            if df is not None:
                ohlc[yf_sym] = df
                recovered += 1
        print(f"Recovered {recovered}/{len(failed)} on retry\n")

    candidates = []
    errors     = 0
    for i, (sym, yf_sym) in enumerate(symbol_pairs, 1):
        try:
            df = ohlc.get(yf_sym)
            if df is None:
                print(f"  [{i:3d}/{len(symbols)}] {sym:10s}  — no data")
                continue

            fresh, k, d, entry_date, price, weeks_ago = is_fresh_stoch_buy_within_lookback(df)
            if not fresh:
                print(f"  [{i:3d}/{len(symbols)}] {sym:10s}  —")
                continue

            candidates.append({
                "ticker": sym,
                "weeks_ago": weeks_ago,
                "entry_date": entry_date.date(),
                "entry_k": k,
                "entry_d": d,
                "price": round(price, 2),
            })
            print(f"  [{i:3d}/{len(symbols)}] {sym:10s}"
                  f"  BUY  {weeks_ago}w ago  K:{k:.1f}  D:{d:.1f}  price:{price:.2f}")

        except Exception as exc:
            errors += 1
            print(f"  [{i:3d}/{len(symbols)}] {sym:10s}  ERROR: {exc}")

    print(f"\n{'=' * 55}")
    print(f"Fresh buy candidates : {len(candidates)}")
    print(f"Errors               : {errors}")

    out_name = f"StochFresh_Candidates_{date.today()}.csv"
    out_path = os.path.join(output_dir, out_name)
    columns = ["ticker", "weeks_ago", "entry_date", "entry_k", "entry_d", "price"]
    pd.DataFrame(candidates, columns=columns).to_csv(out_path, index=False)
    print(f"Saved → OUTPUT/{out_name}")


if __name__ == "__main__":
    main()
