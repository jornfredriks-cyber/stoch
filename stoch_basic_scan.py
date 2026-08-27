import configparser
import os
import sys
import time
from datetime import date

import pandas as pd

from scan_utils import _Tee, fetch_ohlc_bulk, fetch_ohlc_single, find_latest_csv
from stoch_scan import (
    STOCH_D_SMOOTH,
    STOCH_K_SMOOTH,
    STOCH_LENGTH,
    compute_kd,
)

# ── Config ────────────────────────────────────────────────────────────────────
_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_b = _cfg["stoch_basic_scan"] if "stoch_basic_scan" in _cfg else {}

K_CEILING        = float(_b.get("k_ceiling", 32.0))
FETCH_CHUNK_SIZE = 50


def is_stoch_basic_candidate(
    df: pd.DataFrame,
    stoch_length: int = STOCH_LENGTH,
    k_smooth: int = STOCH_K_SMOOTH,
    d_smooth: int = STOCH_D_SMOOTH,
    k_ceiling: float = K_CEILING,
) -> tuple:
    """
    Checks whether a stock's Weekly Stochastic (length/%K-smooth/%D-smooth,
    default 19/4/4) has its latest %K value strictly above %D and strictly
    below k_ceiling (default 32.0) -- the window before a sweet-spot/signal
    buy trigger, not the trigger itself. Pure snapshot of the latest bar,
    no "rising" requirement (unlike is_stoch_sweet_spot()) -- deliberately
    simple per the Basic Scanner's design (2026-08-27).

    Returns:
        (True,  k_value, d_value)  — latest %K above %D and below k_ceiling
        (False, None,    None)     — condition not met, or insufficient
                                       weekly history to compute both lines
    """
    kd = compute_kd(df, stoch_length, k_smooth, d_smooth)

    valid = kd["d"].dropna()
    if len(valid) < 1:
        return False, None, None

    k_now = kd["k"].loc[valid.index[-1]]
    d_now = valid.iloc[-1]

    qualifies = (k_now > d_now) and (k_now < k_ceiling)

    if qualifies:
        return True, round(k_now, 2), round(d_now, 2)

    return False, None, None


def main():
    folder       = os.path.dirname(os.path.abspath(__file__))
    screener_dir = os.path.join(folder, "OUTPUT", "BasicScreener")
    output_dir   = os.path.join(folder, "OUTPUT", "BasicScan")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"stoch_basic_scan_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        _run(screener_dir, output_dir)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/BasicScan/{os.path.basename(log_path)}")


def _run(screener_dir: str, output_dir: str):
    csv_path = find_latest_csv(screener_dir)
    print(f"Screener file : {os.path.basename(csv_path)}")

    tv      = pd.read_csv(csv_path)
    symbols = tv["Symbol"].dropna().tolist()
    print(f"Tickers loaded: {len(symbols)}")
    print(f"Weekly Stoch  : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  Basic candidate: %K>%D and %K<{K_CEILING}")
    print(f"Fetch mode    : bulk chunks of {FETCH_CHUNK_SIZE}\n")

    candidates   = []
    errors       = 0
    symbol_pairs = [
        (sym, sym.replace(".", "-").replace("/", "-"))
        for sym in symbols if isinstance(sym, str)
    ]

    ohlc = fetch_ohlc_bulk(
        [yf for _, yf in symbol_pairs],
        chunk_size=FETCH_CHUNK_SIZE,
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

    for i, (sym, yf_sym) in enumerate(symbol_pairs, 1):
        try:
            df = ohlc.get(yf_sym)
            if df is None:
                print(f"  [{i:3d}/{len(symbols)}] {sym:10s}  — no data")
                continue

            qualifies, k, d = is_stoch_basic_candidate(df)

            if qualifies:
                price = round(float(df["close"].iloc[-1]), 2)
                candidates.append(sym)
                print(f"  [{i:3d}/{len(symbols)}] {sym:10s}"
                      f"  STOCH  K:{k:.1f}  D:{d:.1f}  price:{price:.2f}")
            else:
                print(f"  [{i:3d}/{len(symbols)}] {sym:10s}  —")

        except Exception as exc:
            errors += 1
            print(f"  [{i:3d}/{len(symbols)}] {sym:10s}  ERROR: {exc}")

    print(f"\n{'=' * 55}")
    print(f"Stoch Basic Scanner candidates : {len(candidates)}")
    print(f"Errors                         : {errors}")

    if candidates:
        out_name = f"Stoch_BasicScan_{date.today()}.txt"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w") as f:
            f.write("\n".join(candidates))
        print(f"Saved → OUTPUT/BasicScan/{out_name}")
    else:
        print("No Stoch Basic Scanner candidates found.")


if __name__ == "__main__":
    main()
