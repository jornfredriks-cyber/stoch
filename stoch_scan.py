import configparser
import os
import sys
import time
from datetime import date

import pandas as pd

from scan_utils import _Tee, fetch_ohlc_bulk, fetch_ohlc_single, find_latest_csv

# ── Config ────────────────────────────────────────────────────────────────────
_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_st = _cfg["stoch_scanner"] if "stoch_scanner" in _cfg else {}

STOCH_LENGTH      = int(  _st.get("stoch_length",      19))
STOCH_K_SMOOTH    = int(  _st.get("stoch_k_smooth",     4))
STOCH_D_SMOOTH    = int(  _st.get("stoch_d_smooth",     4))
STOCH_K_MIN       = float(_st.get("stoch_k_min",       32.0))
STOCH_K_MAX       = float(_st.get("stoch_k_max",       80.0))
K_RISING_LOOKBACK = int(  _st.get("k_rising_lookback",  1))
FETCH_CHUNK_SIZE  = int(  _st.get("fetch_chunk_size",  50))


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples the daily OHLC frame into Friday-anchored weekly bins
    (high=max, low=min, close=last). The final bin is NOT dropped/deferred
    when the current week is still in progress — it is used as-is,
    matching how TradingView plots a live weekly candle off the
    still-forming week. Extracted from compute_kd() so a parameter sweep
    (backtest_signal_sweep.py) can resample once per ticker and reuse it
    across many stoch_length/k_smooth/d_smooth combos, instead of
    re-resampling per combo.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["high", "low", "close"])
    return df.resample("W-FRI").agg({"high": "max", "low": "min", "close": "last"}).dropna()


def compute_kd(
    df: pd.DataFrame,
    stoch_length: int = STOCH_LENGTH,
    k_smooth: int = STOCH_K_SMOOTH,
    d_smooth: int = STOCH_D_SMOOTH,
) -> pd.DataFrame:
    """
    Resamples the daily OHLC frame into Friday-anchored weekly bins (see
    resample_weekly()) and computes the weekly Stochastic %K/%D lines.

    Returns a DataFrame indexed by weekly bar with columns "close", "k",
    "d". %k/%d are NaN for the initial weeks where the rolling windows
    aren't full yet. Empty/None input returns an empty DataFrame with
    those columns.
    """
    weekly = resample_weekly(df)
    if weekly.empty:
        return pd.DataFrame(columns=["close", "k", "d"])

    lowest_low   = weekly["low"].rolling(stoch_length).min()
    highest_high = weekly["high"].rolling(stoch_length).max()
    raw_k = 100 * (weekly["close"] - lowest_low) / (highest_high - lowest_low)
    k_line = raw_k.rolling(k_smooth).mean()
    d_line = k_line.rolling(d_smooth).mean()

    return pd.DataFrame({"close": weekly["close"], "k": k_line, "d": d_line})


def is_stoch_sweet_spot(
    df: pd.DataFrame,
    stoch_length: int = STOCH_LENGTH,
    k_smooth: int = STOCH_K_SMOOTH,
    d_smooth: int = STOCH_D_SMOOTH,
    k_min: float = STOCH_K_MIN,
    k_max: float = STOCH_K_MAX,
    k_rising_lookback: int = K_RISING_LOOKBACK,
) -> tuple:
    """
    Checks whether a stock's Weekly Stochastic (length/%K-smooth/%D-smooth,
    default 19/4/4) has its latest %K value inside [k_min, k_max] — the
    strategy's "sweet spot" band (32-80 by default) — with %K above %D and
    %K rising versus `k_rising_lookback` weeks back.

    Returns:
        (True,  k_value, d_value)  — latest %K inside band, above %D, rising
        (False, None,    None)     — condition not met, or insufficient
                                       weekly history to compute both lines
    """
    kd = compute_kd(df, stoch_length, k_smooth, d_smooth)

    valid = kd["d"].dropna()
    if len(valid) < k_rising_lookback + 1:
        return False, None, None

    k_now  = kd["k"].loc[valid.index[-1]]
    k_prev = kd["k"].loc[valid.index[-1 - k_rising_lookback]]
    d_now  = valid.iloc[-1]

    qualifies = (k_min <= k_now <= k_max) and (k_now > d_now) and (k_now > k_prev)

    if qualifies:
        return True, round(k_now, 2), round(d_now, 2)

    return False, None, None


def main():
    folder       = os.path.dirname(os.path.abspath(__file__))
    screener_dir = os.path.join(folder, "OUTPUT", "Screener")
    output_dir   = os.path.join(folder, "OUTPUT", "SweetSpot")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"stoch_scan_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        _run(screener_dir, output_dir)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/SweetSpot/{os.path.basename(log_path)}")


def _run(screener_dir: str, output_dir: str):
    csv_path = find_latest_csv(screener_dir)
    print(f"Screener file : {os.path.basename(csv_path)}")

    tv      = pd.read_csv(csv_path)
    symbols = tv["Symbol"].dropna().tolist()
    print(f"Tickers loaded: {len(symbols)}")
    print(f"Weekly Stoch  : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  %K band: {STOCH_K_MIN}-{STOCH_K_MAX}  |  require %K>%D and rising")
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

            qualifies, k, d = is_stoch_sweet_spot(df)

            if qualifies:
                candidates.append(sym)
                print(f"  [{i:3d}/{len(symbols)}] {sym:10s}"
                      f"  STOCH  K:{k:.1f}  D:{d:.1f}")
            else:
                print(f"  [{i:3d}/{len(symbols)}] {sym:10s}  —")

        except Exception as exc:
            errors += 1
            print(f"  [{i:3d}/{len(symbols)}] {sym:10s}  ERROR: {exc}")

    print(f"\n{'='*55}")
    print(f"Stoch Sweet Spot candidates : {len(candidates)}")
    print(f"Errors                      : {errors}")

    if candidates:
        out_name = f"Stoch_SweetSpot_{date.today()}.txt"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w") as f:
            f.write("\n".join(candidates))
        print(f"Saved → OUTPUT/SweetSpot/{out_name}")
    else:
        print("No Stoch Sweet Spot candidates found.")


if __name__ == "__main__":
    main()
