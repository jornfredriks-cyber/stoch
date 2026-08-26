import configparser
import os
import sys
import time
from datetime import date

import pandas as pd

from backtest_signal import load_darvas_raw_tickers
from scan_utils import _Tee, fetch_ohlc_bulk, fetch_ohlc_single
from stoch_scan import (
    STOCH_D_SMOOTH,
    STOCH_K_MIN,
    STOCH_K_SMOOTH,
    STOCH_LENGTH,
    compute_kd,
)

_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_s = _cfg["stoch_signal_scan"] if "stoch_signal_scan" in _cfg else {}

EMA_LENGTH       = int(  _s.get("ema_length",       50))
EMA_BAND_MIN_PCT = float(_s.get("ema_band_min_pct",  0.0))
EMA_BAND_MAX_PCT = float(_s.get("ema_band_max_pct",  5.0))


def is_fresh_stoch_signal_buy(
    df: pd.DataFrame,
    stoch_length: int = STOCH_LENGTH,
    k_smooth: int = STOCH_K_SMOOTH,
    d_smooth: int = STOCH_D_SMOOTH,
    entry_level: float = STOCH_K_MIN,
) -> tuple:
    """
    Checks whether a stock's Weekly Stochastic %K crossed STRICTLY above
    entry_level on the most recently completed weekly bar (prior bar's %K
    < entry_level, current bar's %K > entry_level), with %K > %D on that
    same bar -- the exact "Stochastic Weekly Signal" Pine indicator's
    buySignal condition (see backtest_signal.simulate_trades_signal),
    evaluated only against the latest bar rather than the full history.

    Returns:
        (True,  k_value, d_value, entry_date, entry_price) — fresh buy signal
        (False, None,    None,    None,       None)        — no fresh signal,
                                                               or insufficient
                                                               weekly history
    """
    kd = compute_kd(df, stoch_length, k_smooth, d_smooth)
    valid = kd.dropna(subset=["k", "d"])
    if len(valid) < 2:
        return False, None, None, None, None

    k_prev = valid["k"].iloc[-2]
    k_now  = valid["k"].iloc[-1]
    d_now  = valid["d"].iloc[-1]
    close_now = valid["close"].iloc[-1]
    entry_date = valid.index[-1]

    fresh = (k_prev < entry_level) and (k_now > entry_level) and (k_now > d_now)
    if fresh:
        return True, round(k_now, 2), round(d_now, 2), entry_date, close_now

    return False, None, None, None, None


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def is_not_stretched(
    df: pd.DataFrame,
    ema_length: int = EMA_LENGTH,
    band_min_pct: float = EMA_BAND_MIN_PCT,
    band_max_pct: float = EMA_BAND_MAX_PCT,
) -> tuple:
    """
    Checks whether a stock's current price sits within [band_min_pct,
    band_max_pct]% above its daily EMA(ema_length) -- i.e. not "stretched"
    (chasing an already-extended move away from its EMA). This is the one
    genuinely new idea from the retired ema50_screener.py supplement
    (2026-07-25/26): everything else that scanner checked (price/volume/ADR
    floors) is already covered by the Darvas Raw screener's own criteria
    that produced this ticker's universe in the first place.

    Drops any trailing NaN close row first -- yfinance (auto_adjust=True
    elsewhere in this project) can leave the most-recent session's OHLC as
    NaN until its adjustment pipeline catches up; a naive .iloc[-1] read
    would silently corrupt this check (see feedback memory from 2026-07-25).

    Returns (True, dist_pct) if in-band, (False, dist_pct) if not,
    (False, None) if there's not enough daily history for the EMA.
    """
    close = df["close"].dropna()
    if len(close) < ema_length + 10:
        return False, None

    ema = float(_ema(close, ema_length).iloc[-1])
    dist_pct = (float(close.iloc[-1]) - ema) / ema * 100
    return (band_min_pct <= dist_pct <= band_max_pct), round(dist_pct, 2)


def main():
    folder     = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(folder, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"stoch_signal_scan_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        _run(output_dir)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/{os.path.basename(log_path)}")


def _run(output_dir: str):
    tickers = load_darvas_raw_tickers()
    tickers = list(dict.fromkeys(tickers))
    print(f"Tickers loaded : {len(tickers)}  (Darvas Raw screener)")
    print(f"Weekly Stoch   : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  Fresh buy: %K crosses strictly above {STOCH_K_MIN} on the latest"
          f" weekly bar, with %K>%D")
    print(f"Not stretched  : close within {EMA_BAND_MIN_PCT}-{EMA_BAND_MAX_PCT}%"
          f" above daily EMA{EMA_LENGTH}\n")

    yf_symbols = [t.replace(".", "-").replace("/", "-") for t in tickers]
    # auto_adjust=False: matches TradingView's default unadjusted chart, same
    # rationale as backtest_signal.py -- the live Pine signal is computed off
    # unadjusted prices, not a dividend-adjusted reconstruction.
    ohlc = fetch_ohlc_bulk(yf_symbols, auto_adjust=False)

    failed = [yf for yf in yf_symbols if yf not in ohlc]
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
    for i, (sym, yf_sym) in enumerate(zip(tickers, yf_symbols), 1):
        try:
            df = ohlc.get(yf_sym)
            if df is None:
                print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  — no data")
                continue

            fresh, k, d, entry_date, price = is_fresh_stoch_signal_buy(df)
            if not fresh:
                print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  —")
                continue

            not_stretched, dist_pct = is_not_stretched(df)
            if not not_stretched:
                reason = "stretched" if dist_pct is not None else "insufficient history"
                print(f"  [{i:3d}/{len(tickers)}] {sym:10s}"
                      f"  BUY signal but {reason} (EMA{EMA_LENGTH} dist: {dist_pct}%) — skipped")
                continue

            candidates.append({
                "ticker": sym,
                "entry_date": entry_date.date(),
                "entry_k": k,
                "entry_d": d,
                "price": round(price, 2),
                "ema50_dist_pct": dist_pct,
            })
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}"
                  f"  BUY  K:{k:.1f}  D:{d:.1f}  price:{price:.2f}  EMA{EMA_LENGTH} dist:{dist_pct}%")

        except Exception as exc:
            errors += 1
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  ERROR: {exc}")

    print(f"\n{'=' * 55}")
    print(f"Fresh buy candidates : {len(candidates)}")
    print(f"Errors               : {errors}")

    out_name = f"StochSignal_Candidates_{date.today()}.csv"
    out_path = os.path.join(output_dir, out_name)
    columns = ["ticker", "entry_date", "entry_k", "entry_d", "price", "ema50_dist_pct"]
    pd.DataFrame(candidates, columns=columns).to_csv(out_path, index=False)
    print(f"Saved → OUTPUT/{out_name}")


if __name__ == "__main__":
    main()
