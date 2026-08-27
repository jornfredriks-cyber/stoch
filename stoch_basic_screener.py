import configparser
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import yfinance as yf

from scan_utils import _Tee, adr, fetch_ohlc_bulk, fetch_universe

# ── Screener parameters — loaded from stoch_config.ini, with hardcoded fallbacks
_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_s = _cfg["stoch_basic_screener"] if "stoch_basic_screener" in _cfg else {}

MIN_PRICE        = float(_s.get("min_price",        10.0))
MIN_MARKET_CAP   = int(  _s.get("min_market_cap",   300_000_000))
MIN_AVG_VOL_30D  = int(  _s.get("min_avg_vol_30d",  1_000_000))
MIN_ADR_PCT      = float(_s.get("min_adr_pct",      2.0))
FETCH_CHUNK_SIZE = int(  _s.get("fetch_chunk_size", 50))

MIN_BARS = 30  # enough for a 30-day avg-volume window and a 14-period ADR


def _passes(df: pd.DataFrame) -> tuple[bool, str]:
    if len(df) < MIN_BARS:
        return False, "bars"
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    if float(c.iloc[-1]) < MIN_PRICE:
        return False, "price"
    if float(v.tail(30).mean()) < MIN_AVG_VOL_30D:
        return False, "volume"
    if adr(h, l, c) < MIN_ADR_PCT:
        return False, "adr"
    return True, "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_screener(output_folder: str | None = None) -> str:
    folder = output_folder or os.path.dirname(os.path.abspath(__file__))
    today  = date.today()

    print("=" * 55)
    print(f"Stoch Basic Screener  |  {today}")
    print(
        f"Price>{MIN_PRICE} | MktCap>{MIN_MARKET_CAP / 1e6:.0f}M | "
        f"AvgVol30D>{MIN_AVG_VOL_30D / 1e3:.0f}K | ADR>{MIN_ADR_PCT}%  "
        f"(no trend/ATH-proximity filter — by design, see stoch_config.ini)"
    )
    print("=" * 55)

    print("\n[1/4] Fetching ticker universe (NASDAQ Trader)…")
    universe = fetch_universe()
    print(f"  Total: {len(universe)} tickers\n")

    print(f"[2/4] Downloading 1y daily OHLC ({len(universe)} tickers)…")
    ohlc = fetch_ohlc_bulk(universe, chunk_size=FETCH_CHUNK_SIZE)
    print(f"  Downloaded: {len(ohlc)}/{len(universe)}\n")

    print("[3/4] Applying technical filters…")
    passed = []
    rejected: dict[str, int] = {}
    for sym, df in ohlc.items():
        ok, reason = _passes(df)
        if ok:
            passed.append(sym)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    print(f"  {len(passed)} pass all filters")
    for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f"  {count:5d} rejected by {reason}")
    print()

    print(f"[4/4] Fetching market cap for {len(passed)} candidates…")

    def _fetch_mcap(sym: str) -> tuple[str, float | None]:
        try:
            return sym, yf.Ticker(sym).fast_info.market_cap
        except Exception:
            return sym, None

    mcap_map: dict[str, float | None] = {}
    checked = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        for sym, mcap in executor.map(_fetch_mcap, passed):
            checked += 1
            mcap_map[sym] = mcap
            if checked % 25 == 0:
                qualifying = sum(1 for m in mcap_map.values() if m and m >= MIN_MARKET_CAP)
                print(f"  …{checked}/{len(passed)} checked, {qualifying} qualifying so far")

    final = [sym for sym in passed if mcap_map.get(sym) and mcap_map[sym] >= MIN_MARKET_CAP]
    print(f"  {len(final)} pass market cap >{MIN_MARKET_CAP / 1e6:.0f}M\n")

    out_name = f"Stoch Basic Screener_{today}.csv"
    out_path = os.path.join(folder, out_name)
    pd.DataFrame({"Symbol": final}).to_csv(out_path, index=False)
    print(f"Saved → OUTPUT/BasicScreener/{out_name}  ({len(final)} tickers)")
    return out_path


def main():
    folder     = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(folder, "OUTPUT", "BasicScreener")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"stoch_basic_screener_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    try:
        run_screener(output_dir)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Screener log → OUTPUT/BasicScreener/{os.path.basename(log_path)}")


if __name__ == "__main__":
    main()
