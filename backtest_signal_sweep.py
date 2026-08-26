import itertools
import os
import sys
from datetime import date

import pandas as pd

from backtest_signal import TRADE_VALUE_USD, load_darvas_raw_tickers, simulate_trades_signal
from scan_utils import _Tee, fetch_ohlc_bulk
from stoch_scan import resample_weekly

# Parameter grid: (stoch_length, k_smooth, d_smooth). Entry/exit thresholds
# (32/80) are held fixed per user request -- only the Stochastic settings
# are swept. Widen/narrow these lists to change the search space; each
# combo costs one pass over all cached tickers, no re-fetching.
#
# LENGTH_GRID: the first sweep (2026-07-21) used range(8,32,2) and every
# top-10 combo landed on length=8, the grid's own floor -- total $ P&L was
# still trending upward as length decreased, so 8 was a boundary artifact,
# not a real optimum. Extended down to 3 (dense 3-8, coarser 10-30 since
# that region already showed a clean declining trend) to find where it
# actually peaks or reverses.
LENGTH_GRID   = sorted(set(list(range(3, 9)) + list(range(8, 32, 2))))  # 3,4,5,6,7,8,10,...,30 (17 values)
K_SMOOTH_GRID = list(range(2, 8))       # 2..7          (6 values)
D_SMOOTH_GRID = list(range(2, 8))       # 2..7          (6 values)


def compute_kd_fast(weekly: pd.DataFrame, stoch_length: int, k_smooth: int, d_smooth: int) -> pd.DataFrame:
    """
    Same math as stoch_scan.compute_kd(), but takes an already-resampled
    weekly frame (see resample_weekly()) so the resample step isn't
    repeated for every parameter combo in the sweep.
    """
    lowest_low   = weekly["low"].rolling(stoch_length).min()
    highest_high = weekly["high"].rolling(stoch_length).max()
    raw_k = 100 * (weekly["close"] - lowest_low) / (highest_high - lowest_low)
    k_line = raw_k.rolling(k_smooth).mean()
    d_line = k_line.rolling(d_smooth).mean()
    return pd.DataFrame({"close": weekly["close"], "k": k_line, "d": d_line})


def run_sweep(folder: str):
    output_dir = os.path.join(folder, "OUTPUT")

    tickers = list(dict.fromkeys(load_darvas_raw_tickers()))
    yf_symbols = [t.replace(".", "-").replace("/", "-") for t in tickers]
    print(f"Tickers loaded : {len(tickers)}  (Darvas Raw screener)")
    print("Fetching full history once (auto_adjust=False), then sweeping in-memory...\n")

    ohlc = fetch_ohlc_bulk(yf_symbols, period="max", auto_adjust=False)

    weekly_by_ticker = {}
    for sym, yf_sym in zip(tickers, yf_symbols):
        df = ohlc.get(yf_sym)
        if df is not None:
            weekly_by_ticker[sym] = resample_weekly(df)

    combos = list(itertools.product(LENGTH_GRID, K_SMOOTH_GRID, D_SMOOTH_GRID))
    print(f"Tickers with data : {len(weekly_by_ticker)}/{len(tickers)}")
    print(f"Parameter combos  : {len(combos)}  (length x k_smooth x d_smooth)")
    print(f"Entry/exit levels : fixed at 32.0/80.0 (not swept)")
    print(f"Ranking by        : total $ P&L, closed trades only, ${TRADE_VALUE_USD:.0f}/trade\n")

    results = []
    for i, (length, k_smooth, d_smooth) in enumerate(combos, 1):
        all_trades = []
        for weekly in weekly_by_ticker.values():
            kd = compute_kd_fast(weekly, length, k_smooth, d_smooth)
            all_trades.extend(simulate_trades_signal(kd))

        closed = [t for t in all_trades if t["status"] == "closed"]
        open_  = [t for t in all_trades if t["status"] == "open"]
        wins   = [t for t in closed if t["return_pct"] > 0]
        losses = [t for t in closed if t["return_pct"] <= 0]
        total_pnl = sum(t["dollar_pnl"] for t in closed)
        win_rate  = (len(wins) / len(closed) * 100) if closed else float("nan")

        results.append({
            "stoch_length": length,
            "k_smooth": k_smooth,
            "d_smooth": d_smooth,
            "total_trades": len(all_trades),
            "closed": len(closed),
            "open": len(open_),
            "win_rate_pct": round(win_rate, 2) if closed else None,
            "avg_win_pct": round(sum(t["return_pct"] for t in wins) / len(wins), 2) if wins else None,
            "avg_loss_pct": round(sum(t["return_pct"] for t in losses) / len(losses), 2) if losses else None,
            "total_pnl": round(total_pnl, 2),
        })

        if i % 25 == 0 or i == len(combos):
            print(f"  [{i:4d}/{len(combos)}] length={length:2d} k={k_smooth} d={d_smooth}"
                  f"  -> ${total_pnl:>12,.2f}  ({len(closed)} closed, "
                  f"{'—' if not closed else f'{win_rate:.1f}%'} win)")

    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    out_name = f"StochSignal_Sweep_{date.today()}.csv"
    out_path = os.path.join(output_dir, out_name)
    pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"\n{'=' * 70}")
    print("Top 10 combos by total $ P&L:\n")
    for r in results[:10]:
        win_str = "—" if r["win_rate_pct"] is None else f"{r['win_rate_pct']}%"
        print(f"  length={r['stoch_length']:2d}  k_smooth={r['k_smooth']}  d_smooth={r['d_smooth']}"
              f"   ->  ${r['total_pnl']:>12,.2f}   {win_str} win   ({r['closed']} closed)")

    best = results[0]
    print(f"\nOptimal setting (plug into the Pine indicator's inputs to observe visually):")
    print(f"  periodK  (Length)       = {best['stoch_length']}")
    print(f"  smoothK  (%K Smoothing) = {best['k_smooth']}")
    print(f"  periodD  (%D Smoothing) = {best['d_smooth']}")
    print(f"  Total $ P&L: ${best['total_pnl']:,.2f}  |  Win rate: {best['win_rate_pct']}%  |  {best['closed']} closed trades")
    print(f"\nSaved -> OUTPUT/{out_name}")


def main():
    folder     = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(folder, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"stoch_signal_sweep_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        run_sweep(folder)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/{os.path.basename(log_path)}")


if __name__ == "__main__":
    main()
