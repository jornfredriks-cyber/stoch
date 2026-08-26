import configparser
import os
import sys
from datetime import date

import pandas as pd

from backtest_signal import TRADE_VALUE_USD, load_backtest_signal_tickers
from scan_utils import _Tee, fetch_ohlc_bulk
from stoch_scan import (
    STOCH_D_SMOOTH,
    STOCH_K_MAX,
    STOCH_K_MIN,
    STOCH_K_SMOOTH,
    STOCH_LENGTH,
    compute_kd,
    resample_weekly,
)

_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stoch_config.ini"))
_e = _cfg["backtest_signal_sltp_sweep"] if "backtest_signal_sltp_sweep" in _cfg else {}

EMA_LONG_FAST = int(_e.get("ema_long_fast", 50))
EMA_LONG_SLOW = int(_e.get("ema_long_slow", 200))

# Sweep grids -- module-level constants, same convention as
# backtest_signal_sweep.py's stoch-param grids (widen/narrow as needed).
SL_GRID     = [4, 5, 6, 7, 8, 9, 10]        # percent below entry
TP_GRID     = [1.5, 2.0, 2.5, 3.0, 3.5]     # R-multiple of the SL risk
FILTER_GRID = ["none", "long_trend"]        # "long_trend": require weekly EMA50>EMA200 at entry


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def build_weekly_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly close/k/d (from compute_kd) plus weekly ema_long_fast/ema_long_slow
    (EMA50/EMA200 on weekly closes -- a regime-level trend check, ~1yr vs
    ~4yr), computed ONCE per ticker and shared across every SL/TP/filter
    combo in the sweep -- none of those parameters change these values,
    only how simulate_trades_signal_sltp() interprets them.
    """
    weekly = resample_weekly(daily)
    kd = compute_kd(daily, STOCH_LENGTH, STOCH_K_SMOOTH, STOCH_D_SMOOTH)
    if weekly.empty or kd.empty:
        return kd.assign(ema_long_fast=pd.Series(dtype=float), ema_long_slow=pd.Series(dtype=float))
    return kd.assign(
        ema_long_fast=_ema(weekly["close"], EMA_LONG_FAST),
        ema_long_slow=_ema(weekly["close"], EMA_LONG_SLOW),
    )


def _find_sl_tp_touch(daily_slice: pd.DataFrame, stop_price: float, target_price: float):
    """
    Scans daily bars in chronological order for the first day whose low
    breaches stop_price or high breaches target_price. If a single day's
    range touches both (a wide-range day), SL wins -- a conservative
    worst-case assumption since raw OHLC can't tell us the intraday
    sequencing of the two touches.

    Returns (exit_date, exit_price, reason) for the first touch, or None.
    """
    for dt, day in daily_slice.iterrows():
        if day["low"] <= stop_price:
            return dt, stop_price, "SL"
        if day["high"] >= target_price:
            return dt, target_price, "TP"
    return None


def _trade_record(entry: dict, exit_date, exit_price: float, reason: str,
                   trade_value_usd: float, status: str = "closed") -> dict:
    return_pct = (exit_price / entry["entry_price"] - 1) * 100
    dollar_pnl = trade_value_usd * (exit_price / entry["entry_price"] - 1)
    return {
        "entry_date": entry["entry_date"],
        "entry_price": entry["entry_price"],
        "entry_k": entry["entry_k"],
        "exit_date": exit_date,
        "exit_price": round(exit_price, 4),
        "exit_reason": reason,
        "status": status,
        "return_pct": round(return_pct, 2),
        "dollar_pnl": round(dollar_pnl, 2),
        "holding_days": (pd.Timestamp(exit_date) - pd.Timestamp(entry["entry_date"])).days,
    }


def simulate_trades_signal_sltp(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    sl_pct: float,
    tp_r: float,
    filter_type: str = "none",
    entry_level: float = STOCH_K_MIN,
    exit_level: float = STOCH_K_MAX,
    trade_value_usd: float = TRADE_VALUE_USD,
) -> list[dict]:
    """
    Extends backtest_signal.simulate_trades_signal() with an SL/TP overlay
    and an optional entry filter, evaluated with "first hit wins" priority:

    Entry: same exact crossover as simulate_trades_signal() (prior week %K
    < entry_level, current week %K > entry_level, %K > %D), gated by a
    regime-level long-term trend check if filter_type="long_trend": weekly
    EMA(long_fast) > EMA(long_slow) (default 50/200, ~1yr vs ~4yr) on the
    same bar. filter_type="none" (default) applies no gate. Fill at that
    week's close. stop_price = entry_price*(1-sl_pct/100);
    target_price = entry_price + tp_r * (entry_price - stop_price).

    Exit, checked in this order once in a position:
      1. Daily intraweek touch of stop_price or target_price (checked over
         the daily bars strictly after the previous processed weekly bar,
         through and including the current weekly bar) -- exits the day
         it's touched, at the stop/target price itself.
      2. If no daily touch, the original signal exit (prior week %K >
         exit_level, current week %K < exit_level, %K < %D) -- exits at
         that week's close.
    Whichever fires first (in date order) wins; a week with no daily touch
    and no signal fire simply continues the position into the next week.

    Still-open positions when data ends are recorded as "open", marked to
    market at the last available daily close, excluded from win/loss
    stats by callers (same convention as simulate_trades_signal()).
    """
    weekly = weekly.dropna(subset=["k", "d"])

    trades: list[dict] = []
    in_position = False
    entry: dict | None = None
    prev_k = None
    prev_week_end = None

    for dt, row in weekly.iterrows():
        k, d, close = row["k"], row["d"], row["close"]

        if not in_position:
            if prev_k is not None and prev_k < entry_level and k > entry_level and k > d:
                filter_ok = True
                if filter_type == "long_trend":
                    ema_f, ema_s = row["ema_long_fast"], row["ema_long_slow"]
                    filter_ok = pd.notna(ema_f) and pd.notna(ema_s) and ema_f > ema_s
                if filter_ok:
                    stop_price = close * (1 - sl_pct / 100)
                    risk = close - stop_price
                    entry = {
                        "entry_date": dt,
                        "entry_price": close,
                        "entry_k": round(k, 2),
                        "stop_price": stop_price,
                        "target_price": close + tp_r * risk,
                    }
                    in_position = True
        else:
            window = (
                daily.loc[(daily.index > prev_week_end) & (daily.index <= dt)]
                if prev_week_end is not None
                else daily.iloc[0:0]
            )
            touch = _find_sl_tp_touch(window, entry["stop_price"], entry["target_price"])
            if touch is not None:
                exit_date, exit_price, reason = touch
                trades.append(_trade_record(entry, exit_date, exit_price, reason, trade_value_usd))
                in_position = False
                entry = None
            elif prev_k is not None and prev_k > exit_level and k < exit_level and k < d:
                trades.append(_trade_record(entry, dt, close, "signal", trade_value_usd))
                in_position = False
                entry = None

        prev_k = k
        prev_week_end = dt

    if in_position:
        last_date = daily.index[-1]
        last_close = float(daily["close"].iloc[-1])
        trades.append(_trade_record(entry, last_date, last_close, "open", trade_value_usd, status="open"))

    return trades


def _run(folder: str, output_dir: str):
    tickers, source = load_backtest_signal_tickers(folder)
    tickers = list(dict.fromkeys(tickers))
    print(f"Tickers loaded : {len(tickers)}  ({source})")
    print(f"Weekly Stoch   : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  Entry: K crosses >{STOCH_K_MIN} & K>D  |  Signal exit: K crosses <{STOCH_K_MAX} & K<D")
    print(f"Filter types   : none, long_trend (weekly EMA{EMA_LONG_FAST}>EMA{EMA_LONG_SLOW})")
    combos = [(sl, tp, filt) for sl in SL_GRID for tp in TP_GRID for filt in FILTER_GRID]
    print(f"Grid           : SL{SL_GRID}% x TP{TP_GRID}R x filter{FILTER_GRID} = {len(combos)} combinations\n")

    yf_symbols = [t.replace(".", "-").replace("/", "-") for t in tickers]
    ohlc = fetch_ohlc_bulk(yf_symbols, period="max", auto_adjust=False)

    print(f"Downloaded     : {len(ohlc)}/{len(tickers)} tickers, building weekly frames…\n")
    tickers_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for sym, yf_sym in zip(tickers, yf_symbols):
        df = ohlc.get(yf_sym)
        if df is None:
            continue
        tickers_data[sym] = (df, build_weekly_frame(df))

    sweep_results = []
    per_ticker_by_combo: dict[tuple, dict[str, dict]] = {}
    trades_by_combo: dict[tuple, list[dict]] = {}

    for combo_idx, (sl, tp, filter_type) in enumerate(combos, 1):
        all_trades = []
        per_ticker: dict[str, dict] = {}

        for sym, (daily_df, weekly_df) in tickers_data.items():
            trades = simulate_trades_signal_sltp(daily_df, weekly_df, sl, tp, filter_type)
            for t in trades:
                t["ticker"] = sym
            all_trades.extend(trades)

            closed = [t for t in trades if t["status"] == "closed"]
            if closed:
                wins = sum(1 for t in closed if t["return_pct"] > 0)
                losses = len(closed) - wins
                per_ticker[sym] = {"wins": wins, "losses": losses, "win_pct": round(wins / len(closed) * 100, 1)}

        closed_trades = [t for t in all_trades if t["status"] == "closed"]
        open_trades = [t for t in all_trades if t["status"] == "open"]
        wins_all = [t for t in closed_trades if t["return_pct"] > 0]
        losses_all = [t for t in closed_trades if t["return_pct"] <= 0]
        sl_exits = sum(1 for t in all_trades if t["exit_reason"] == "SL")
        tp_exits = sum(1 for t in all_trades if t["exit_reason"] == "TP")
        signal_exits = sum(1 for t in all_trades if t["exit_reason"] == "signal")

        total_pnl = sum(t["dollar_pnl"] for t in closed_trades)
        win_rate = (len(wins_all) / len(closed_trades) * 100) if closed_trades else 0.0
        avg_win = (sum(t["return_pct"] for t in wins_all) / len(wins_all)) if wins_all else 0.0
        avg_loss = (sum(t["return_pct"] for t in losses_all) / len(losses_all)) if losses_all else 0.0

        sweep_results.append({
            "sl_pct": sl, "tp_r": tp, "filter_type": filter_type,
            "trades": len(all_trades), "closed": len(closed_trades), "open": len(open_trades),
            "sl_exits": sl_exits, "tp_exits": tp_exits, "signal_exits": signal_exits,
            "win_rate_pct": round(win_rate, 1), "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2), "total_pnl": round(total_pnl, 2),
        })
        per_ticker_by_combo[(sl, tp, filter_type)] = per_ticker
        trades_by_combo[(sl, tp, filter_type)] = all_trades

        print(f"  [{combo_idx:3d}/{len(combos)}] SL={sl:2d}%  TP={tp:.2f}R  filter={filter_type:10s}"
              f"   trades={len(all_trades):4d}  SL={sl_exits} TP={tp_exits} sig={signal_exits}"
              f"  win%={win_rate:5.1f}  pnl=${total_pnl:,.2f}")

    sweep_results.sort(key=lambda r: r["total_pnl"], reverse=True)

    total_sl_exits = sum(r["sl_exits"] for r in sweep_results)
    print(f"\nSL exits across all {len(combos)} combos: {total_sl_exits} total"
          f" ({'none hit' if total_sl_exits == 0 else 'SL is triggering'})")

    out_name = f"StochSignal_SLTP_Sweep_{date.today()}.md"
    out_path = os.path.join(output_dir, out_name)
    _write_markdown(out_path, sweep_results, per_ticker_by_combo, source, len(tickers_data))

    for rank, r in enumerate(sweep_results[:3], 1):
        key = (r["sl_pct"], r["tp_r"], r["filter_type"])
        trades_out_name = f"StochSignal_SLTP_Trades_Rank{rank}_{date.today()}.csv"
        trades_out_path = os.path.join(output_dir, trades_out_name)
        columns = ["ticker", "entry_date", "entry_price", "entry_k", "exit_date",
                   "exit_price", "exit_reason", "status", "return_pct", "dollar_pnl", "holding_days"]
        pd.DataFrame(trades_by_combo[key], columns=columns).to_csv(trades_out_path, index=False)
        print(f"Saved          : OUTPUT/{trades_out_name}  ({len(trades_by_combo[key])} trades)")

    print(f"\n{'=' * 55}")
    best = sweep_results[0]
    print(f"Best combo     : SL={best['sl_pct']}%  TP={best['tp_r']}R  filter={best['filter_type']}"
          f"  ->  ${best['total_pnl']:,.2f} total P&L, {best['win_rate_pct']}% win rate")
    print(f"Saved          : OUTPUT/{out_name}")


def _write_markdown(path: str, sweep_results: list[dict], per_ticker_by_combo: dict,
                     source: str, ticker_count: int):
    lines = [
        "# Stochastic Weekly Signal — SL/TP + Trend Filter Sweep",
        f"\n*Date: {date.today()}*  ",
        f"*Universe: {ticker_count} tickers ({source})*  ",
        f"*Weekly Stoch: {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}, entry/exit levels "
        f"{STOCH_K_MIN}/{STOCH_K_MAX}*\n",
        f"SL grid: {SL_GRID}% | TP grid: {TP_GRID}R (R-multiple of SL risk) | "
        f"Filter types: none, long_trend (weekly EMA{EMA_LONG_FAST}>EMA{EMA_LONG_SLOW}) | "
        f"{len(sweep_results)} combinations tested\n",
        "SL/TP checked against daily intraweek high/low (not just weekly close); if both are touched "
        "on the same day, SL is assumed to hit first (conservative). Exit priority each week: SL/TP "
        "touch (if any) wins over the strategy's own sell signal.\n",
        "Full trade-level detail (entry/exit dates, prices, exit reason) for the top 3 combos is saved "
        "alongside this file as `StochSignal_SLTP_Trades_Rank1/2/3_<date>.csv`.\n",
        "## Sweep Results (sorted by total $ P&L)\n",
        "| Rank | SL% | TP (R) | Filter | Trades | Closed | Open | SL Exits | TP Exits | Signal Exits "
        "| Win % | Avg Win % | Avg Loss % | Total P&L |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(sweep_results, 1):
        lines.append(
            f"| {rank} | {r['sl_pct']} | {r['tp_r']} | {r['filter_type']} | "
            f"{r['trades']} | {r['closed']} | {r['open']} | {r['sl_exits']} | {r['tp_exits']} | "
            f"{r['signal_exits']} | {r['win_rate_pct']}% | "
            f"{r['avg_win_pct']}% | {r['avg_loss_pct']}% | ${r['total_pnl']:,.2f} |"
        )

    lines.append("\n## Per-Ticker Results — Top 3 Combos\n")
    for rank, r in enumerate(sweep_results[:3], 1):
        key = (r["sl_pct"], r["tp_r"], r["filter_type"])
        per_ticker = per_ticker_by_combo[key]
        lines.append(
            f"### Rank #{rank} — SL={r['sl_pct']}%, TP={r['tp_r']}R, filter={r['filter_type']}\n"
        )
        lines.append("| Ticker | Wins | Losses | Win % |")
        lines.append("|---|---|---|---|")
        for sym, stats in sorted(per_ticker.items(), key=lambda kv: -kv[1]["win_pct"]):
            lines.append(f"| {sym} | {stats['wins']} | {stats['losses']} | {stats['win_pct']}% |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(folder, "OUTPUT")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"backtest_signal_sltp_sweep_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        _run(folder, output_dir)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/{os.path.basename(log_path)}")


if __name__ == "__main__":
    main()
