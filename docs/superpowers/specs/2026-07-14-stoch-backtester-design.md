---
date: 2026-07-14
topic: stoch-backtester
status: approved
---

# Weekly Stochastic Sweet Spot — Backtester Design

## Purpose

Validate the "Weekly Stochastic Sweet Spot" strategy (`Weekly Stochastic Sweet Spot Strategy.md`) against real historical data: simulate the documented long entry/exit rules across a ticker universe and report whether the strategy's claimed ~70% win rate holds up. This is phase one of a two-phase effort — a later phase will sweep Stochastic parameter combinations (length/%K-smooth/%D-smooth) on top of this same engine to find the best-performing settings. That sweep is explicitly **out of scope** here; this spec covers only the single-parameter-set backtest engine, built at the current default (19/4/4).

A separate, independent follow-up (also out of scope here) will add a "tightened" live scanner variant that surfaces candidates *approaching* 32 from below with %K > %D and rising — the existing 32–80 `is_stoch_sweet_spot` filter in `stoch_scan.py` is left untouched.

## Trading Rules (long-only)

Confirmed against the strategy doc:

- **Entry**: first week where %K crosses above 32 (prior week %K < 32, current week %K ≥ 32). A pure %K level-cross — %D is not checked at entry. Fill at that week's close.
- **Exit**: first week *after entry* where **%K < %D AND %K < 80** are simultaneously true. This correctly handles both real-world orderings: if %K/%D cross while %K is still ≥ 80, the position is held ("ride it") until %K also drops back under 80; if %K drops under 80 while still above %D, the position is held until %K actually crosses below %D. Fill at that week's close.
- **No stop loss.** Price action between entry and exit is irrelevant — only the weekly %K/%D relationship matters.
- **Open-ended trades**: if a position is still open when a ticker's data ends, it is recorded as **open**, marked-to-market at the last available close, and excluded from win/loss classification (it isn't a completed round-trip).
- **Long-only.** The strategy doc's mirror short-side rule (short near 80, cover on %K crossing back above %D) is not implemented in this phase.
- Each ticker is simulated independently as a repeating state machine (Flat → Long → Flat → …) — multiple sequential trades per ticker over its history are expected and counted separately. No portfolio/capital simulation; each trade is an isolated % return.

## Folder Structure

Two new folders, scoped to the backtester only (not adopted by `screener.py`/`stoch_scan.py` in this phase, to avoid touching already-shipped, tested code):

- **`INPUT/`** — where you drop a TradingView Watchlist export for the backtester to pick up.
- **`OUTPUT/`** — where the backtester writes its trades CSV and log.

Both are created automatically (with `.gitkeep`) if missing when `backtest.py` runs.

## Ticker Source

New `scan_utils.find_ticker_list(input_folder, fallback_folder)`:

1. Look for a file matching `*Watchlist*` OR TradingView's actual default export naming, `[Tt]radingview*` (confirmed in practice: `Tradingview <ListName>_<date>_<id>.txt` — it does not contain the word "Watchlist" at all), in `input_folder` — plain text, comma-separated `EXCHANGE:TICKER`, e.g. `NASDAQ:AAPL,NYSE:V` — same parsing convention as `Dashboard/data_loader.py`'s `parse_ticker_line`, including the `OSLO:` → `.OL` yfinance-symbol conversion for consistency, even though the current universe is NASDAQ/NYSE-only.
2. If no watchlist file is found in `INPUT/`, fall back to the latest `*Screener*.csv` in `fallback_folder` via the existing `find_latest_csv` — this stays pointed at `Stoch/` root, exactly where `screener.py` already writes it today. No need to move or copy that file anywhere for the fallback to keep working.

`backtest.py` calls this as `find_ticker_list(input_folder=INPUT_DIR, fallback_folder=folder)`. `stoch_scan.py` (the live scanner) is **not** changed to use this in this phase — it keeps reading the screener CSV from `Stoch/` root as it does today. The shared function is written once so wiring the live scanner to it later is a small follow-up, not a redesign.

## Data Range

Full available yfinance daily history (`period="max"`) per ticker, resampled to Friday-anchored weekly bars — same `W-FRI` resample convention already used in `is_stoch_sweet_spot` (final in-progress week included as-is).

`scan_utils.fetch_ohlc_bulk` gains an optional `period` parameter (defaults to the existing `HISTORY_PERIOD = "1y"` constant so `screener.py` and `stoch_scan.py` are unaffected); `backtest.py` calls it with `period="max"`.

## Architecture

```
scan_utils.py
  + find_ticker_list(input_folder, fallback_folder) -> list[str]
      # watchlist (INPUT/) with screener-CSV fallback (project root)
  ~ fetch_ohlc_bulk(..., period=HISTORY_PERIOD) # period now overridable

stoch_scan.py
  ~ is_stoch_sweet_spot() refactored to call the new compute_kd() below
    (behavior unchanged — existing 7 tests must still pass unmodified)
  + compute_kd(df, stoch_length, k_smooth, d_smooth) -> pd.DataFrame
    columns: close, k, d — indexed by weekly bar
    (extracted so backtest.py reuses the exact same math as the live scanner)

backtest.py                                     # new
  simulate_trades(weekly: pd.DataFrame, entry_level=32.0, exit_level=80.0) -> list[dict]
    # pure state-machine over a precomputed weekly close/k/d frame — no I/O,
    # independently unit-testable with synthetic K/D series
  run_backtest(folder) -> writes CSV + log to OUTPUT/, prints summary
    # orchestration: find_ticker_list(INPUT/, fallback=folder) -> fetch_ohlc_bulk(period="max")
    #   -> compute_kd per ticker -> simulate_trades per ticker -> aggregate
    # ensures INPUT/ and OUTPUT/ exist (creates with .gitkeep if missing)
  main()                                        # _Tee logging wrapper, same pattern as stoch_scan.py

INPUT/.gitkeep                                   # new — drop a *Watchlist* export file here
OUTPUT/.gitkeep                                  # new — Stoch_Backtest_Trades_*.csv + logs land here

RUN Stoch Backtest.command                       # new launcher, same style as existing .command files
```

Stochastic settings (length=19, k_smooth=4, d_smooth=4) and the sweet-spot band (32/80, reused directly as entry/exit levels) are read from the existing `[stoch_scanner]` section of `stoch_config.ini` — no new config section. When the future parameter-sweep phase arrives, it will call `compute_kd`/`simulate_trades` directly with overridden arguments rather than adding config plumbing for a grid.

## Trade record fields

Each trade (in the output CSV and in `simulate_trades`'s return value):

| Field | Description |
|---|---|
| `ticker` | Symbol |
| `entry_date`, `entry_price`, `entry_k` | Fill week |
| `exit_date`, `exit_price`, `exit_k`, `exit_d` | Fill week (empty if open) |
| `status` | `closed` or `open` |
| `return_pct` | `(exit_price / entry_price - 1) * 100`, using the last available weekly close if open |
| `holding_weeks` | Weeks between entry and exit (or entry and last available week, if open) |

## Output

Written to `OUTPUT/`:

- `OUTPUT/Stoch_Backtest_Trades_YYYY-MM-DD.csv` — one row per trade across all tickers
- `OUTPUT/stoch_backtest_log_YYYY-MM-DD.txt` — full run log (same `_Tee` pattern as `screener.py`/`stoch_scan.py`)
- Console/log summary block: total trades, closed vs. open count, win rate (closed trades only), avg winning return %, avg losing return %, avg holding period (weeks) — framed to compare directly against the strategy doc's claimed ~70% win rate / variable win size.

## Testing

`tests/test_backtest.py`, unit-testing `simulate_trades` directly against synthetic `pd.DataFrame(columns=["close","k","d"])` fixtures (bypassing OHLC/resampling, which is already covered by `test_stoch_scan.py`):

1. Single clean trade: K crosses 32, later K<D while K<80 → one closed trade with correct entry/exit fields.
2. Sequential trades: two independent entry/exit cycles on one series → two trades recorded.
3. Holds through K≥80 with K<D already true → exit only fires once K also drops <80 (not on the K<D week).
4. Holds while K<80 but K still >D → exit only fires once K<D becomes true (not on the K<80 week).
5. Never exits before data ends → one `open` trade, marked at last close, correct `holding_weeks`.
6. Never enters (K never reaches 32) → empty trade list.

`tests/test_stoch_scan.py`'s existing 7 tests must continue passing unmodified after `compute_kd` is extracted — confirms the refactor didn't change live-scan behavior.

## Explicitly Out of Scope (this phase)

- Parameter sweep across Stochastic length/%K-smooth/%D-smooth combinations
- The "tightened" approaching-32 live scanner variant
- Short-side simulation
- Portfolio-level simulation (position sizing, capital, overlapping-trade limits)
- Wiring `stoch_scan.py`'s live scan to `find_ticker_list`'s watchlist support
