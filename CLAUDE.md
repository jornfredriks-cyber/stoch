# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project Overview

Standalone toolkit implementing the "Weekly Stochastic Sweet Spot" trend-following strategy (`Weekly Stochastic Sweet Spot Strategy.md`, sourced from a YouTube video — treat it as reference material, don't edit it with derived analysis). Originally built inside `Darvas Phase3 Scan/`, relocated to its own project with its own venv/config/tests on 2026-07-14 so it has no dependency on that project.

Three-stage pipeline, same shape as `Darvas Phase3 Scan/`:

| Stage | Script | Purpose | Output |
|---|---|---|---|
| 1 | `screener.py` | Pre-filter NASDAQ+NYSE universe (price/volume/EMA/ADR/RSI/ATH) | `Stoch Raw Screener_YYYY-MM-DD.csv` |
| 2 | `stoch_scan.py` | Live scan: flag tickers currently inside the 32–80 sweet-spot band, %K>%D, rising | `Stoch_SweetSpot_YYYY-MM-DD.txt` |
| 3 | `backtest.py` | Backtest the strategy's long entry/exit rules against full historical data | `OUTPUT/Stoch_Backtest_Trades_YYYY-MM-DD.csv` |

## Running

```bash
cd "Stoch"

# Live scan pipeline (screener → sweet-spot scan)
venv/bin/python3 screener.py && venv/bin/python3 stoch_scan.py
# or double-click "RUN Stoch Sweet Scan.command"

# Backtest (reads ticker list from INPUT/, or falls back to the screener CSV)
venv/bin/python3 backtest.py
# or double-click "RUN Stoch Backtest.command"

# Tests
venv/bin/python3 -m pytest tests/ -v
```

## Configuration

All tunable parameters live in `stoch_config.ini` — don't hardcode values in the Python files.

- `[screener]` — pre-filter thresholds (price, market cap, volume, ADR, EMA trend, RSI, ATH proximity). Mirrors Darvas Phase3 Scan's screener defaults.
- `[stoch_scanner]` — Stochastic settings (`stoch_length`/`stoch_k_smooth`/`stoch_d_smooth`, default 19/4/4 per the strategy doc) and the sweet-spot band (`stoch_k_min`=32.0, `stoch_k_max`=80.0). **`backtest.py` reads these same constants from `stoch_scan.py` — there is no separate `[backtest]` section.** The 32/80 band doubles as the backtest's entry/exit levels.

## Architecture

**`scan_utils.py`** — shared utility layer, no strategy logic:
- `fetch_ohlc_bulk(yf_symbols, period=HISTORY_PERIOD)` — chunked yfinance download with retry. `period` defaults to `"1y"` (what `screener.py`/`stoch_scan.py` need) but `backtest.py` overrides it to `"max"` for full history.
- `fetch_ohlc_single`, `find_latest_csv`, `_Tee` (stdout+logfile tee)
- `find_ticker_list(input_folder, fallback_folder)` — resolves the ticker universe. Looks in `input_folder` (i.e. `INPUT/`) for a file matching `*Watchlist*` **or** TradingView's actual default export naming `[Tt]radingview*` (confirmed in practice: `Tradingview <ListName>_<date>_<id>.txt` — it never contains the word "Watchlist"). Falls back to the latest `*Screener*.csv` in `fallback_folder` if neither is found. Only `backtest.py` uses this today — `screener.py`/`stoch_scan.py` still read the screener CSV directly.
- `_parse_watchlist_line(line)` — parses `EXCHANGE:TICKER,EXCHANGE:TICKER` (TradingView watchlist export format), converting `OSLO:` to a `.OL` yfinance suffix. Same convention as `Dashboard/data_loader.py`'s `parse_ticker_line`.

**`stoch_scan.py`** — live scanner + the shared Stochastic math:
- `compute_kd(df, stoch_length, k_smooth, d_smooth)` — resamples daily OHLC to Friday-anchored weekly bars and computes %K/%D. Returns a DataFrame (`close`, `k`, `d`) indexed by weekly bar. The final (possibly still-forming) week is included as-is, matching how TradingView plots a live weekly Stochastic — deliberate, not an oversight. `backtest.py` reuses this directly so the backtest and live scan always agree on the math.
- `is_stoch_sweet_spot(df, ...)` — calls `compute_kd`, then checks: latest %K inside `[k_min, k_max]`, %K > %D, %K rising vs. `k_rising_lookback` weeks back. Returns `(True, k, d)` or `(False, None, None)`.

**`backtest.py`** — trade simulation:
- `simulate_trades(weekly, entry_level=32.0, exit_level=80.0)` — pure function, no I/O. Takes a `compute_kd`-shaped DataFrame and returns a list of trade dicts. State machine, confirmed rules (see below). Fully unit-tested in isolation from OHLC/resampling concerns.
- `run_backtest(folder)` / `main()` — orchestration: `find_ticker_list` → `fetch_ohlc_bulk(period="max")` → `compute_kd` → `simulate_trades` per ticker → CSV + log + console summary. Wraps the per-ticker loop in try/except (one bad ticker doesn't abort the run) and de-dupes the ticker list before processing (a duplicate ticker would otherwise double-count its trades into the win-rate stat).

### Confirmed backtest rules (long-only)

- **Entry**: first week %K crosses above 32 (prior week <32, current week ≥32). Pure %K level-cross — %D not checked. Fill at that week's close.
- **Exit**: first week *after entry* where %K < %D **and** %K < 80 are simultaneously true. This holds through %K≥80 even if %K has already crossed below %D ("ride it"), and holds below 80 as long as %K stays above %D. Fill at that week's close.
- No stop loss. No portfolio/capital simulation — each trade is an isolated % return. A ticker can produce multiple sequential trades over its history.
- Still-open positions when data ends are recorded as `open`, marked-to-market at the last close, excluded from win/loss stats.
- **Not yet implemented**: short side, parameter sweep across Stochastic settings, a trailing-stop/breakeven variant of the exit rule (see Open Threads below).

## Folder Structure

- `INPUT/` — drop a TradingView Watchlist export here for `backtest.py` to pick up (see the two accepted filename patterns above). Falls back to the screener CSV in the project root if empty.
- `OUTPUT/` — `backtest.py`'s trades CSV + log land here. Gitignored.
- Both created automatically (with `.gitkeep`) by `backtest.py` if missing.

## Docs

- `docs/superpowers/specs/2026-07-14-stoch-backtester-design.md` — backtester design spec (entry/exit rules, folder structure, out-of-scope items)
- `docs/superpowers/plans/2026-07-14-stoch-backtester.md` — implementation plan (historical record of the 7-task build)

## Domain Vocabulary

- **Sweet spot band** — %K between 32 and 80 on the *weekly* Stochastic (19/4/4). The strategy's core signal zone.
- **%K / %D** — Stochastic fast line (red) / slow signal line (yellow) in the strategy doc's terminology.
- **Weekly Stoch** — always the 19/4/4 weekly Stochastic, computed off daily OHLC resampled to `W-FRI` bins. Never the daily 10/3/3 Stochastic mentioned in the strategy doc (shown only for contrast there, not used for decisions).

## Open Threads (as of 2026-07-14)

Backtest of the confirmed mechanical rules across a 30-ticker watchlist (1,060 trades, full history): **43.9% win rate**, avg win +19.6%, avg loss -6.6%, avg hold ~8.8 weeks — notably below the strategy doc's claimed ~70% win rate, but with a strong enough win/loss skew that expectancy is still positive.

User observation from chart review: the mechanical single-shot exit appears to cut many trades short during otherwise-intact uptrends (visible on WT's chart — multiple exit signals fire within one larger trend). Under discussion, not yet built: a trailing-stop variant — move stop to breakeven on the first exit signal, then ratchet to each subsequent confirmed swing low, only truly exiting on a stop hit rather than the raw indicator signal. This would need to be spec'd (swing-low detection logic) and backtested against the same sample before deciding whether to adopt it. Also still pending: the "approaching 32" tightened live-scanner filter (deferred from the backtester work, kept as a fully separate follow-up per user request), and the Stochastic-parameter sweep (length/%K-smooth/%D-smooth) to find better settings than 19/4/4.
