# Stoch Backtester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backtester that simulates the Weekly Stochastic Sweet Spot strategy's long entry/exit rules across the Stoch project's ticker universe, using full historical data, and reports whether the strategy's claimed ~70% win rate holds up.

**Architecture:** A new `backtest.py` reuses `scan_utils.py` (extended with a ticker-list resolver and a `period` override on the existing bulk fetcher) and a `compute_kd` helper extracted from `stoch_scan.py` (so the backtest uses the exact same %K/%D math as the live scanner). The core simulation logic (`simulate_trades`) is a pure function over a precomputed weekly close/%K/%D frame — no I/O — so it's fully unit-testable. Orchestration (`run_backtest`/`main`) wires fetch → compute → simulate → CSV/log output, following the existing `screener.py`/`stoch_scan.py` pattern (`_Tee` logging, chunked bulk fetch).

**Tech Stack:** Python 3.14, pandas, yfinance, pytest (all already installed in `Stoch/venv/`).

## Global Constraints

- Never pass `threads=N` to `yf.download()` — Yahoo rate-limits concurrent requests and returns silent empty responses. `fetch_ohlc_bulk` already uses `threads=False`; preserve this.
- `tests/test_stoch_scan.py`'s existing 7 tests must pass **unmodified** after the `compute_kd` extraction in Task 3 — this is the regression gate proving the refactor didn't change live-scan behavior.
- No new config section: Stochastic settings and entry/exit levels are read from the existing `[stoch_scanner]` section of `stoch_config.ini` via `stoch_scan.py`'s module-level constants (`STOCH_LENGTH`, `STOCH_K_SMOOTH`, `STOCH_D_SMOOTH`, `STOCH_K_MIN` = 32.0, `STOCH_K_MAX` = 80.0).
- Output filenames must be `OUTPUT/Stoch_Backtest_Trades_YYYY-MM-DD.csv` and `OUTPUT/stoch_backtest_log_YYYY-MM-DD.txt` — these already match the `.gitignore` patterns committed in `Stoch/.gitignore` (`Stoch_Backtest_Trades_*.csv`, `stoch_backtest_log_*.txt`), so generated output stays untracked without further `.gitignore` changes.
- Long-only, no stop loss, no portfolio/capital simulation, no parameter sweep, no short side — all explicitly out of scope per the spec (`docs/superpowers/specs/2026-07-14-stoch-backtester-design.md`).
- Run all commands from `/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch` using `venv/bin/python3` (never system `python3`).

---

## Task 1: `fetch_ohlc_bulk` gains a `period` override

**Files:**
- Modify: `scan_utils.py:95-135`
- Test: `tests/test_scan_utils.py` (new)

**Interfaces:**
- Produces: `fetch_ohlc_bulk(yf_symbols: list[str], chunk_size: int = FETCH_CHUNK_SIZE, retries: int = FETCH_RETRIES, retry_delay: float = FETCH_RETRY_DELAY, period: str = HISTORY_PERIOD) -> dict[str, pd.DataFrame]` — new `period` kwarg, defaults preserve current 1y behavior for `screener.py`/`stoch_scan.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scan_utils.py`:

```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import scan_utils
from scan_utils import fetch_ohlc_bulk


def test_fetch_ohlc_bulk_uses_period_override(monkeypatch):
    captured = {}

    def fake_download(tickers, period, **kwargs):
        captured["period"] = period
        return pd.DataFrame()

    monkeypatch.setattr(scan_utils.yf, "download", fake_download)
    fetch_ohlc_bulk(["AAPL"], period="max")
    assert captured["period"] == "max"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/test_scan_utils.py::test_fetch_ohlc_bulk_uses_period_override -v`
Expected: FAIL with `TypeError: fetch_ohlc_bulk() got an unexpected keyword argument 'period'`

- [ ] **Step 3: Add the `period` parameter**

In `scan_utils.py`, find:

```python
def fetch_ohlc_bulk(
    yf_symbols: list[str],
    chunk_size: int = FETCH_CHUNK_SIZE,
    retries: int = FETCH_RETRIES,
    retry_delay: float = FETCH_RETRY_DELAY,
) -> dict[str, pd.DataFrame]:
```

Replace with:

```python
def fetch_ohlc_bulk(
    yf_symbols: list[str],
    chunk_size: int = FETCH_CHUNK_SIZE,
    retries: int = FETCH_RETRIES,
    retry_delay: float = FETCH_RETRY_DELAY,
    period: str = HISTORY_PERIOD,
) -> dict[str, pd.DataFrame]:
```

Then find, inside the same function:

```python
                downloaded = yf.download(
                    " ".join(chunk),
                    period=HISTORY_PERIOD,
                    interval="1d",
```

Replace with:

```python
                downloaded = yf.download(
                    " ".join(chunk),
                    period=period,
                    interval="1d",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python3 -m pytest tests/test_scan_utils.py::test_fetch_ohlc_bulk_uses_period_override -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scan_utils.py tests/test_scan_utils.py
git commit -m "feat: add period override to fetch_ohlc_bulk"
```

---

## Task 2: `find_ticker_list` — watchlist-first, screener-CSV-fallback ticker resolver

**Files:**
- Modify: `scan_utils.py:36-41` (insert after `find_latest_csv`)
- Test: `tests/test_scan_utils.py` (extend from Task 1)

**Interfaces:**
- Consumes: `find_latest_csv(folder: str) -> str` (existing, in `scan_utils.py`)
- Produces: `find_ticker_list(input_folder: str, fallback_folder: str) -> list[str]` — used by `backtest.py` in Task 5 as `find_ticker_list(input_folder=<Stoch>/INPUT, fallback_folder=<Stoch>)`.
- Produces: `_parse_watchlist_line(line: str) -> list[str]` — internal helper, also directly tested.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scan_utils.py`:

```python
from scan_utils import _parse_watchlist_line, find_ticker_list


def test_parse_watchlist_line_extracts_symbols():
    assert _parse_watchlist_line("NASDAQ:AAPL,NYSE:V") == ["AAPL", "V"]


def test_parse_watchlist_line_converts_oslo_suffix():
    assert _parse_watchlist_line("OSLO:EQNR") == ["EQNR.OL"]


def test_parse_watchlist_line_handles_plain_ticker():
    assert _parse_watchlist_line("AAPL") == ["AAPL"]


def test_parse_watchlist_line_skips_empty_entries():
    assert _parse_watchlist_line("AAPL,,  ,V") == ["AAPL", "V"]


def test_find_ticker_list_prefers_watchlist_file(tmp_path):
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    (input_dir / "My Watchlist.txt").write_text("NASDAQ:AAPL,NYSE:V")
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()

    result = find_ticker_list(str(input_dir), str(fallback_dir))
    assert result == ["AAPL", "V"]


def test_find_ticker_list_falls_back_to_screener_csv(tmp_path):
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    csv_path = fallback_dir / "Stoch Raw Screener_2026-07-14.csv"
    pd.DataFrame({"Symbol": ["AAPL", "MSFT"]}).to_csv(csv_path, index=False)

    result = find_ticker_list(str(input_dir), str(fallback_dir))
    assert result == ["AAPL", "MSFT"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/test_scan_utils.py -v`
Expected: the 6 new tests FAIL with `ImportError: cannot import name '_parse_watchlist_line'` (or `find_ticker_list`)

- [ ] **Step 3: Implement `_parse_watchlist_line` and `find_ticker_list`**

In `scan_utils.py`, find:

```python
def find_latest_csv(folder: str) -> str:
    files = glob.glob(os.path.join(folder, "*[Ss]creener*.csv"))
    if not files:
        raise FileNotFoundError(f"No screener CSV found in {folder}")
    return max(files, key=os.path.getmtime)


def _chunks(items: list[str], size: int):
```

Replace with:

```python
def find_latest_csv(folder: str) -> str:
    files = glob.glob(os.path.join(folder, "*[Ss]creener*.csv"))
    if not files:
        raise FileNotFoundError(f"No screener CSV found in {folder}")
    return max(files, key=os.path.getmtime)


NORWEGIAN_EXCHANGES = {"OSLO"}


def _parse_watchlist_line(line: str) -> list[str]:
    tickers = []
    for raw in line.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            exchange, symbol = raw.split(":", 1)
            exchange = exchange.strip().upper()
            symbol = symbol.strip()
            if exchange in NORWEGIAN_EXCHANGES:
                tickers.append(f"{symbol}.OL")
            else:
                tickers.append(symbol)
        else:
            tickers.append(raw)
    return tickers


def find_ticker_list(input_folder: str, fallback_folder: str) -> list[str]:
    """
    Looks for a *Watchlist* file (TradingView export: plain text,
    comma-separated EXCHANGE:TICKER) in input_folder. Falls back to the
    latest screener CSV in fallback_folder if no watchlist file exists.
    """
    watchlist_files = glob.glob(os.path.join(input_folder, "*Watchlist*"))
    if watchlist_files:
        watchlist_path = max(watchlist_files, key=os.path.getmtime)
        with open(watchlist_path) as f:
            content = f.read()
        tickers = []
        for line in content.splitlines():
            tickers.extend(_parse_watchlist_line(line))
        return tickers

    csv_path = find_latest_csv(fallback_folder)
    df = pd.read_csv(csv_path)
    return df["Symbol"].dropna().tolist()


def _chunks(items: list[str], size: int):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/test_scan_utils.py -v`
Expected: PASS (7 tests total: 1 from Task 1 + 6 new)

- [ ] **Step 5: Commit**

```bash
git add scan_utils.py tests/test_scan_utils.py
git commit -m "feat: add find_ticker_list watchlist resolver with screener-CSV fallback"
```

---

## Task 3: Extract `compute_kd` from `stoch_scan.py`

**Files:**
- Modify: `stoch_scan.py:31-82`
- Verify: `tests/test_stoch_scan.py` (existing, must pass unmodified — no edits to this file in this task)

**Interfaces:**
- Produces: `compute_kd(df: pd.DataFrame, stoch_length: int = STOCH_LENGTH, k_smooth: int = STOCH_K_SMOOTH, d_smooth: int = STOCH_D_SMOOTH) -> pd.DataFrame` — columns `close`, `k`, `d`, indexed by weekly bar (`W-FRI`). Empty/None input returns an empty DataFrame with those columns. `k`/`d` are `NaN` for weeks where the rolling windows aren't full yet. Consumed by `backtest.py` in Task 4/5.
- Preserves: `is_stoch_sweet_spot(df, stoch_length=..., k_smooth=..., d_smooth=..., k_min=..., k_max=..., k_rising_lookback=...) -> tuple` — same signature and behavior as before, now implemented in terms of `compute_kd`.

- [ ] **Step 1: Run the existing tests first to confirm the baseline**

Run: `venv/bin/python3 -m pytest tests/test_stoch_scan.py -v`
Expected: PASS (7 tests) — this is the behavior the refactor must not change.

- [ ] **Step 2: Extract `compute_kd` and rewrite `is_stoch_sweet_spot` in terms of it**

In `stoch_scan.py`, find:

```python
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
    default 19/4/4) has its latest %K value inside [k_min, k_max], with %K
    above %D and %K rising versus `k_rising_lookback` weeks back.

    Resamples the daily OHLC frame into Friday-anchored weekly bins
    (high=max, low=min, close=last). Unlike is_near_last_week_low(), the
    final bin is NOT dropped/deferred when the current week is still in
    progress — it is used as-is, matching how TradingView plots a live
    weekly Stochastic off the still-forming candle. This is a deliberate
    difference from lw_low_scan.py's completed-week convention, not an
    oversight.

    Returns:
        (True,  k_value, d_value)  — latest %K inside band, above %D, rising
        (False, None,    None)     — condition not met, or insufficient
                                       weekly history to compute both lines
    """
    if df is None or len(df) == 0:
        return False, None, None

    weekly = df.resample("W-FRI").agg({"high": "max", "low": "min", "close": "last"}).dropna()

    lowest_low   = weekly["low"].rolling(stoch_length).min()
    highest_high = weekly["high"].rolling(stoch_length).max()
    raw_k = 100 * (weekly["close"] - lowest_low) / (highest_high - lowest_low)
    k_line = raw_k.rolling(k_smooth).mean()
    d_line = k_line.rolling(d_smooth).mean()

    valid = d_line.dropna()
    if len(valid) < k_rising_lookback + 1:
        return False, None, None

    k_now  = k_line.loc[valid.index[-1]]
    k_prev = k_line.loc[valid.index[-1 - k_rising_lookback]]
    d_now  = valid.iloc[-1]

    qualifies = (k_min <= k_now <= k_max) and (k_now > d_now) and (k_now > k_prev)

    if qualifies:
        return True, round(k_now, 2), round(d_now, 2)

    return False, None, None
```

Replace with:

```python
def compute_kd(
    df: pd.DataFrame,
    stoch_length: int = STOCH_LENGTH,
    k_smooth: int = STOCH_K_SMOOTH,
    d_smooth: int = STOCH_D_SMOOTH,
) -> pd.DataFrame:
    """
    Resamples the daily OHLC frame into Friday-anchored weekly bins
    (high=max, low=min, close=last) and computes the weekly Stochastic
    %K/%D lines. Unlike is_near_last_week_low(), the final bin is NOT
    dropped/deferred when the current week is still in progress — it is
    used as-is, matching how TradingView plots a live weekly Stochastic
    off the still-forming candle. This is a deliberate difference from
    lw_low_scan.py's completed-week convention, not an oversight.

    Returns a DataFrame indexed by weekly bar with columns "close", "k",
    "d". %k/%d are NaN for the initial weeks where the rolling windows
    aren't full yet. Empty/None input returns an empty DataFrame with
    those columns.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["close", "k", "d"])

    weekly = df.resample("W-FRI").agg({"high": "max", "low": "min", "close": "last"}).dropna()

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
```

- [ ] **Step 3: Run the existing tests to confirm no regression**

Run: `venv/bin/python3 -m pytest tests/test_stoch_scan.py -v`
Expected: PASS (same 7 tests, unmodified, still passing)

- [ ] **Step 4: Commit**

```bash
git add stoch_scan.py
git commit -m "refactor: extract compute_kd from is_stoch_sweet_spot for reuse in backtester"
```

---

## Task 4: `simulate_trades` — the pure trade-simulation state machine

**Files:**
- Create: `backtest.py` (this task only adds `simulate_trades`; orchestration comes in Task 5)
- Test: `tests/test_backtest.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks — pure function over a `pd.DataFrame` with columns `close`, `k`, `d` (the shape `compute_kd` produces).
- Produces: `simulate_trades(weekly: pd.DataFrame, entry_level: float = 32.0, exit_level: float = 80.0) -> list[dict]`. Each dict has keys: `entry_date`, `entry_price`, `entry_k`, `exit_date`, `exit_price`, `exit_k`, `exit_d`, `status` (`"closed"` or `"open"`), `return_pct`, `holding_weeks`. For open trades, `exit_date`/`exit_price`/`exit_k`/`exit_d` are `None`. Consumed by `run_backtest` in Task 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest.py`:

```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from backtest import simulate_trades


def _weekly(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"close": [r[1] for r in rows], "k": [r[2] for r in rows], "d": [r[3] for r in rows]},
        index=idx,
    )


def test_single_clean_trade():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # entry: K crosses above 32
        ("2026-01-16", 110, 50, 40),  # holding
        ("2026-01-23", 115, 60, 55),  # holding
        ("2026-01-30", 108, 58, 62),  # exit: K<D and K<80
    ])
    trades = simulate_trades(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "closed"
    assert t["entry_date"] == pd.Timestamp("2026-01-09")
    assert t["entry_price"] == 105
    assert t["entry_k"] == 35
    assert t["exit_date"] == pd.Timestamp("2026-01-30")
    assert t["exit_price"] == 108
    assert t["exit_k"] == 58
    assert t["exit_d"] == 62
    assert t["return_pct"] == pytest.approx((108 / 105 - 1) * 100, abs=0.01)
    assert t["holding_weeks"] == 3


def test_sequential_trades():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # entry 1
        ("2026-01-16", 108, 58, 62),  # exit 1
        ("2026-01-23", 100, 20, 55),  # flat
        ("2026-01-30", 106, 40, 30),  # entry 2
        ("2026-02-06", 112, 55, 60),  # exit 2
    ])
    trades = simulate_trades(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 2
    assert trades[0]["entry_date"] == pd.Timestamp("2026-01-09")
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-16")
    assert trades[1]["entry_date"] == pd.Timestamp("2026-01-30")
    assert trades[1]["exit_date"] == pd.Timestamp("2026-02-06")


def test_holds_through_k_above_80_until_k_drops_below_80():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # entry
        ("2026-01-16", 120, 85, 70),  # K>=80, K>D -> hold
        ("2026-01-23", 118, 82, 90),  # K>=80, K<D already -> still hold ("ride it")
        ("2026-01-30", 110, 75, 88),  # K<80 now, K<D -> exit
    ])
    trades = simulate_trades(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 1
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-30")
    assert trades[0]["status"] == "closed"


def test_holds_while_k_below_80_until_k_crosses_below_d():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # entry
        ("2026-01-16", 108, 50, 40),  # K<80, K>D -> hold
        ("2026-01-23", 104, 45, 47),  # K<80, K<D -> exit
    ])
    trades = simulate_trades(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 1
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-23")


def test_open_trade_when_never_exits():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # entry
        ("2026-01-16", 110, 50, 40),  # holding
        ("2026-01-23", 115, 60, 55),  # holding, data ends
    ])
    trades = simulate_trades(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "open"
    assert t["exit_date"] is None
    assert t["return_pct"] == pytest.approx((115 / 105 - 1) * 100, abs=0.01)
    assert t["holding_weeks"] == 2


def test_never_enters_returns_empty_list():
    weekly = _weekly([
        ("2026-01-02", 100, 10, 8),
        ("2026-01-09", 102, 15, 12),
        ("2026-01-16", 101, 18, 16),
    ])
    trades = simulate_trades(weekly, entry_level=32.0, exit_level=80.0)
    assert trades == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest'`

- [ ] **Step 3: Create `backtest.py` with `simulate_trades`**

Create `backtest.py`:

```python
import pandas as pd


def simulate_trades(
    weekly: pd.DataFrame,
    entry_level: float = 32.0,
    exit_level: float = 80.0,
) -> list[dict]:
    """
    Pure state-machine over a precomputed weekly close/k/d frame (as
    produced by stoch_scan.compute_kd). No I/O.

    Entry: first week %K crosses above entry_level (prior week %K <
    entry_level, current week %K >= entry_level). Fill at that week's
    close. %D is not checked at entry.

    Exit: first week after entry where %K < %D AND %K < exit_level are
    both true simultaneously. This holds through %K >= exit_level even
    if %K has already crossed below %D ("ride it"), and holds below
    exit_level as long as %K remains above %D. Fill at that week's close.

    A ticker can produce multiple sequential trades (Flat -> Long -> Flat
    -> ...). If still in a position when the data ends, the trade is
    recorded as "open", marked-to-market at the last available close.
    Weeks where %K or %D is NaN (rolling windows not yet full) are
    skipped entirely.
    """
    weekly = weekly.dropna(subset=["k", "d"])

    trades: list[dict] = []
    in_position = False
    entry: dict | None = None
    prev_k = None

    for dt, row in weekly.iterrows():
        k, d, close = row["k"], row["d"], row["close"]

        if not in_position:
            if prev_k is not None and prev_k < entry_level <= k:
                in_position = True
                entry = {"entry_date": dt, "entry_price": close, "entry_k": round(k, 2)}
        else:
            if k < d and k < exit_level:
                trades.append({
                    "entry_date": entry["entry_date"],
                    "entry_price": entry["entry_price"],
                    "entry_k": entry["entry_k"],
                    "exit_date": dt,
                    "exit_price": close,
                    "exit_k": round(k, 2),
                    "exit_d": round(d, 2),
                    "status": "closed",
                    "return_pct": round((close / entry["entry_price"] - 1) * 100, 2),
                    "holding_weeks": weekly.index.get_loc(dt) - weekly.index.get_loc(entry["entry_date"]),
                })
                in_position = False
                entry = None

        prev_k = k

    if in_position:
        last_date = weekly.index[-1]
        last_close = weekly["close"].iloc[-1]
        trades.append({
            "entry_date": entry["entry_date"],
            "entry_price": entry["entry_price"],
            "entry_k": entry["entry_k"],
            "exit_date": None,
            "exit_price": None,
            "exit_k": None,
            "exit_d": None,
            "status": "open",
            "return_pct": round((last_close / entry["entry_price"] - 1) * 100, 2),
            "holding_weeks": weekly.index.get_loc(last_date) - weekly.index.get_loc(entry["entry_date"]),
        })

    return trades
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/test_backtest.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add backtest.py tests/test_backtest.py
git commit -m "feat: add simulate_trades state machine for backtest engine"
```

---

## Task 5: `run_backtest`/`main` — orchestration, INPUT/OUTPUT wiring, CSV + log output

**Files:**
- Modify: `backtest.py` (append orchestration below `simulate_trades`)

**Interfaces:**
- Consumes: `scan_utils.find_ticker_list`, `scan_utils.fetch_ohlc_bulk`, `scan_utils._Tee` (Tasks 1-2); `stoch_scan.compute_kd`, `stoch_scan.STOCH_LENGTH`, `stoch_scan.STOCH_K_SMOOTH`, `stoch_scan.STOCH_D_SMOOTH`, `stoch_scan.STOCH_K_MIN`, `stoch_scan.STOCH_K_MAX` (Task 3); `simulate_trades` (Task 4).
- Produces: `run_backtest(folder: str) -> None` (writes `OUTPUT/Stoch_Backtest_Trades_YYYY-MM-DD.csv`, prints summary); `main() -> None` (creates `INPUT/`/`OUTPUT/` with `.gitkeep` if missing, wraps `run_backtest` in `_Tee` logging to `OUTPUT/stoch_backtest_log_YYYY-MM-DD.txt`).

This task is I/O-heavy (network fetch, filesystem, logging) and is not unit-tested, matching how `screener.py`'s and `stoch_scan.py`'s `main`/`_run` functions are verified today: via the manual smoke test in Task 7, not automated tests.

- [ ] **Step 1: Add imports and orchestration to `backtest.py`**

At the top of `backtest.py`, find:

```python
import pandas as pd


def simulate_trades(
```

Replace with:

```python
import os
import sys
from datetime import date

import pandas as pd

from scan_utils import _Tee, fetch_ohlc_bulk, find_ticker_list
from stoch_scan import (
    STOCH_D_SMOOTH,
    STOCH_K_MAX,
    STOCH_K_MIN,
    STOCH_K_SMOOTH,
    STOCH_LENGTH,
    compute_kd,
)


def simulate_trades(
```

At the end of `backtest.py`, after the `simulate_trades` function, add:

```python


def run_backtest(folder: str):
    input_dir = os.path.join(folder, "INPUT")
    output_dir = os.path.join(folder, "OUTPUT")

    tickers = find_ticker_list(input_dir, folder)
    print(f"Tickers loaded : {len(tickers)}")
    print(f"Weekly Stoch   : {STOCH_LENGTH}/{STOCH_K_SMOOTH}/{STOCH_D_SMOOTH}"
          f"  |  Entry K>={STOCH_K_MIN}  |  Exit K<D and K<{STOCH_K_MAX}\n")

    yf_symbols = [t.replace(".", "-").replace("/", "-") for t in tickers]
    ohlc = fetch_ohlc_bulk(yf_symbols, period="max")

    all_trades = []
    for i, (sym, yf_sym) in enumerate(zip(tickers, yf_symbols), 1):
        df = ohlc.get(yf_sym)
        if df is None:
            print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  — no data")
            continue

        kd = compute_kd(df, STOCH_LENGTH, STOCH_K_SMOOTH, STOCH_D_SMOOTH)
        trades = simulate_trades(kd, entry_level=STOCH_K_MIN, exit_level=STOCH_K_MAX)
        for t in trades:
            t["ticker"] = sym
        all_trades.extend(trades)

        closed = sum(1 for t in trades if t["status"] == "closed")
        print(f"  [{i:3d}/{len(tickers)}] {sym:10s}  {len(trades)} trades ({closed} closed)")

    out_name = f"Stoch_Backtest_Trades_{date.today()}.csv"
    out_path = os.path.join(output_dir, out_name)
    columns = ["ticker", "entry_date", "entry_price", "entry_k",
               "exit_date", "exit_price", "exit_k", "exit_d",
               "status", "return_pct", "holding_weeks"]
    pd.DataFrame(all_trades, columns=columns).to_csv(out_path, index=False)

    closed_trades = [t for t in all_trades if t["status"] == "closed"]
    open_trades   = [t for t in all_trades if t["status"] == "open"]
    wins   = [t for t in closed_trades if t["return_pct"] > 0]
    losses = [t for t in closed_trades if t["return_pct"] <= 0]

    print(f"\n{'=' * 55}")
    print(f"Total trades  : {len(all_trades)}")
    print(f"Closed / Open : {len(closed_trades)} / {len(open_trades)}")
    if closed_trades:
        print(f"Win rate      : {len(wins) / len(closed_trades) * 100:.1f}%  ({len(wins)}/{len(closed_trades)})")
    if wins:
        print(f"Avg win       : {sum(t['return_pct'] for t in wins) / len(wins):.2f}%")
    if losses:
        print(f"Avg loss      : {sum(t['return_pct'] for t in losses) / len(losses):.2f}%")
    if closed_trades:
        avg_hold = sum(t["holding_weeks"] for t in closed_trades) / len(closed_trades)
        print(f"Avg hold (wk) : {avg_hold:.1f}")
    print(f"Saved → OUTPUT/{out_name}")


def main():
    folder     = os.path.dirname(os.path.abspath(__file__))
    input_dir  = os.path.join(folder, "INPUT")
    output_dir = os.path.join(folder, "OUTPUT")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    for d in (input_dir, output_dir):
        gitkeep = os.path.join(d, ".gitkeep")
        if not os.path.exists(gitkeep):
            open(gitkeep, "a").close()

    log_path = os.path.join(output_dir, f"stoch_backtest_log_{date.today()}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        run_backtest(folder)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Log saved  → OUTPUT/{os.path.basename(log_path)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file imports cleanly and existing tests still pass**

Run: `venv/bin/python3 -c "import backtest"`
Expected: no output, no error (confirms imports resolve and no syntax errors)

Run: `venv/bin/python3 -m pytest tests/ -v`
Expected: PASS — all tests from Tasks 1-4 (`test_scan_utils.py`, `test_stoch_scan.py`, `test_backtest.py`) still pass; this file introduces no new test cases but must not break existing ones.

- [ ] **Step 3: Commit**

```bash
git add backtest.py
git commit -m "feat: add run_backtest/main orchestration with INPUT/OUTPUT wiring"
```

---

## Task 6: `RUN Stoch Backtest.command` launcher

**Files:**
- Create: `RUN Stoch Backtest.command`

**Interfaces:**
- Consumes: `backtest.py` (Task 5)

- [ ] **Step 1: Create the launcher**

Create `RUN Stoch Backtest.command`:

```bash
#!/bin/zsh
# Double-click this file in Finder to run the Stoch Sweet Spot backtester.
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 backtest.py

echo ""
echo "Press any key to close this window..."
read -k 1
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x "RUN Stoch Backtest.command"`

- [ ] **Step 3: Verify**

Run: `ls -la "RUN Stoch Backtest.command"`
Expected: permissions show `-rwxr-xr-x`

- [ ] **Step 4: Commit**

```bash
git add "RUN Stoch Backtest.command"
git commit -m "feat: add RUN Stoch Backtest.command launcher"
```

---

## Task 7: End-to-end smoke test

**Files:** none (verification only)

**Interfaces:** none — this task exercises the full pipeline built in Tasks 1-6 against real data.

- [ ] **Step 1: Run the full test suite one more time**

Run: `venv/bin/python3 -m pytest tests/ -v`
Expected: PASS — all tests across `test_scan_utils.py`, `test_stoch_scan.py`, `test_backtest.py`

- [ ] **Step 2: Drop a small test watchlist into INPUT/**

Run:
```bash
mkdir -p INPUT
echo "NASDAQ:AAPL,NASDAQ:GOOGL,NYSE:COST" > "INPUT/Smoke Test Watchlist.txt"
```

(COST and GOOGL are the two tickers cited by name in `Weekly Stochastic Sweet Spot Strategy.md`'s worked examples — useful for eyeballing whether the trade dates look directionally sane.)

- [ ] **Step 3: Run the backtester**

Run: `venv/bin/python3 backtest.py`
Expected: console output shows 3 tickers loaded, per-ticker trade counts, and a summary block (total trades, closed/open, win rate, avg win/loss, avg hold). Exits 0.

- [ ] **Step 4: Inspect the output**

Run: `cat OUTPUT/Stoch_Backtest_Trades_*.csv`
Expected: CSV with header `ticker,entry_date,entry_price,entry_k,exit_date,exit_price,exit_k,exit_d,status,return_pct,holding_weeks` and multiple rows for AAPL/GOOGL/COST. At least one trade should have `status=open` (the current in-progress position, if any) or all closed if none is currently active — either is valid, just confirm the field values look sane (prices are real-looking numbers, `entry_k` is between 32 and ~40, dates are chronological, `return_pct` matches `(exit_price/entry_price-1)*100`).

- [ ] **Step 5: Remove the smoke-test watchlist**

Run: `rm "INPUT/Smoke Test Watchlist.txt"`

(Leaves `INPUT/` ready for a real watchlist drop later. `OUTPUT/` artifacts from the smoke run are gitignored, so no cleanup needed there — leave them as a reference or delete them, your choice.)

- [ ] **Step 6: Confirm nothing unintended is staged**

Run: `git status`
Expected: clean or only showing files intentionally left over (e.g. `OUTPUT/` contents, which are gitignored and won't appear); no unexpected modifications.
