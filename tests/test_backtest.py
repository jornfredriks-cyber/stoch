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


def test_skips_nan_k_or_d_rows():
    weekly = _weekly([
        ("2025-12-05", 90, float("nan"), float("nan")),  # NaN k/d — rolling window not full yet
        ("2025-12-12", 95, float("nan"), float("nan")),  # NaN k/d — skipped
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # entry: K crosses above 32
        ("2026-01-16", 110, 50, 40),  # holding
        ("2026-01-23", 115, 60, 55),  # holding
        ("2026-01-30", 108, 58, 62),  # exit: K<D and K<80
    ])
    trades = simulate_trades(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_date"] == pd.Timestamp("2026-01-09")
    assert t["exit_date"] == pd.Timestamp("2026-01-30")
    assert t["holding_weeks"] == 3
