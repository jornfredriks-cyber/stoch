import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from backtest_basic_trail import simulate_trades_basic_trail


def _weekly(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"close": [r[1] for r in rows], "k": [r[2] for r in rows], "d": [r[3] for r in rows]},
        index=idx,
    )


def test_never_enters_returns_empty_list():
    weekly = _weekly([
        ("2026-01-02", 100, 10, 8),
        ("2026-01-09", 102, 15, 12),
        ("2026-01-16", 101, 18, 16),
    ])
    trades = simulate_trades_basic_trail(weekly, entry_level=32.0, trail_pct=20.0)
    assert trades == []


def test_single_trade_stopped_out_at_a_loss():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry: K crosses above 32, close=100, stop=80
        ("2026-01-16", 90,  40, 32),  # 90 > stop(80) -> hold; highest still 100, stop still 80
        ("2026-01-23", 75,  30, 33),  # 75 <= stop(80) -> exit at 75 (gapped below the stop level)
    ])
    trades = simulate_trades_basic_trail(weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "closed"
    assert t["entry_date"] == pd.Timestamp("2026-01-09")
    assert t["entry_price"] == 100
    assert t["entry_k"] == 35
    assert t["exit_date"] == pd.Timestamp("2026-01-23")
    assert t["exit_price"] == 75
    assert t["stop_price"] == 80
    assert t["return_pct"] == pytest.approx(-25.0, abs=0.01)
    assert t["holding_weeks"] == 2


def test_stop_ratchets_up_and_exits_at_a_gain():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry: close=100, stop=80
        ("2026-01-16", 130, 60, 45),  # highest=130, stop=104 -> hold
        ("2026-01-23", 150, 70, 55),  # highest=150, stop=120 -> hold
        ("2026-01-30", 115, 50, 60),  # 115 <= stop(120) -> exit at 115, still a win vs entry(100)
    ])
    trades = simulate_trades_basic_trail(weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "closed"
    assert t["exit_date"] == pd.Timestamp("2026-01-30")
    assert t["exit_price"] == 115
    assert t["stop_price"] == 120
    assert t["return_pct"] == pytest.approx(15.0, abs=0.01)


def test_open_trade_when_never_stopped_out():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry: close=100, stop=80
        ("2026-01-16", 110, 50, 40),  # highest=110, stop=88 -> hold, data ends
    ])
    trades = simulate_trades_basic_trail(weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "open"
    assert t["exit_date"] is None
    assert t["exit_price"] is None
    assert t["return_pct"] == pytest.approx(10.0, abs=0.01)
    assert t["holding_weeks"] == 1


def test_exit_exactly_at_stop_boundary_is_inclusive():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry: close=100, stop=80
        ("2026-01-16", 80,  25, 30),  # close == stop(80) exactly -> exit, not held
    ])
    trades = simulate_trades_basic_trail(weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    assert trades[0]["status"] == "closed"
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-16")
    assert trades[0]["return_pct"] == pytest.approx(-20.0, abs=0.01)


def test_sequential_trades_after_a_stop_out():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry 1: close=100, stop=80
        ("2026-01-16", 75,  30, 33),  # exit 1: 75 <= 80
        ("2026-01-23", 80,  20, 25),  # flat
        ("2026-01-30", 90,  40, 30),  # entry 2: close=90, stop=72
        ("2026-02-06", 65,  25, 35),  # exit 2: 65 <= 72
    ])
    trades = simulate_trades_basic_trail(weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 2
    assert trades[0]["entry_date"] == pd.Timestamp("2026-01-09")
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-16")
    assert trades[1]["entry_date"] == pd.Timestamp("2026-01-30")
    assert trades[1]["exit_date"] == pd.Timestamp("2026-02-06")


def test_skips_nan_k_or_d_rows():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),               # entry: close=100, stop=80
        ("2026-01-16", 110, 50, 40),                # hold
        ("2026-01-23", 112, float("nan"), float("nan")),  # NaN -- must be skipped
        ("2026-01-30", 115, 60, 55),                # hold
        ("2026-02-06", 90,  40, 45),                # 90 <= stop(max(80,115*0.8=92)=92) -> exit
    ])
    trades = simulate_trades_basic_trail(weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_date"] == pd.Timestamp("2026-01-09")
    assert t["exit_date"] == pd.Timestamp("2026-02-06")
    assert t["holding_weeks"] == 3
