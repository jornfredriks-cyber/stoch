import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from backtest_signal_sltp_sweep import simulate_trades_signal_sltp


def _weekly(rows):
    """rows: (date, close, k, d, ema_long_fast, ema_long_slow) tuples."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({
        "close": [r[1] for r in rows],
        "k": [r[2] for r in rows],
        "d": [r[3] for r in rows],
        "ema_long_fast": [r[4] for r in rows],
        "ema_long_slow": [r[5] for r in rows],
    }, index=idx)


def _daily(rows):
    """rows: (date, close, high, low) tuples."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({
        "close": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
    }, index=idx)


def test_stop_loss_hit_intraweek_exits_at_stop_price():
    # Entry Monday 2026-01-09 week (weekly bar labeled Friday) at close=100.
    # SL=5% -> stop=95. Following week's Wednesday low touches 94 -> exit
    # there at exactly 95 (the stop price, not the day's low).
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25, 50, 50),
        ("2026-01-09", 100, 35, 28, 51, 50),   # entry: prev_k=20<32, k=35>32, k>d
        ("2026-01-16", 90,  50, 45, 51, 50),   # would still be "in position" going in
    ])
    daily = _daily([
        ("2026-01-12", 99, 100, 98),
        ("2026-01-13", 97, 99, 96),
        ("2026-01-14", 96, 98, 94),   # low=94 breaches stop=95 -> exit here
        ("2026-01-15", 95, 96, 93),
        ("2026-01-16", 90, 91, 89),
    ])
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=5.0, tp_r=2.0, filter_type="none")
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_reason"] == "SL"
    assert t["exit_date"] == pd.Timestamp("2026-01-14")
    assert t["exit_price"] == pytest.approx(95.0)
    assert t["status"] == "closed"


def test_take_profit_hit_intraweek_exits_at_target_price():
    # Entry at close=100, SL=5% -> stop=95, risk=5, TP=2.0R -> target=110.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25, 50, 50),
        ("2026-01-09", 100, 35, 28, 51, 50),
        ("2026-01-16", 108, 60, 55, 51, 50),
    ])
    daily = _daily([
        ("2026-01-12", 103, 105, 102),
        ("2026-01-13", 106, 108, 105),
        ("2026-01-14", 109, 111, 108),   # high=111 breaches target=110 -> exit here
        ("2026-01-15", 108, 109, 107),
        ("2026-01-16", 108, 109, 107),
    ])
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=5.0, tp_r=2.0, filter_type="none")
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_reason"] == "TP"
    assert t["exit_date"] == pd.Timestamp("2026-01-14")
    assert t["exit_price"] == pytest.approx(110.0)


def test_same_day_sl_and_tp_touch_sl_wins():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25, 50, 50),
        ("2026-01-09", 100, 35, 28, 51, 50),
        ("2026-01-16", 100, 60, 55, 51, 50),
    ])
    daily = _daily([
        ("2026-01-12", 100, 111, 94),   # touches both target(110) and stop(95) same day
    ])
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=5.0, tp_r=2.0, filter_type="none")
    assert trades[0]["exit_reason"] == "SL"


def test_signal_exit_when_no_sl_tp_touch_that_week():
    # No daily bar breaches SL(95) or TP(110); the weekly sell signal
    # (prev_k>80, k<80, k<d) fires instead, at that week's close.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25, 50, 50),
        ("2026-01-09", 100, 35, 28, 51, 50),   # entry
        ("2026-01-16", 105, 90, 60, 51, 50),   # holding, K>80
        ("2026-01-23", 103, 75, 90, 51, 50),   # sell signal: prev_k=90>80, k=75<80, k<d
    ])
    daily = _daily([
        ("2026-01-12", 101, 102, 100),
        ("2026-01-13", 102, 103, 101),
        ("2026-01-19", 104, 106, 103),
        ("2026-01-20", 105, 107, 104),
        ("2026-01-22", 103, 105, 102),
    ])
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=5.0, tp_r=2.0, filter_type="none")
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_reason"] == "signal"
    assert t["exit_date"] == pd.Timestamp("2026-01-23")
    assert t["exit_price"] == 103


def test_long_trend_filter_blocks_entry_when_fast_not_above_slow():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25, 45, 50),
        ("2026-01-09", 100, 35, 28, 46, 50),   # crossover fires, but ema_long_fast(46)<ema_long_slow(50)
    ])
    daily = _daily([("2026-01-12", 100, 101, 99)])
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=5.0, tp_r=2.0, filter_type="long_trend")
    assert trades == []


def test_long_trend_filter_allows_entry_when_fast_above_slow():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25, 55, 50),
        ("2026-01-09", 100, 35, 28, 56, 50),   # ema_long_fast(56)>ema_long_slow(50) -> allowed
    ])
    daily = _daily([("2026-01-12", 100, 101, 99)])
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=5.0, tp_r=2.0, filter_type="long_trend")
    assert len(trades) == 1
    assert trades[0]["status"] == "open"


def test_rmultiple_target_math():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25, 50, 50),
        ("2026-01-09", 200, 35, 28, 51, 50),   # entry_price=200, sl_pct=10 -> stop=180, risk=20
        ("2026-01-16", 200, 40, 35, 51, 50),
    ])
    daily = _daily([("2026-01-12", 200, 240.1, 199)])  # target = 200 + 1.5*20 = 230; high 240.1 > 230
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=10.0, tp_r=1.5, filter_type="none")
    assert trades[0]["exit_reason"] == "TP"
    assert trades[0]["exit_price"] == pytest.approx(230.0)


def test_open_trade_when_never_exits_marked_to_market_at_last_daily_close():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25, 50, 50),
        ("2026-01-09", 100, 35, 28, 51, 50),
        ("2026-01-16", 102, 45, 40, 51, 50),
    ])
    daily = _daily([
        ("2026-01-12", 101, 102, 100),
        ("2026-01-16", 102, 103, 101),
    ])
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=5.0, tp_r=2.0, filter_type="none")
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "open"
    assert t["exit_reason"] == "open"
    assert t["exit_price"] == pytest.approx(102.0)
    assert t["exit_date"] == pd.Timestamp("2026-01-16")


def test_never_enters_returns_empty_list():
    weekly = _weekly([
        ("2026-01-02", 100, 10, 8, 50, 50),
        ("2026-01-09", 102, 15, 12, 50, 50),
    ])
    daily = _daily([("2026-01-05", 100, 101, 99)])
    trades = simulate_trades_signal_sltp(daily, weekly, sl_pct=5.0, tp_r=2.0, filter_type="none")
    assert trades == []
