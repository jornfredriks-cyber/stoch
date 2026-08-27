import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from backtest_basic_trail import simulate_trades_basic_trail


def _weekly(rows):
    """rows: (date, close, k, d) tuples."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"close": [r[1] for r in rows], "k": [r[2] for r in rows], "d": [r[3] for r in rows]},
        index=idx,
    )


def _daily(rows):
    """rows: (date, close, low) tuples."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"close": [r[1] for r in rows], "low": [r[2] for r in rows]}, index=idx)


def test_never_enters_returns_empty_list():
    weekly = _weekly([
        ("2026-01-02", 100, 10, 8),
        ("2026-01-09", 102, 15, 12),
        ("2026-01-16", 101, 18, 16),
    ])
    daily = _daily([])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0)
    assert trades == []


def test_stop_touched_intraweek_exits_at_stop_price_not_the_close():
    # Entry Friday 2026-01-09 at close=100 -> stop=80. During the following
    # week the daily low dips to 78 on Wednesday (breaching the stop) but
    # the week's own CLOSE recovers to 90 (above the stop) by Friday. Under
    # the old weekly-close-only check this would have been missed entirely
    # (90 > 80 -> held); checking daily lows catches the real touch and
    # exits AT the stop price (80), not at the day's low (78) or the
    # week's close (90).
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry: stop=80
        ("2026-01-16", 90,  40, 32),  # close recovers above stop -- irrelevant, daily low already breached it
    ])
    daily = _daily([
        ("2026-01-12", 95, 95),
        ("2026-01-13", 88, 85),
        ("2026-01-14", 82, 78),  # low=78 <= stop(80) -> exit here at 80
        ("2026-01-15", 85, 83),
        ("2026-01-16", 90, 89),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "closed"
    assert t["exit_date"] == pd.Timestamp("2026-01-14")
    assert t["exit_price"] == 80.0
    assert t["stop_price"] == 80.0
    assert t["return_pct"] == pytest.approx(-20.0, abs=0.01)
    assert t["holding_days"] == 5


def test_stop_ratchets_across_weeks_and_exits_intraweek_at_new_stop():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry: stop=80
        ("2026-01-16", 130, 60, 45),  # week1 close=130, no intraweek touch -> ratchet stop to 104
        ("2026-01-23", 150, 70, 55),  # week2 close=150, no intraweek touch -> ratchet stop to 120
        ("2026-01-30", 118, 50, 60),  # week3 close irrelevant -- exits intraweek at 120
    ])
    daily = _daily([
        ("2026-01-12", 110, 105), ("2026-01-13", 120, 112), ("2026-01-14", 125, 118),
        ("2026-01-15", 128, 120), ("2026-01-16", 130, 126),
        ("2026-01-19", 140, 135), ("2026-01-20", 145, 138), ("2026-01-21", 148, 140),
        ("2026-01-22", 149, 142), ("2026-01-23", 150, 145),
        ("2026-01-26", 140, 130), ("2026-01-27", 125, 115),  # low=115 <= stop(120) -> exit here
        ("2026-01-28", 118, 113), ("2026-01-29", 118, 114), ("2026-01-30", 118, 116),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_date"] == pd.Timestamp("2026-01-27")
    assert t["exit_price"] == 120.0
    assert t["stop_price"] == 120.0
    assert t["return_pct"] == pytest.approx(20.0, abs=0.01)


def test_no_lookahead_a_surviving_weeks_low_cannot_trigger_next_weeks_ratcheted_stop():
    # Regression test for the core sequencing fix: week1's own daily lows
    # dip to 90 -- below what week1's OWN close (130) would ratchet the
    # stop up to (104) -- but must be checked against the stop as it stood
    # at the END OF THE PRIOR week (80, from entry), not a stop this same
    # week's close would newly justify. 90 > 80, so week1 survives clean,
    # and the stop only becomes 104 for week2 onward. Week2 has an
    # identical low of 90, which NOW breaches the (correctly, non-look-
    # ahead) 104 stop and exits there.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry: stop=80
        ("2026-01-16", 130, 50, 40),  # week1 close=130 -> would-be stop 104, but only takes effect next week
        ("2026-01-23", 135, 55, 45),  # week2: exits intraweek at the now-active 104 stop
    ])
    daily = _daily([
        ("2026-01-12", 95, 95), ("2026-01-13", 92, 90), ("2026-01-14", 98, 98),
        ("2026-01-15", 100, 100), ("2026-01-16", 130, 125),
        ("2026-01-19", 92, 90),  # low=90 <= stop(104, now active) -> exit here
        ("2026-01-20", 130, 128), ("2026-01-21", 132, 130),
        ("2026-01-22", 134, 132), ("2026-01-23", 135, 133),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_date"] == pd.Timestamp("2026-01-19")
    assert t["exit_price"] == 104.0
    assert t["return_pct"] == pytest.approx(4.0, abs=0.01)


def test_open_trade_marked_at_last_daily_close_when_never_stopped_out():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry: stop=80
        ("2026-01-16", 110, 50, 40),  # no intraweek touch, data ends
    ])
    daily = _daily([
        ("2026-01-12", 105, 102), ("2026-01-13", 108, 105), ("2026-01-14", 109, 106),
        ("2026-01-15", 110, 107), ("2026-01-16", 108, 106),  # last daily close = 108
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "open"
    assert t["exit_date"] is None
    assert t["exit_price"] is None
    assert t["return_pct"] == pytest.approx(8.0, abs=0.01)  # 108 (last daily close), not 110 (weekly close)


def test_sequential_trades_after_an_intraweek_stop_out():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # entry 1: stop=80
        ("2026-01-16", 90,  40, 32),  # exit 1 happens intraweek (see daily)
        ("2026-01-23", 80,  20, 25),  # flat
        ("2026-01-30", 90,  40, 30),  # entry 2: stop=72
        ("2026-02-06", 85,  25, 35),  # exit 2 happens intraweek (see daily)
    ])
    daily = _daily([
        ("2026-01-12", 88, 85), ("2026-01-13", 82, 79),  # low=79 <= stop(80) -> exit 1 here
        ("2026-01-14", 85, 82), ("2026-01-15", 88, 85), ("2026-01-16", 90, 87),
        ("2026-01-19", 82, 80), ("2026-01-20", 80, 78), ("2026-01-21", 83, 79),
        ("2026-01-22", 85, 82), ("2026-01-23", 80, 78),
        ("2026-01-26", 88, 85), ("2026-01-27", 90, 87), ("2026-01-28", 89, 86),
        ("2026-01-29", 88, 85), ("2026-01-30", 90, 87),
        ("2026-02-02", 85, 82), ("2026-02-03", 78, 71),  # low=71 <= stop(72) -> exit 2 here
        ("2026-02-04", 80, 77), ("2026-02-05", 82, 79), ("2026-02-06", 85, 82),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 2
    assert trades[0]["entry_date"] == pd.Timestamp("2026-01-09")
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-13")
    assert trades[1]["entry_date"] == pd.Timestamp("2026-01-30")
    assert trades[1]["exit_date"] == pd.Timestamp("2026-02-03")


def test_dropped_nan_week_still_has_its_daily_bars_checked():
    # The NaN k/d week (2026-01-23) is dropped from the weekly frame before
    # the loop runs, but its daily bars must NOT become a blind spot -- the
    # touch-check window is date-range-based off the raw daily frame, so it
    # still spans across the gap.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),                 # entry: stop=80
        ("2026-01-16", 110, 50, 40),                 # no touch -> ratchet stop to 88
        ("2026-01-23", 112, float("nan"), float("nan")),  # dropped
        ("2026-01-30", 115, 60, 55),                 # next surviving week
    ])
    daily = _daily([
        ("2026-01-12", 105, 102), ("2026-01-13", 108, 105), ("2026-01-14", 109, 106),
        ("2026-01-15", 110, 107), ("2026-01-16", 110, 108),
        ("2026-01-19", 100, 97), ("2026-01-20", 95, 92), ("2026-01-21", 90, 87),  # low=87 <= stop(88) -> exit here, inside the dropped week's date range
        ("2026-01-22", 90, 88), ("2026-01-23", 112, 110),
        ("2026-01-26", 113, 111), ("2026-01-27", 114, 112), ("2026-01-28", 114, 112),
        ("2026-01-29", 115, 113), ("2026-01-30", 115, 113),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_date"] == pd.Timestamp("2026-01-21")
    assert t["exit_price"] == 88.0
