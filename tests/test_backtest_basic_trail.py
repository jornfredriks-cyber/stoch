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
    """rows: (date, close, low) or (date, close, low, open) or (date, close, low, open, high) tuples."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({
        "close": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "open": [r[3] if len(r) > 3 else r[1] for r in rows],
        "high": [r[4] if len(r) > 4 else r[1] for r in rows],
    }, index=idx)


def _weekly_ohlc(rows):
    """rows: (date, high, low, close) tuples -- a precomputed weekly OHLC frame."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"high": [r[1] for r in rows], "low": [r[2] for r in rows], "close": [r[3] for r in rows]},
        index=idx,
    )


def test_never_enters_returns_empty_list():
    weekly = _weekly([
        ("2026-01-02", 100, 10, 8),
        ("2026-01-09", 102, 15, 12),
        ("2026-01-16", 101, 18, 16),
    ])
    daily = _daily([])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, breakout_window_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, breakout_window_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, breakout_window_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, breakout_window_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, breakout_window_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, breakout_window_days=0)
    assert len(trades) == 2
    assert trades[0]["entry_date"] == pd.Timestamp("2026-01-09")
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-13")
    assert trades[1]["entry_date"] == pd.Timestamp("2026-01-30")
    assert trades[1]["exit_date"] == pd.Timestamp("2026-02-03")


def test_breakout_above_weekly_pivot_fills_at_the_pivot_price():
    # 9 weeks of weekly OHLC ending at the signal date: a swing-high pivot
    # of 120 sits 3 weeks before AND after itself (100,105,110 | 120 |
    # 115,110,95) -- the only bar in range with 3 confirmed lower weeks on
    # both sides, so it's the "last significant high." Signal fires on
    # 2026-01-02; scanning forward, day 8 (2026-01-14) is the first day
    # whose HIGH reaches 120 -> filled there, at 120 (the level itself,
    # not the day's actual high of 121).
    weekly = _weekly([
        ("2025-12-26", 100, 20, 25),
        ("2026-01-02", 100, 35, 28),  # signal
    ])
    weekly_ohlc = _weekly_ohlc([
        ("2025-11-07", 100, 90, 95), ("2025-11-14", 105, 95, 100), ("2025-11-21", 110, 100, 105),
        ("2025-11-28", 120, 105, 115),  # the pivot
        ("2025-12-05", 115, 100, 105), ("2025-12-12", 110, 95, 100), ("2025-12-19", 95, 85, 90),
        ("2025-12-26", 90, 80, 85), ("2026-01-02", 100, 90, 95),
    ])
    daily = _daily([
        ("2026-01-05", 98,  95,  98,  102),
        ("2026-01-06", 100, 97,  98,  105),
        ("2026-01-07", 103, 100, 100, 108),
        ("2026-01-08", 105, 102, 103, 110),
        ("2026-01-09", 108, 105, 105, 112),
        ("2026-01-12", 110, 107, 108, 115),
        ("2026-01-13", 113, 110, 110, 118),
        ("2026-01-14", 119, 112, 113, 121),  # day 8: high=121 >= 120 -> BREAKOUT, fill at 120
    ])
    trades = simulate_trades_basic_trail(
        daily, weekly, entry_level=32.0, trail_pct=20.0,
        breakout_window_days=10, weekly_ohlc=weekly_ohlc, pivot_weeks=3, lookback_weeks=26,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_date"] == pd.Timestamp("2026-01-14")
    assert t["entry_price"] == 120
    assert t["entry_k"] == 35  # still reports the signal week's %K


def test_no_qualifying_pivot_skips():
    # Only 5 weeks of weekly_ohlc history -- fewer than the 7 needed for a
    # pivot_weeks=3 pivot (3 on each side + the pivot itself). No crash,
    # no trade.
    weekly = _weekly([
        ("2025-12-26", 100, 20, 25),
        ("2026-01-02", 100, 35, 28),
    ])
    weekly_ohlc = _weekly_ohlc([
        ("2025-12-05", 100, 90, 95), ("2025-12-12", 105, 95, 100), ("2025-12-19", 102, 92, 98),
        ("2025-12-26", 98, 88, 93), ("2026-01-02", 100, 90, 95),
    ])
    daily = _daily([])
    trades = simulate_trades_basic_trail(
        daily, weekly, entry_level=32.0, trail_pct=20.0,
        breakout_window_days=10, weekly_ohlc=weekly_ohlc, pivot_weeks=3, lookback_weeks=26,
    )
    assert trades == []


def test_breakout_never_touched_within_window_no_trade():
    # Same 120 pivot as above, but every day's high in the 10-day window
    # stays below it -- "No trade!" (the exact case the user validated by
    # hand: a setup that never clears resistance gets correctly skipped).
    weekly = _weekly([
        ("2025-12-26", 100, 20, 25),
        ("2026-01-02", 100, 35, 28),
    ])
    weekly_ohlc = _weekly_ohlc([
        ("2025-11-07", 100, 90, 95), ("2025-11-14", 105, 95, 100), ("2025-11-21", 110, 100, 105),
        ("2025-11-28", 120, 105, 115),
        ("2025-12-05", 115, 100, 105), ("2025-12-12", 110, 95, 100), ("2025-12-19", 95, 85, 90),
        ("2025-12-26", 90, 80, 85), ("2026-01-02", 100, 90, 95),
    ])
    daily = _daily([
        (f"2026-01-{d:02d}", 105, 100, 103, 110) for d in [5, 6, 7, 8, 9, 12, 13, 14, 15, 16]
    ])  # 10 days, all highs=110, never reaches the 120 pivot
    trades = simulate_trades_basic_trail(
        daily, weekly, entry_level=32.0, trail_pct=20.0,
        breakout_window_days=10, weekly_ohlc=weekly_ohlc, pivot_weeks=3, lookback_weeks=26,
    )
    assert trades == []


def test_breakout_after_window_expires_does_not_count():
    # Same 120 pivot; window is only 5 trading days. The breakout touch
    # (day 8's high=121) exists in the data but falls OUTSIDE the 5-day
    # window -- proves the window is a hard cutoff, not "eventually find
    # a touch however far out."
    weekly = _weekly([
        ("2025-12-26", 100, 20, 25),
        ("2026-01-02", 100, 35, 28),
    ])
    weekly_ohlc = _weekly_ohlc([
        ("2025-11-07", 100, 90, 95), ("2025-11-14", 105, 95, 100), ("2025-11-21", 110, 100, 105),
        ("2025-11-28", 120, 105, 115),
        ("2025-12-05", 115, 100, 105), ("2025-12-12", 110, 95, 100), ("2025-12-19", 95, 85, 90),
        ("2025-12-26", 90, 80, 85), ("2026-01-02", 100, 90, 95),
    ])
    daily = _daily([
        ("2026-01-05", 98,  95,  98,  102),
        ("2026-01-06", 100, 97,  98,  105),
        ("2026-01-07", 103, 100, 100, 108),
        ("2026-01-08", 105, 102, 103, 110),
        ("2026-01-09", 108, 105, 105, 112),
        ("2026-01-12", 110, 107, 108, 115),
        ("2026-01-13", 113, 110, 110, 118),
        ("2026-01-14", 119, 112, 113, 121),  # day 8: would touch, but window=5 stops after day 5
    ])
    trades = simulate_trades_basic_trail(
        daily, weekly, entry_level=32.0, trail_pct=20.0,
        breakout_window_days=5, weekly_ohlc=weekly_ohlc, pivot_weeks=3, lookback_weeks=26,
    )
    assert trades == []


def test_pre_breakout_days_cannot_affect_the_subsequent_exit():
    # Full round trip: enters via the day-8 breakout at 120 (same setup as
    # the first test), including a deliberately deep LOW on a pre-entry
    # day (2026-01-07, low=10) that must have zero effect -- the
    # breakout scan only ever reads HIGHs pre-entry, never lows, and no
    # position exists yet anyway. Then the trailing stop behaves exactly
    # as in the non-breakout exit tests: survives week 1 (close=130 ->
    # ratchet to 104), exits intraweek in week 2.
    weekly = _weekly([
        ("2025-12-26", 100, 20, 25),
        ("2026-01-02", 100, 35, 28),   # signal
        ("2026-01-16", 130, 50, 40),   # first weekly row after entry -> ratchet stop to 104
        ("2026-01-23", 92,  40, 45),   # exits intraweek
    ])
    weekly_ohlc = _weekly_ohlc([
        ("2025-11-07", 100, 90, 95), ("2025-11-14", 105, 95, 100), ("2025-11-21", 110, 100, 105),
        ("2025-11-28", 120, 105, 115),
        ("2025-12-05", 115, 100, 105), ("2025-12-12", 110, 95, 100), ("2025-12-19", 95, 85, 90),
        ("2025-12-26", 90, 80, 85), ("2026-01-02", 100, 90, 95),
    ])
    daily = _daily([
        ("2026-01-05", 98,  95,  98,  102),
        ("2026-01-06", 100, 97,  98,  105),
        ("2026-01-07", 103, 10,  100, 108),   # pre-entry: low=10, deep, must be IGNORED
        ("2026-01-08", 105, 102, 103, 110),
        ("2026-01-09", 108, 105, 105, 112),
        ("2026-01-12", 110, 107, 108, 115),
        ("2026-01-13", 113, 110, 110, 118),
        ("2026-01-14", 119, 112, 113, 121),   # day 8: BREAKOUT, fill at 120, stop=96
        ("2026-01-15", 122, 118, 119, 125),   # post-entry: low=118 > stop(96) -> hold
        ("2026-01-16", 124, 120, 122, 126),   # post-entry Friday: low=120 > 96 -> hold; week survives, ratchet to 104
        ("2026-01-19", 126, 110, 124, 128),   # low=110 > stop(104) -> hold
        ("2026-01-20", 108, 100, 115, 115),   # low=100 <= stop(104) -> exit here at 104
    ])
    trades = simulate_trades_basic_trail(
        daily, weekly, entry_level=32.0, trail_pct=20.0,
        breakout_window_days=10, weekly_ohlc=weekly_ohlc, pivot_weeks=3, lookback_weeks=26,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_date"] == pd.Timestamp("2026-01-14")
    assert t["entry_price"] == 120
    assert t["exit_date"] == pd.Timestamp("2026-01-20")
    assert t["exit_price"] == pytest.approx(104.0)
    assert t["return_pct"] == pytest.approx(-13.33, abs=0.01)


def test_multiweek_breakout_cannot_produce_an_exit_before_its_own_entry():
    # Regression test for a real bug found against live IREN data: when
    # the breakout takes longer than one calendar week to trigger (the
    # window is ~2 weeks, so this is common, not an edge case), the very
    # next weekly row after the signal (2026-01-09) is EARLIER than the
    # actual fill date (2026-01-15) -- it falls in the gap the breakout
    # window spans. A day between that skipped weekly row and the real
    # entry (2026-01-13, low=10 -- deep, deliberately) must NOT be
    # scanned for a stop touch; only after 2026-01-15 (the real entry)
    # should the exit touch-check window start. Before the fix, this
    # produced a reported exit_date BEFORE its own entry_date.
    weekly = _weekly([
        ("2025-12-26", 100, 20, 25),
        ("2026-01-02", 100, 35, 28),   # signal
        ("2026-01-09", 105, 40, 30),   # skipped entirely -- covered by the later fill date
        ("2026-01-16", 108, 45, 35),   # fill date (01-15) falls within this week
        ("2026-01-23", 92,  40, 45),   # exits intraweek
    ])
    weekly_ohlc = _weekly_ohlc([
        ("2025-11-07", 100, 90, 95), ("2025-11-14", 105, 95, 100), ("2025-11-21", 110, 100, 105),
        ("2025-11-28", 120, 105, 115),
        ("2025-12-05", 115, 100, 105), ("2025-12-12", 110, 95, 100), ("2025-12-19", 95, 85, 90),
        ("2025-12-26", 90, 80, 85), ("2026-01-02", 100, 90, 95),
    ])
    daily = _daily([
        ("2026-01-05", 98,  95,  98,  102),
        ("2026-01-06", 100, 98,  99,  105),
        ("2026-01-07", 103, 100, 100, 108),
        ("2026-01-08", 105, 102, 103, 110),
        ("2026-01-09", 108, 105, 105, 112),
        ("2026-01-12", 110, 107, 108, 115),
        ("2026-01-13", 113, 10,  110, 118),   # TRAP: low=10, after the skipped week but before the real entry -- must be IGNORED
        ("2026-01-14", 116, 112, 113, 119),
        ("2026-01-15", 119, 115, 116, 121),   # day 9: high=121 >= 120 -> BREAKOUT, fill at 120
        ("2026-01-16", 108, 118, 119, 122),   # post-entry Friday: low=118 > stop(96) -> hold
        ("2026-01-19", 106, 100, 105, 108),   # low=100 > stop(96) -> hold
        ("2026-01-20", 95,  90,  95,  100),   # low=90 <= stop(96) -> exit here at 96
    ])
    trades = simulate_trades_basic_trail(
        daily, weekly, entry_level=32.0, trail_pct=20.0,
        breakout_window_days=10, weekly_ohlc=weekly_ohlc, pivot_weeks=3, lookback_weeks=26,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_date"] == pd.Timestamp("2026-01-15")
    assert t["entry_price"] == 120
    assert t["exit_date"] == pd.Timestamp("2026-01-20")
    assert t["exit_date"] > t["entry_date"]
    assert t["exit_price"] == pytest.approx(96.0)
    assert t["return_pct"] == pytest.approx(-20.0, abs=0.01)


def test_breakout_window_days_zero_enters_immediately_as_before():
    # Default/off behavior must reproduce the no-breakout-filter results
    # exactly.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),
        ("2026-01-16", 90,  40, 32),
    ])
    daily = _daily([
        ("2026-01-12", 95, 95),
        ("2026-01-13", 88, 85),
        ("2026-01-14", 82, 78),  # low=78 <= stop(80) -> exit here at 80
        ("2026-01-15", 85, 83),
        ("2026-01-16", 90, 89),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, breakout_window_days=0)
    assert len(trades) == 1
    assert trades[0]["entry_date"] == pd.Timestamp("2026-01-09")
    assert trades[0]["entry_price"] == 100
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-14")


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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, breakout_window_days=0)
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_date"] == pd.Timestamp("2026-01-21")
    assert t["exit_price"] == 88.0
