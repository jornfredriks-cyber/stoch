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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=0)
    assert len(trades) == 2
    assert trades[0]["entry_date"] == pd.Timestamp("2026-01-09")
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-13")
    assert trades[1]["entry_date"] == pd.Timestamp("2026-01-30")
    assert trades[1]["exit_date"] == pd.Timestamp("2026-02-03")


def test_limit_fills_the_day_after_confirmation_at_day3_close():
    # Signal Friday 2026-01-09 at close=100. confirm_days=3 -> day 3 after
    # the signal is Wed 2026-01-14. Its close (105) is above the signal
    # close (100) -> confirmed, 105 becomes a day-only limit price valid
    # for Thu 2026-01-15. Thu's low (103) touches it -> filled AT 105
    # (the limit price, not Thu's actual low), dated Thu -- NOT Wednesday.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # signal fires, but entry is deferred
        ("2026-01-16", 110, 50, 40),
    ])
    daily = _daily([
        ("2026-01-12", 98,  95),
        ("2026-01-13", 102, 99),
        ("2026-01-14", 105, 101),  # day 3: close=105 > signal close(100) -> CONFIRMED, limit=105
        ("2026-01-15", 106, 103),  # day 4: low=103 <= limit(105) -> FILLED here at 105
        ("2026-01-16", 108, 104),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=3)
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_date"] == pd.Timestamp("2026-01-15")
    assert t["entry_price"] == 105
    assert t["entry_k"] == 35  # still reports the signal week's %K


def test_limit_not_touched_next_day_the_trade_is_off():
    # Same confirmed limit (105), but day 4's low (108) never comes back
    # down to it -- the order expires unfilled, no trade, no retry on day
    # 5 or later.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),
        ("2026-01-16", 115, 50, 40),
    ])
    daily = _daily([
        ("2026-01-12", 98,  95),
        ("2026-01-13", 102, 99),
        ("2026-01-14", 105, 101),  # day 3: confirmed, limit=105
        ("2026-01-15", 112, 108),  # day 4: low=108 > limit(105) -> NOT filled, trade is off
        ("2026-01-16", 115, 110),  # would have touched 105 by now, but no retry
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=3)
    assert trades == []


def test_no_next_trading_day_after_confirmation_trade_is_off():
    # Day 3 confirms, but it's the LAST daily bar in the dataset -- no
    # day 4 exists to even attempt the fill. No crash, just no trade.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),
    ])
    daily = _daily([
        ("2026-01-12", 98,  95),
        ("2026-01-13", 102, 99),
        ("2026-01-14", 105, 101),  # day 3: confirmed, limit=105 -- but no day 4 follows
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=3)
    assert trades == []


def test_confirmation_fails_skips_the_signal_entirely():
    # Same signal, but day 3's close (95) is NOT above the signal close
    # (100) -> confirmation fails, no trade at all this cycle.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),
        ("2026-01-16", 90,  30, 32),
    ])
    daily = _daily([
        ("2026-01-12", 97, 94),
        ("2026-01-13", 96, 93),
        ("2026-01-14", 95, 92),  # day 3: close=95 <= signal close(100) -> NOT confirmed
        ("2026-01-15", 93, 90),
        ("2026-01-16", 90, 87),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=3)
    assert trades == []


def test_confirmation_skipped_when_insufficient_future_trading_days():
    # Signal fires on the very last weekly bar -- only 2 daily bars exist
    # after it, but confirm_days=3 needs a 3rd. No crash, just no trade.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),
    ])
    daily = _daily([
        ("2026-01-12", 102, 99),
        ("2026-01-13", 104, 101),
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=3)
    assert trades == []


def test_pre_entry_days_before_the_fill_cannot_trigger_a_stop():
    # Regression test: neither the confirmation window's daily lows (days
    # 1-3) NOR anything before the actual fill date (day 4) may ever be
    # scanned for a stop touch, even though they're far deeper than the
    # stop that gets established once the position actually exists.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 100, 35, 28),  # signal; confirmation+fill resolves entry on day 4
        ("2026-01-16", 108, 50, 40),  # first full week post-entry: survives -> ratchet
        ("2026-01-23", 92,  40, 45),  # exits intraweek at the ratcheted stop
    ])
    daily = _daily([
        ("2026-01-12", 90,  70),   # pre-confirmation: low=70, deep, must be IGNORED
        ("2026-01-13", 95,  75),   # pre-confirmation: low=75, must be IGNORED
        ("2026-01-14", 105, 65),   # day 3: close=105 > 100 -> CONFIRMED, limit=105; low=65 IGNORED (fill checks day 4, not day 3)
        ("2026-01-15", 106, 90),   # day 4: low=90 <= limit(105) -> FILLED here at 105, stop=84
        ("2026-01-16", 108, 90),   # post-entry: low=90 > stop(84) -> hold; week survives, ratchet to 86.4
        ("2026-01-19", 92,  88),   # low=88 > stop(86.4) -> hold
        ("2026-01-20", 90,  85),   # low=85 <= stop(86.4) -> exit here at 86.4
    ])
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=3)
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_date"] == pd.Timestamp("2026-01-15")
    assert t["entry_price"] == 105
    assert t["exit_date"] == pd.Timestamp("2026-01-20")
    assert t["exit_price"] == pytest.approx(86.4)
    assert t["return_pct"] == pytest.approx(-17.71, abs=0.01)


def test_confirm_days_zero_enters_immediately_as_before():
    # Default/off behavior must reproduce the pre-confirmation-filter
    # results exactly.
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=0)
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
    trades = simulate_trades_basic_trail(daily, weekly, entry_level=32.0, trail_pct=20.0, confirm_days=0)
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_date"] == pd.Timestamp("2026-01-21")
    assert t["exit_price"] == 88.0
