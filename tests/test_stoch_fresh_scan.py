import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from stoch_fresh_scan import is_fresh_stoch_buy_within_lookback


def _kd_frame(rows):
    """(date, close, k, d) rows -> a DataFrame shaped like compute_kd()'s output.
    The LAST row is always treated as the current, possibly still-forming
    week and dropped before any freshness check runs."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"close": [r[1] for r in rows], "k": [r[2] for r in rows], "d": [r[3] for r in rows]},
        index=idx,
    )


def test_crossover_1_week_ago_is_fresh(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 90,  15, 10),
        ("2026-01-09", 95,  20, 25),  # k_prev for weeks_ago=1: 20<32
        ("2026-01-16", 105, 35, 28),  # weeks_ago=1: 35>32, 35>28 -> fresh
        ("2026-01-23", 110, 40, 30),  # still-forming live week, must be dropped
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price, weeks_ago = is_fresh_stoch_buy_within_lookback(pd.DataFrame())
    assert fresh is True
    assert weeks_ago == 1
    assert k == 35 and d == 28
    assert entry_date == pd.Timestamp("2026-01-16")
    assert price == 105


def test_crossover_2_weeks_ago_is_fresh(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 90,  20, 25),  # k_prev for weeks_ago=2: 20<32
        ("2026-01-09", 100, 35, 28),  # weeks_ago=2: 35>32, 35>28 -> qualifies
        ("2026-01-16", 108, 38, 33),  # weeks_ago=1: k_prev=35 not<32 -> not fresh here
        ("2026-01-23", 112, 40, 34),  # still-forming live week, dropped
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price, weeks_ago = is_fresh_stoch_buy_within_lookback(pd.DataFrame())
    assert fresh is True
    assert weeks_ago == 2
    assert k == 35 and d == 28
    assert entry_date == pd.Timestamp("2026-01-09")


def test_crossover_3_weeks_ago_is_not_fresh(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 85,  15, 10),  # k_prev for the (unchecked) 3-weeks-ago event
        ("2026-01-09", 95,  35, 20),  # the crossover itself -- outside the 2-week window
        ("2026-01-16", 100, 40, 38),  # weeks_ago=2: k_prev=35 not<32 -> fails
        ("2026-01-23", 105, 42, 41),  # weeks_ago=1: k_prev=40 not<32 -> fails
        ("2026-01-30", 110, 44, 40),  # still-forming live week, dropped
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, *_ = is_fresh_stoch_buy_within_lookback(pd.DataFrame())
    assert fresh is False


def test_crossover_only_in_still_forming_week_not_fresh(monkeypatch):
    # The only qualifying transition is between the last COMPLETED week and
    # the still-forming live week. stoch_signal_scan.is_fresh_stoch_signal_buy()
    # would catch this (it treats the live bar as "now"); this function must
    # NOT, since the live week is dropped before any check runs.
    kd = _kd_frame([
        ("2026-01-02", 80, 15, 10),
        ("2026-01-09", 82, 18, 12),
        ("2026-01-16", 84, 20, 15),  # last completed week, still below 32
        ("2026-01-23", 95, 40, 25),  # live week: would be a fresh cross if counted
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, *_ = is_fresh_stoch_buy_within_lookback(pd.DataFrame())
    assert fresh is False


def test_boundary_exact_level_not_fresh(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 80, 10, 5),
        ("2026-01-09", 85, 20, 15),  # k_prev for weeks_ago=1: 20<32
        ("2026-01-16", 90, 32.0, 28),  # k==32 exactly -> strict > fails
        ("2026-01-23", 95, 33, 29),  # still-forming live week, dropped
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, *_ = is_fresh_stoch_buy_within_lookback(pd.DataFrame())
    assert fresh is False


def test_insufficient_history_returns_false(monkeypatch):
    # Only 1 completed week after the live week is dropped -- below the
    # lookback_weeks(2)+1 minimum required.
    kd = _kd_frame([
        ("2026-01-09", 95, 20, 15),
        ("2026-01-16", 105, 40, 30),  # live week, dropped
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price, weeks_ago = is_fresh_stoch_buy_within_lookback(pd.DataFrame())
    assert fresh is False
    assert k is None and d is None and entry_date is None and price is None and weeks_ago is None


def test_live_week_dropped_even_when_its_kd_is_nan(monkeypatch):
    # Regression test for drop-then-dropna ordering: the live week's k/d is
    # NaN (thin trailing data). It must still be excluded via .iloc[:-1]
    # BEFORE dropna(), not mistaken for a real completed week that dropna()
    # would otherwise strip.
    kd = _kd_frame([
        ("2026-01-02", 90,  15, 10),
        ("2026-01-09", 95,  20, 25),  # k_prev for weeks_ago=1: 20<32
        ("2026-01-16", 105, 35, 28),  # weeks_ago=1: 35>32, 35>28 -> fresh
        ("2026-01-23", 110, float("nan"), float("nan")),  # live week, NaN k/d
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price, weeks_ago = is_fresh_stoch_buy_within_lookback(pd.DataFrame())
    assert fresh is True
    assert weeks_ago == 1
    assert entry_date == pd.Timestamp("2026-01-16")


def test_level_crossed_but_k_not_above_d_not_fresh(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 80, 10, 5),
        ("2026-01-09", 85, 20, 15),  # k_prev for weeks_ago=1: 20<32
        ("2026-01-16", 90, 35, 40),  # k_now=35>32, but k_now<=d_now -> not fresh
        ("2026-01-23", 95, 36, 33),  # still-forming live week, dropped
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, *_ = is_fresh_stoch_buy_within_lookback(pd.DataFrame())
    assert fresh is False


def test_custom_entry_level_and_lookback_params_respected(monkeypatch):
    # A crossover of the default 32 level happened long before this window,
    # so with the default entry_level(32) neither checked week would look
    # "fresh" -- but overriding entry_level=50 makes the 1-week-ago
    # transition a genuine fresh cross, and lookback_weeks=1 restricts the
    # check to that single week.
    kd = _kd_frame([
        ("2026-01-09", 100, 45, 40),  # k_prev for weeks_ago=1 (entry_level=50): 45<50
        ("2026-01-16", 108, 55, 42),  # weeks_ago=1: 55>50, 55>42 -> fresh
        ("2026-01-23", 112, 58, 50),  # still-forming live week, dropped
    ])
    monkeypatch.setattr("stoch_fresh_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price, weeks_ago = is_fresh_stoch_buy_within_lookback(
        pd.DataFrame(), entry_level=50.0, lookback_weeks=1
    )
    assert fresh is True
    assert weeks_ago == 1
    assert k == 55 and d == 42
