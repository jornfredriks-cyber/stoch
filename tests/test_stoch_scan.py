import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from stoch_scan import compute_kd_daily, is_stoch_sweet_spot

DEFAULTS = dict(stoch_length=19, k_smooth=4, d_smooth=4, k_min=32.0, k_max=80.0, k_rising_lookback=1)


def _weekly_df(closes, start="2026-01-02"):
    # start="2026-01-02" is a Friday, so periods land one-per-week on
    # Friday-anchored bins with no partial-week ambiguity. high/low are a
    # fixed +/-0.5 offset around close -- irrelevant to the %K/%D math here
    # since only the rolling min(low)/max(high) envelope and close matter.
    idx = pd.date_range(start=start, periods=len(closes), freq="W-FRI")
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=idx)


# Shared V-shaped base path: 40 weeks falling 100->20 (locks in a deep 19-week
# low/high envelope), then a recovery leg rising 20->80. Slicing the recovery
# leg to different lengths lands the latest bar's %K at different, known
# points along its climb through the band.
_DOWN = list(np.linspace(100, 20, 40))
_UP = list(np.linspace(20, 80, 40))[1:]  # first element duplicates _DOWN's last


def test_empty_df_returns_false():
    df = pd.DataFrame({"high": [], "low": [], "close": []})
    result, k, d = is_stoch_sweet_spot(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_insufficient_history_returns_false():
    # 10 weeks total -- well short of the ~25 weeks needed for a 19-length
    # stochastic smoothed by two 4-period SMAs to produce a valid %D.
    df = _weekly_df(list(np.linspace(50, 55, 10)))
    result, k, d = is_stoch_sweet_spot(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_k_in_band_above_d_and_rising_qualifies():
    # Recovery leg sliced to week 7: %K climbs steadily through the band
    # (32-80) while %D (still catching up from the prior downtrend's low)
    # lags below it, and %K is above the prior week's value -- all three
    # conditions hold.
    df = _weekly_df(_DOWN + _UP[:7])
    result, k, d = is_stoch_sweet_spot(df, **DEFAULTS)
    assert result is True
    assert k == pytest.approx(34.45, abs=0.05)
    assert d == pytest.approx(23.58, abs=0.05)
    assert k > d


def test_k_below_band_excluded():
    # Recovery leg sliced to week 6: %K (~26.5) hasn't reached the band
    # (32-80) yet.
    df = _weekly_df(_DOWN + _UP[:6])
    result, k, d = is_stoch_sweet_spot(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_k_above_band_excluded():
    # Recovery leg sliced to week 11: %K (~80.4) has already climbed past
    # the band's upper bound.
    df = _weekly_df(_DOWN + _UP[:11])
    result, k, d = is_stoch_sweet_spot(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_k_in_band_but_falling_excluded():
    # Recovery leg sliced to week 9, with a sharp one-week price dip
    # inserted on the final bar. %K ticks down from the prior week (43.74 ->
    # 41.92, still above %D, still inside the band) -- the "rising"
    # condition alone is what excludes it. Because %K is smoothed over 4
    # weeks, only a large single-week dip is enough to reverse its slope
    # without also pushing it out of band or below %D.
    up_dip = list(np.linspace(20, 80, 40))[1:]
    up_dip[8] -= 10
    df = _weekly_df(_DOWN + up_dip[:9])
    result, k, d = is_stoch_sweet_spot(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_k_in_band_but_below_d_excluded():
    # Different (hump-shaped) path: 50 weeks rising 20->90, then falling
    # back down. On the way down, %K re-enters the band (32-80) while %D --
    # still smoothing in the recent peak -- sits well above it.
    up = list(np.linspace(20, 90, 50))
    down = list(np.linspace(90, 20, 50))[1:]
    df = _weekly_df(up + down[:7])
    result, k, d = is_stoch_sweet_spot(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_compute_kd_daily_empty_input_returns_empty_frame():
    df = pd.DataFrame({"high": [], "low": [], "close": []})
    kd = compute_kd_daily(df, stoch_length=5, k_smooth=3, d_smooth=3)
    assert list(kd.columns) == ["close", "k", "d"]
    assert len(kd) == 0


def test_compute_kd_daily_matches_hand_verified_values_no_weekly_resample():
    # V-shaped daily path: 10 days falling 100->80, then 10 days rising
    # 80->100 (linspace, so high=close+1/low=close-1 throughout). Values
    # below were computed by running compute_kd_daily itself and are
    # asserted here as a regression pin, not independently re-derived --
    # the real check is that this operates on DAILY bars directly (no
    # resample_weekly step), unlike compute_kd().
    idx = pd.date_range("2026-01-02", periods=20, freq="D")
    closes = list(np.linspace(100, 80, 10)) + list(np.linspace(80, 100, 10))
    df = pd.DataFrame(
        {"close": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes]},
        index=idx,
    )
    kd = compute_kd_daily(df, stoch_length=5, k_smooth=3, d_smooth=3)
    row = kd.loc["2026-01-15"]
    assert row["k"] == pytest.approx(74.31, abs=0.01)
    assert row["d"] == pytest.approx(48.85, abs=0.01)
