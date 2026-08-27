import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from stoch_basic_scan import is_stoch_basic_candidate

DEFAULTS = dict(stoch_length=19, k_smooth=4, d_smooth=4, k_ceiling=32.0)


def _weekly_df(closes, start="2026-01-02"):
    # start="2026-01-02" is a Friday, so periods land one-per-week on
    # Friday-anchored bins with no partial-week ambiguity.
    idx = pd.date_range(start=start, periods=len(closes), freq="W-FRI")
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=idx)


# Same V-shaped base path as test_stoch_scan.py: 40 weeks falling 100->20,
# then a recovery leg rising 20->80.
_DOWN = list(np.linspace(100, 20, 40))
_UP = list(np.linspace(20, 80, 40))[1:]


def test_empty_df_returns_false():
    df = pd.DataFrame({"high": [], "low": [], "close": []})
    result, k, d = is_stoch_basic_candidate(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_insufficient_history_returns_false():
    # 10 weeks total -- well short of the ~25 weeks needed for a 19-length
    # stochastic smoothed by two 4-period SMAs to produce a valid %D.
    df = _weekly_df(list(np.linspace(50, 55, 10)))
    result, k, d = is_stoch_basic_candidate(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_k_above_d_and_below_ceiling_qualifies():
    # Recovery leg sliced to week 5: %K (19.67) has started climbing off
    # the deep low but hasn't reached the 32 ceiling yet, while %D (11.63,
    # still catching up from the prior downtrend) lags below it -- exactly
    # the "starting to turn, worth watching" window this scanner targets.
    df = _weekly_df(_DOWN + _UP[:5])
    result, k, d = is_stoch_basic_candidate(df, **DEFAULTS)
    assert result is True
    assert k == pytest.approx(19.67, abs=0.05)
    assert d == pytest.approx(11.63, abs=0.05)
    assert k > d


def test_k_at_or_above_ceiling_excluded():
    # Recovery leg sliced to week 7: the exact point test_stoch_scan.py
    # uses to show %K (34.45) already inside the 32-80 sweet-spot band --
    # this scanner's job ends where that one's begins, so it's excluded
    # here even though %K is still above %D.
    df = _weekly_df(_DOWN + _UP[:7])
    result, k, d = is_stoch_basic_candidate(df, **DEFAULTS)
    assert result is False and k is None and d is None


def test_k_below_d_excluded():
    # Hump-shaped path: 50 weeks rising 20->90, then falling. Sliced to
    # week 10 of the decline: %K (16.51) has already dropped back under
    # the 32 ceiling, but %D (35.46, still smoothing in the recent peak)
    # sits well above it -- still falling, not turning yet.
    up = list(np.linspace(20, 90, 50))
    down = list(np.linspace(90, 20, 50))[1:]
    df = _weekly_df(up + down[:10])
    result, k, d = is_stoch_basic_candidate(df, **DEFAULTS)
    assert result is False and k is None and d is None
