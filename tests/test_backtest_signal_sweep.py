import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from backtest_signal_sweep import compute_kd_fast
from stoch_scan import compute_kd, resample_weekly


def _synthetic_daily(n=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1.5, n))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


@pytest.mark.parametrize("length,k_smooth,d_smooth", [
    (19, 4, 4),
    (10, 2, 2),
    (28, 6, 7),
])
def test_compute_kd_fast_matches_canonical_compute_kd(length, k_smooth, d_smooth):
    df = _synthetic_daily()
    weekly = resample_weekly(df)

    fast_result = compute_kd_fast(weekly, length, k_smooth, d_smooth)
    canonical_result = compute_kd(df, stoch_length=length, k_smooth=k_smooth, d_smooth=d_smooth)

    pd.testing.assert_frame_equal(
        fast_result.reset_index(drop=True),
        canonical_result.reset_index(drop=True),
    )
