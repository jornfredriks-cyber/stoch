import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

import stoch_signal_scan as mod
from stoch_signal_scan import is_fresh_stoch_signal_buy, is_not_stretched


def _kd_frame(rows):
    """(date, close, k, d) rows -> a DataFrame shaped like compute_kd()'s output."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"close": [r[1] for r in rows], "k": [r[2] for r in rows], "d": [r[3] for r in rows]},
        index=idx,
    )


def test_fresh_crossover_detected(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 102, 25, 22),
        ("2026-01-16", 105, 35, 28),  # fresh: prev_k=25<32, k=35>32, k>d
    ])
    monkeypatch.setattr("stoch_signal_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price = is_fresh_stoch_signal_buy(pd.DataFrame())
    assert fresh is True
    assert k == 35
    assert d == 28
    assert entry_date == pd.Timestamp("2026-01-16")
    assert price == 105


def test_not_fresh_when_crossover_happened_prior_week(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # crossover happened here, not on latest bar
        ("2026-01-16", 108, 40, 30),  # already above 32 last week too -> not fresh
    ])
    monkeypatch.setattr("stoch_signal_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price = is_fresh_stoch_signal_buy(pd.DataFrame())
    assert fresh is False
    assert k is None and d is None and entry_date is None and price is None


def test_no_signal_when_k_below_d(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 40),  # prev_k<32, k>32, but k<d -> not fresh
    ])
    monkeypatch.setattr("stoch_signal_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price = is_fresh_stoch_signal_buy(pd.DataFrame())
    assert fresh is False


def test_boundary_exact_level_not_fresh(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 100, 20, 15),
        ("2026-01-09", 102, 32, 28),  # k==32 exactly -> no fresh signal (k>32 is false)
    ])
    monkeypatch.setattr("stoch_signal_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, *_ = is_fresh_stoch_signal_buy(pd.DataFrame())
    assert fresh is False


def test_insufficient_history_returns_false(monkeypatch):
    kd = _kd_frame([("2026-01-09", 105, 35, 28)])
    monkeypatch.setattr("stoch_signal_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price = is_fresh_stoch_signal_buy(pd.DataFrame())
    assert fresh is False
    assert k is None and d is None and entry_date is None and price is None


def test_nan_rows_are_skipped_before_checking_latest(monkeypatch):
    kd = _kd_frame([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 103, np.nan, np.nan),  # dropped by dropna(subset=["k","d"])
        ("2026-01-16", 105, 35, 28),          # becomes the "latest" bar after dropna
    ])
    monkeypatch.setattr("stoch_signal_scan.compute_kd", lambda df, *a, **k: kd)
    fresh, k, d, entry_date, price = is_fresh_stoch_signal_buy(pd.DataFrame())
    assert fresh is True
    assert entry_date == pd.Timestamp("2026-01-16")


# ── is_not_stretched() ─────────────────────────────────────────────────────

def _flat_daily_df(n_bars=100, price=20.0):
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    return pd.DataFrame({"close": pd.Series(price, index=idx)})


def _x_for_target_pct(price: float, target_pct: float, span: int) -> float:
    """Same closed-form solver as the retired ema50_screener.py tests: solves
    for a final close X such that, appended after a long flat-`price`
    history (whose EMA(span) has converged exactly to `price`), the
    resulting EMA(span) sits `target_pct` below/above X. Used only to
    construct input data -- each test re-derives the actual distance from
    mod._ema()'s real output."""
    alpha = 2 / (span + 1)
    k = 1 + target_pct / 100
    return (1 - alpha) * k * price / (1 - alpha * k)


def _daily_df_with_final_close_pct(n_bars, price, target_pct, ema_length):
    df = _flat_daily_df(n_bars, price)
    x = _x_for_target_pct(price, target_pct, ema_length)
    df.loc[df.index[-1], "close"] = x
    return df


def test_not_stretched_within_band_passes():
    df = _daily_df_with_final_close_pct(100, 20.0, target_pct=2.5, ema_length=mod.EMA_LENGTH)
    ok, dist_pct = is_not_stretched(df)
    assert ok is True
    assert 0.0 <= dist_pct <= 5.0


def test_stretched_above_band_rejected():
    df = _daily_df_with_final_close_pct(100, 20.0, target_pct=8.0, ema_length=mod.EMA_LENGTH)
    ok, dist_pct = is_not_stretched(df)
    assert ok is False
    assert dist_pct > 5.0


def test_below_ema_rejected():
    df = _daily_df_with_final_close_pct(100, 20.0, target_pct=-2.0, ema_length=mod.EMA_LENGTH)
    ok, dist_pct = is_not_stretched(df)
    assert ok is False
    assert dist_pct < 0.0


def test_insufficient_daily_history_returns_none():
    df = _flat_daily_df(n_bars=30, price=20.0)
    ok, dist_pct = is_not_stretched(df)
    assert ok is False
    assert dist_pct is None


def test_trailing_nan_close_row_is_ignored():
    df = _daily_df_with_final_close_pct(100, 20.0, target_pct=2.5, ema_length=mod.EMA_LENGTH)
    incomplete = pd.DataFrame({"close": [float("nan")]}, index=[df.index[-1] + pd.Timedelta(days=1)])
    df_with_trailing_nan = pd.concat([df, incomplete])
    ok, dist_pct = is_not_stretched(df_with_trailing_nan)
    assert ok is True
    assert 0.0 <= dist_pct <= 5.0
