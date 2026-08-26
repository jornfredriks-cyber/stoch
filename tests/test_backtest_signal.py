import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

import backtest_signal
from backtest_signal import load_backtest_signal_tickers, simulate_trades_signal


def _weekly(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"close": [r[1] for r in rows], "k": [r[2] for r in rows], "d": [r[3] for r in rows]},
        index=idx,
    )


def test_single_clean_trade():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # entry: K crosses strictly above 32, K>D
        ("2026-01-16", 110, 85, 70),  # holding
        ("2026-01-23", 108, 75, 90),  # exit: K crosses strictly below 80, K<D
    ])
    trades = simulate_trades_signal(weekly, entry_level=32.0, exit_level=80.0, trade_value_usd=1000.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "closed"
    assert t["entry_date"] == pd.Timestamp("2026-01-09")
    assert t["entry_price"] == 105
    assert t["entry_k"] == 35
    assert t["exit_date"] == pd.Timestamp("2026-01-23")
    assert t["exit_price"] == 108
    assert t["exit_k"] == 75
    assert t["exit_d"] == 90
    assert t["return_pct"] == pytest.approx((108 / 105 - 1) * 100, abs=0.01)
    assert t["dollar_pnl"] == pytest.approx(1000 * (108 / 105 - 1), abs=0.01)
    assert t["holding_weeks"] == 2


def test_sequential_trades():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 25),
        ("2026-01-09", 105, 35, 28),  # entry 1
        ("2026-01-16", 110, 85, 70),  # holding
        ("2026-01-23", 108, 75, 90),  # exit 1
        ("2026-01-30", 100, 20, 55),  # flat
        ("2026-02-06", 106, 40, 30),  # entry 2
        ("2026-02-13", 112, 90, 60),  # holding
        ("2026-02-20", 109, 78, 95),  # exit 2
    ])
    trades = simulate_trades_signal(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 2
    assert trades[0]["entry_date"] == pd.Timestamp("2026-01-09")
    assert trades[0]["exit_date"] == pd.Timestamp("2026-01-23")
    assert trades[1]["entry_date"] == pd.Timestamp("2026-02-06")
    assert trades[1]["exit_date"] == pd.Timestamp("2026-02-20")


def test_boundary_exact_level_does_not_trigger():
    # K lands exactly at 32 (entry) and exactly at 80 (exit) -- strict
    # inequality required, so touching the boundary exactly doesn't fire,
    # and (since the comparison always uses the immediately preceding
    # bar's K) it also doesn't retroactively count once K moves further
    # past the level without ever having been strictly on the near side.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 15),
        ("2026-01-09", 102, 32, 28),   # K==32 exactly -> no entry (prev_k=20<32 but k>32 is false)
        ("2026-01-16", 105, 30, 28),   # dips back strictly below 32
        ("2026-01-23", 108, 35, 28),   # now prev_k=30<32 and k=35>32 and k>d -> entry
        ("2026-01-30", 110, 90, 70),   # holding, K>80
        ("2026-02-06", 108, 80, 85),   # K==80 exactly -> no exit (prev_k=90>80 but k<80 is false)
        ("2026-02-13", 109, 85, 60),   # back strictly above 80
        ("2026-02-20", 106, 75, 90),   # now prev_k=85>80 and k=75<80 and k<d -> exit
    ])
    trades = simulate_trades_signal(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 1
    assert trades[0]["entry_date"] == pd.Timestamp("2026-01-23")
    assert trades[0]["exit_date"] == pd.Timestamp("2026-02-20")


def test_quirk_stays_open_if_crossunder_not_paired_with_k_below_d():
    # K crosses below 80 while K is still >= D (no exit that bar, since
    # the crossunder event itself doesn't satisfy K<D). K later drops
    # below D, but without a *fresh* crossunder-of-80 event on that same
    # bar (prev_k is no longer >80) -- under the literal Pine rule, this
    # trade never gets an exit signal and stays open.
    weekly = _weekly([
        ("2026-01-02", 100, 20, 15),
        ("2026-01-09", 105, 35, 28),  # entry
        ("2026-01-16", 112, 90, 60),  # holding, K>80
        ("2026-01-23", 108, 75, 70),  # K crosses under 80 (90->75), but K<D is false (75>70) -> no exit
        ("2026-01-30", 106, 70, 75),  # K<D now true (70<75), but prev_k=75 is not >80 -> no exit (quirk)
        ("2026-02-06", 104, 65, 60),  # still no crossunder-of-80 event -> still open
    ])
    trades = simulate_trades_signal(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 1
    assert trades[0]["status"] == "open"
    assert trades[0]["exit_date"] is None


def test_open_trade_when_never_exits():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 15),
        ("2026-01-09", 105, 35, 28),  # entry
        ("2026-01-16", 110, 50, 40),  # holding, never reaches 80
        ("2026-01-23", 115, 60, 55),  # holding, data ends
    ])
    trades = simulate_trades_signal(weekly, entry_level=32.0, exit_level=80.0, trade_value_usd=1000.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "open"
    assert t["exit_date"] is None
    assert t["return_pct"] == pytest.approx((115 / 105 - 1) * 100, abs=0.01)
    assert t["dollar_pnl"] == pytest.approx(1000 * (115 / 105 - 1), abs=0.01)
    assert t["holding_weeks"] == 2


def test_never_enters_returns_empty_list():
    weekly = _weekly([
        ("2026-01-02", 100, 10, 8),
        ("2026-01-09", 102, 15, 12),
        ("2026-01-16", 101, 18, 16),
    ])
    trades = simulate_trades_signal(weekly, entry_level=32.0, exit_level=80.0)
    assert trades == []


def test_load_backtest_signal_tickers_uses_newest_input_file(tmp_path, monkeypatch):
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    (input_dir / ".gitkeep").write_text("")
    (input_dir / "EMA50_StochSignal_2026-07-25.txt").write_text("EXE\nLITE\nNVDA\n")

    monkeypatch.setattr(
        backtest_signal, "load_darvas_raw_tickers",
        lambda: (_ for _ in ()).throw(AssertionError("should not fall back to Darvas")),
    )

    tickers, source = load_backtest_signal_tickers(str(tmp_path))
    assert tickers == ["EXE", "LITE", "NVDA"]
    assert source == "INPUT/EMA50_StochSignal_2026-07-25.txt"


def test_load_backtest_signal_tickers_ignores_hidden_files_and_picks_newest(tmp_path, monkeypatch):
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    older = input_dir / "old_list.txt"
    newer = input_dir / "new_list.txt"
    older.write_text("OLD\n")
    newer.write_text("NEW\n")
    os.utime(older, (1_000_000_000, 1_000_000_000))
    os.utime(newer, (2_000_000_000, 2_000_000_000))

    monkeypatch.setattr(
        backtest_signal, "load_darvas_raw_tickers",
        lambda: (_ for _ in ()).throw(AssertionError("should not fall back to Darvas")),
    )

    tickers, source = load_backtest_signal_tickers(str(tmp_path))
    assert tickers == ["NEW"]
    assert source == "INPUT/new_list.txt"


def test_load_backtest_signal_tickers_falls_back_to_darvas_when_input_empty(tmp_path, monkeypatch):
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    (input_dir / ".gitkeep").write_text("")

    monkeypatch.setattr(backtest_signal, "load_darvas_raw_tickers", lambda: ["AAPL", "MSFT"])

    tickers, source = load_backtest_signal_tickers(str(tmp_path))
    assert tickers == ["AAPL", "MSFT"]
    assert source == "Darvas Raw screener"


def test_load_backtest_signal_tickers_parses_comma_separated_watchlist_format(tmp_path, monkeypatch):
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    (input_dir / "Tradingview Watchlist_2026-07-25_999.txt").write_text("NASDAQ:AAPL,NYSE:V")

    monkeypatch.setattr(
        backtest_signal, "load_darvas_raw_tickers",
        lambda: (_ for _ in ()).throw(AssertionError("should not fall back to Darvas")),
    )

    tickers, source = load_backtest_signal_tickers(str(tmp_path))
    assert tickers == ["AAPL", "V"]


def test_skips_nan_k_or_d_rows():
    weekly = _weekly([
        ("2026-01-02", 100, 20, 15),
        ("2026-01-09", 105, 35, 28),  # entry
        ("2026-01-16", 110, 85, 70),  # holding
        ("2026-01-23", 112, float("nan"), float("nan")),  # NaN -- must be skipped
        ("2026-01-30", 108, 75, 90),  # exit: K crosses below 80, K<D
    ])
    trades = simulate_trades_signal(weekly, entry_level=32.0, exit_level=80.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_date"] == pd.Timestamp("2026-01-09")
    assert t["exit_date"] == pd.Timestamp("2026-01-30")
    assert t["holding_weeks"] == 2
