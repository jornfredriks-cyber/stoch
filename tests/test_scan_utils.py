import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import scan_utils
from scan_utils import fetch_ohlc_bulk, _parse_watchlist_line, find_ticker_list


def test_fetch_ohlc_bulk_uses_period_override(monkeypatch):
    captured = {}

    def fake_download(tickers, period, **kwargs):
        captured["period"] = period
        return pd.DataFrame()

    monkeypatch.setattr(scan_utils.yf, "download", fake_download)
    fetch_ohlc_bulk(["AAPL"], period="max")
    assert captured["period"] == "max"


def test_parse_watchlist_line_extracts_symbols():
    assert _parse_watchlist_line("NASDAQ:AAPL,NYSE:V") == ["AAPL", "V"]


def test_parse_watchlist_line_converts_oslo_suffix():
    assert _parse_watchlist_line("OSLO:EQNR") == ["EQNR.OL"]


def test_parse_watchlist_line_handles_plain_ticker():
    assert _parse_watchlist_line("AAPL") == ["AAPL"]


def test_parse_watchlist_line_skips_empty_entries():
    assert _parse_watchlist_line("AAPL,,  ,V") == ["AAPL", "V"]


def test_find_ticker_list_prefers_watchlist_file(tmp_path):
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    (input_dir / "My Watchlist.txt").write_text("NASDAQ:AAPL,NYSE:V")
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()

    result = find_ticker_list(str(input_dir), str(fallback_dir))
    assert result == ["AAPL", "V"]


def test_find_ticker_list_matches_tradingview_default_export_name(tmp_path):
    # TradingView's actual default export filename doesn't contain the word
    # "Watchlist" at all -- it's "Tradingview <ListName>_<date>_<id>.txt".
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    (input_dir / "Tradingview Stoch_SweetSpot_2026-07-14_19337.txt").write_text(
        "NASDAQ:AAPL,NYSE:V"
    )
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()

    result = find_ticker_list(str(input_dir), str(fallback_dir))
    assert result == ["AAPL", "V"]


def test_find_ticker_list_falls_back_to_screener_csv(tmp_path):
    input_dir = tmp_path / "INPUT"
    input_dir.mkdir()
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    csv_path = fallback_dir / "Stoch Raw Screener_2026-07-14.csv"
    pd.DataFrame({"Symbol": ["AAPL", "MSFT"]}).to_csv(csv_path, index=False)

    result = find_ticker_list(str(input_dir), str(fallback_dir))
    assert result == ["AAPL", "MSFT"]
