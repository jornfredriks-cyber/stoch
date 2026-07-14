import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import scan_utils
from scan_utils import fetch_ohlc_bulk


def test_fetch_ohlc_bulk_uses_period_override(monkeypatch):
    captured = {}

    def fake_download(tickers, period, **kwargs):
        captured["period"] = period
        return pd.DataFrame()

    monkeypatch.setattr(scan_utils.yf, "download", fake_download)
    fetch_ohlc_bulk(["AAPL"], period="max")
    assert captured["period"] == "max"
