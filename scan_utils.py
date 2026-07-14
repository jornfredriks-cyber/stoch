import glob
import os
import time

import pandas as pd
import yfinance as yf

# Point libcurl at certifi's CA bundle so Homebrew OpenSSL mismatches don't fail
try:
    import certifi
    for _k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(_k, certifi.where())
except ImportError:
    pass

HISTORY_PERIOD    = "1y"
FETCH_CHUNK_SIZE  = 50
FETCH_RETRIES     = 2
FETCH_RETRY_DELAY = 2.0


class _Tee:
    """Writes to both the terminal and a log file simultaneously."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def find_latest_csv(folder: str) -> str:
    files = glob.glob(os.path.join(folder, "*[Ss]creener*.csv"))
    if not files:
        raise FileNotFoundError(f"No screener CSV found in {folder}")
    return max(files, key=os.path.getmtime)


NORWEGIAN_EXCHANGES = {"OSLO"}


def _parse_watchlist_line(line: str) -> list[str]:
    tickers = []
    for raw in line.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            exchange, symbol = raw.split(":", 1)
            exchange = exchange.strip().upper()
            symbol = symbol.strip()
            if exchange in NORWEGIAN_EXCHANGES:
                tickers.append(f"{symbol}.OL")
            else:
                tickers.append(symbol)
        else:
            tickers.append(raw)
    return tickers


def find_ticker_list(input_folder: str, fallback_folder: str) -> list[str]:
    """
    Looks for a *Watchlist* file (TradingView export: plain text,
    comma-separated EXCHANGE:TICKER) in input_folder. Falls back to the
    latest screener CSV in fallback_folder if no watchlist file exists.
    """
    watchlist_files = glob.glob(os.path.join(input_folder, "*Watchlist*"))
    if watchlist_files:
        watchlist_path = max(watchlist_files, key=os.path.getmtime)
        with open(watchlist_path) as f:
            content = f.read()
        tickers = []
        for line in content.splitlines():
            tickers.extend(_parse_watchlist_line(line))
        return tickers

    csv_path = find_latest_csv(fallback_folder)
    df = pd.read_csv(csv_path)
    return df["Symbol"].dropna().tolist()


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _clean_ohlc_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    if df.empty:
        return None
    df.columns = df.columns.str.lower()
    required = {"high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return None
    return df


def _extract_symbol_frame(downloaded: pd.DataFrame, symbol: str, chunk_len: int) -> pd.DataFrame | None:
    if downloaded is None or downloaded.empty:
        return None

    if isinstance(downloaded.columns, pd.MultiIndex):
        if symbol in downloaded.columns.get_level_values(0):
            return _clean_ohlc_frame(downloaded[symbol])
        if symbol in downloaded.columns.get_level_values(1):
            return _clean_ohlc_frame(downloaded.xs(symbol, level=1, axis=1))
        return None

    if chunk_len == 1:
        return _clean_ohlc_frame(downloaded)
    return None


def fetch_ohlc_bulk(
    yf_symbols: list[str],
    chunk_size: int = FETCH_CHUNK_SIZE,
    retries: int = FETCH_RETRIES,
    retry_delay: float = FETCH_RETRY_DELAY,
    period: str = HISTORY_PERIOD,
) -> dict[str, pd.DataFrame]:
    result = {}
    unique_symbols = list(dict.fromkeys(yf_symbols))

    for chunk_index, chunk in enumerate(_chunks(unique_symbols, chunk_size), 1):
        downloaded = None
        for attempt in range(retries + 1):
            try:
                downloaded = yf.download(
                    " ".join(chunk),
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                break
            except Exception as exc:
                if attempt >= retries:
                    print(f"  chunk {chunk_index}: yfinance ERROR after {retries + 1} attempts: {exc}")
                else:
                    time.sleep(retry_delay)

        if downloaded is None:
            continue

        for yf_symbol in chunk:
            df = _extract_symbol_frame(downloaded, yf_symbol, len(chunk))
            if df is not None:
                result[yf_symbol] = df

        if chunk_index * chunk_size < len(unique_symbols):
            time.sleep(0.5)

    return result


def fetch_ohlc_single(yf_symbol: str) -> pd.DataFrame | None:
    """Fetch OHLC for one symbol via Ticker.history — used for post-bulk retries."""
    try:
        df = yf.Ticker(yf_symbol).history(period=HISTORY_PERIOD, interval="1d", auto_adjust=True)
        return _clean_ohlc_frame(df)
    except Exception:
        return None
