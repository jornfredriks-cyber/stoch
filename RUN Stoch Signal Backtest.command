#!/bin/zsh
# Double-click this file in Finder to run the Stochastic Weekly Signal backtester
# (fixed $1000/trade, no stop-loss/take-profit). Ticker universe: the newest
# file in INPUT/ if one exists (any name, no formatting required -- e.g. drop
# in a hand-picked ticker list or a scan's candidate output), else falls back
# to the Darvas Raw screener CSV.
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 backtest_signal.py

echo ""
echo "Press any key to close this window..."
read -k 1
