#!/bin/zsh
# Double-click this file in Finder to run the SL/TP + long-term trend filter
# sweep on top of the Stochastic Weekly Signal strategy (SL 4-10%, TP
# 1.5-3.5R, with/without a weekly EMA50>EMA200 regime filter -- 70
# combinations). Ticker universe: newest file in INPUT/ if present, else the
# Darvas Raw screener CSV.
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 backtest_signal_sltp_sweep.py

echo ""
echo "Press any key to close this window..."
read -k 1
