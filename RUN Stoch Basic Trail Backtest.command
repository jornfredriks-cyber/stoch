#!/bin/zsh
# Double-click this file in Finder to run the Basic Scanner trailing-stop
# backtest. Entry signal: same crossover as the Signal backtest (%K crosses
# above 32, %K>%D) -- by default confirmed 3 trading days later, checking
# the DAILY Stochastic (independent 14/3/3 settings): if that day's daily
# %K is above daily %D, confirmed, and entry is a market buy at the NEXT
# trading day's open; otherwise the signal is skipped. Exit: the ONLY exit
# is a 20% trailing stop off the highest weekly close since (actual) entry,
# checked against daily intraweek lows -- no signal exit, no take-profit,
# no time limit. Ticker source: newest file in INPUT/, else the newest
# Basic Scanner output in OUTPUT/BasicScan/.
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 backtest_basic_trail.py

echo ""
echo "Press any key to close this window..."
read -k 1
