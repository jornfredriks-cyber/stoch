#!/bin/zsh
# Double-click this file in Finder to run the Basic Scanner trailing-stop
# backtest. Entry signal: same crossover as the Signal backtest (%K crosses
# above 32, %K>%D) -- on the signal, looks backward for the LAST
# SIGNIFICANT HIGH (a weekly swing-high pivot) and places a BUY-STOP at
# that price, active for breakout_window_days (10) trading days; fills the
# moment a day's high reaches it, otherwise the signal is skipped ("No
# trade!" -- never cleared resistance). Exit: the ONLY exit is a 20%
# trailing stop off the highest weekly close since (actual) entry, checked
# against daily intraweek lows -- no signal exit, no take-profit, no time
# limit. Ticker source: newest file in INPUT/, else the newest Basic
# Scanner output in OUTPUT/BasicScan/.
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 backtest_basic_trail.py

echo ""
echo "Press any key to close this window..."
read -k 1
