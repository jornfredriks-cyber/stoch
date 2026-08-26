#!/bin/zsh
# Double-click this file in Finder to run the Stochastic parameter sweep
# (finds the best stoch_length/k_smooth/d_smooth by total $ P&L, Darvas Raw
# screener universe, entry/exit thresholds fixed at 32/80).
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 backtest_signal_sweep.py

echo ""
echo "Press any key to close this window..."
read -k 1
