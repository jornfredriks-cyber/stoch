#!/bin/zsh
# Double-click this file in Finder to run the Stoch Basic Scanner.
# Loose full-market pre-filter (price/volume/market-cap/ADR only, no
# trend/ATH-proximity requirement) followed by a Weekly Stochastic
# "starting to turn" check: %K above %D but still below 32.
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 stoch_basic_screener.py && venv/bin/python3 stoch_basic_scan.py

echo ""
echo "Press any key to close this window..."
read -k 1
