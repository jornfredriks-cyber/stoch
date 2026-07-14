#!/bin/zsh
# Double-click this file in Finder to run the Stoch Sweet Spot scan.
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 screener.py && venv/bin/python3 stoch_scan.py

echo ""
echo "Press any key to close this window..."
read -k 1
