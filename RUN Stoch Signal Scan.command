#!/bin/zsh
# Double-click this file in Finder to scan the Darvas Raw screener universe
# for fresh "Stochastic Weekly Signal" buy candidates (weekly %K crossed
# strictly above 32 with %K>%D on the latest completed weekly bar).
# macOS may ask you to allow it once in System Settings → Privacy & Security.

SCRIPT_DIR="/Users/jamesblond/Documents/1-Projects/AI Trade/Stoch"

cd "$SCRIPT_DIR"
venv/bin/python3 stoch_signal_scan.py

echo ""
echo "Press any key to close this window..."
read -k 1
