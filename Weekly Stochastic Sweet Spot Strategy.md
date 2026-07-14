---
source: https://youtu.be/Tr_RXi6wQko
tags: [strategy, stochastic, long-term-trading, trend-following]
---

# Weekly Stochastic "Sweet Spot" Strategy

Long-term/swing trend-following approach. Core idea: only two indicators matter — **trend** (weekly stochastic) and **volume**. Ignore day-to-day price noise and news; hold trades from weeks to many months based purely on the trend indicator.

## Indicator Setup

| Indicator | Timeframe | Settings | Purpose |
|---|---|---|---|
| Stochastic Oscillator | **Weekly** (plotted on daily chart via the indicator's own timeframe input) | Length 19, %K smoothing 4, %D smoothing 4 | **Trend** — the only signal used to decide entries/exits |
| Stochastic Oscillator | Daily | Length 10, %K smoothing 3, %D smoothing 3 | Shown only for contrast — daily stoch is noisy/choppy vs. the smooth weekly one. Not used for trade decisions. |
| Volume | Daily | Default | Confirms trend strength ("gasoline" pushing the move); used at entry, not monitored while holding |

Two lines on the stochastic:
- **Red line** = %K (faster)
- **Yellow line** = %D (slower/signal)

## The "Sweet Spot" Zone

A horizontal band on the weekly stochastic (0–100 scale) between:
- **Upper line: 80**
- **Lower line: 32**

This zone is where the trend has historically had the **highest probability** of producing a sustained, profitable move — not a guarantee, just better odds.

## Entry Rules — Long

1. Wait for the red and yellow lines to move up **into** the sweet spot zone (32–80) from below.
2. Enter long when the **red line crosses above the yellow line** inside/entering that zone.
3. Hold the position. Crossing **above 80 does not mean exit** — staying above 80 with red still on top of yellow is fine; ride it.

## Exit Rules — Long

- Exit **only** when the **red line crosses below the yellow line while under 80**. That crossover — not the price action, not a red day, not news — is the signal.
- Ignore interim pullbacks entirely as long as red stays above yellow (examples below).

## Entry / Exit Rules — Short (mirror image)

- Enter short when the red line crosses **below** the yellow line near/just below 80 (trend rolling over at the top of the sweet spot).
- Stay short as long as red remains below yellow.
- Cover when red crosses back above yellow.

## Real Examples Cited

- **Costco (2021–2022):** Entered ~$350 when weekly stoch turned up through the sweet spot (Apr 2021). Stayed in through an overbought dip (red stayed above yellow the whole time) until the exit signal in early Jan 2022 (~$511) — a ~9-month hold.
- **Google (2020–2022):** Entered ~$79 (Oct 2020) on the same sweet-spot turn-up. Held through multiple sharp pullbacks (e.g. $121→$109, $146→$130) because red never crossed below yellow, exiting around ~$144 (roughly doubled).
- **Nvidia (counter-example):** Shown as a "sweet spot reversal" — price/stochastic entered the sweet spot, started up, then reversed and rolled back down. Not every entry works.

## Probability / Mindset

- Expect roughly **30% of entries to fail** (a sweet-spot reversal that turns back down before trending).
- Over a large sample (~100 trades), a **~70% win rate** with variable-sized wins (some +4%, some +40%) is the expected edge. Profitability comes from repetition and patience, not any single trade.
- The hardest part isn't the rule — it's the discipline to trust the trend through daily/weekly price noise and not exit early out of fear, and not enter too early before the cross confirms.

## Volume's Role

- Volume = confirmation that money is flowing in the trend's direction (analogy: more people pushing the same boulder / rowing the same direction = bigger, more reliable moves).
- Used qualitatively at entry ("wait until you have enough volume") — no specific numeric threshold given in the video. Not tracked once in the trade; the weekly stochastic cross is what triggers the exit.

## Summary Checklist

- [ ] Add Stochastic to chart, set its timeframe input to **Weekly** (19, 4, 4) — this is the trend signal.
- [ ] Identify the 32–80 "sweet spot" band on that stochastic panel.
- [ ] **Long:** buy when red crosses above yellow while entering/inside the sweet spot; hold regardless of price noise; exit only when red crosses below yellow under 80.
- [ ] **Short:** mirror the above — sell when red crosses below yellow near the top of the sweet spot; cover when red crosses back above yellow.
- [ ] Confirm entries with rising volume in the trade direction.
- [ ] Expect ~30% of entries to fail; size and diversify across trades accordingly — the edge is statistical over many trades, not per-trade.
- [ ] This is a multi-week-to-multi-month holding strategy, not a day-trading system.
