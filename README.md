# Pairs Trading Mean Reversion Strategy: KO vs PEP

A statistical arbitrage strategy that exploits the mean-reverting relationship between Coca-Cola (KO) and PepsiCo (PEP) stock prices using a z-score based entry/exit system.

## How It Works

1. **Data** — Downloads 2 years of daily adjusted close prices for KO and PEP via `yfinance`
2. **Hedge ratio** — Estimates beta using OLS regression: `KO ~ α + β·PEP`
3. **Spread** — Constructs the spread: `spread = KO − β·PEP`
4. **Z-score** — Computes a 30-day rolling z-score of the spread
5. **Signals** — Generates long/short entries based on z-score thresholds
6. **P&L** — Simulates daily returns with no look-ahead bias
7. **Chart** — Saves a 4-panel visualisation to `pairs_trading.png`

### Trading Rules

| Condition | Action |
|---|---|
| z < −2 | Enter **long** spread (long KO, short PEP) |
| z > +2 | Enter **short** spread (short KO, long PEP) |
| \|z\| < 0.5 | **Exit** — mean reversion achieved |
| \|z\| > 3.5 | **Stop-loss** — exit to limit losses |

## Requirements

- Python 3.8+
- `yfinance`
- `pandas`
- `numpy`
- `statsmodels`
- `matplotlib`

Install all dependencies:

```bash
pip install yfinance pandas numpy statsmodels matplotlib
```

## Usage

```bash
python3 main.py
```

The script will:
- Print the OLS fit parameters (alpha, beta, R²) to the terminal
- Print a strategy summary with annualised return, Sharpe ratio, and max drawdown
- Save the chart as `pairs_trading.png` in the project directory

To view the chart after running:

```bash
open pairs_trading.png        # macOS
xdg-open pairs_trading.png   # Linux
start pairs_trading.png       # Windows
```

## Output

**Terminal output example:**
```
Downloading price data …
OLS fit  →  alpha = 78.4804,  beta = -0.0833
R²       →  0.0358
Chart saved → pairs_trading.png

── Strategy Summary ───────────────────────────────
  Period          : 2024-04-29 → 2026-03-17
  Hedge ratio β   : -0.0833
  Total days      : 471
  Days in market  : 161  (34.2%)
  Approx trades   : 13
  Final return    : -8.3%
  Ann. return     : -4.6%
  Ann. volatility : 11.5%
  Sharpe ratio    : -0.40
  Max drawdown    : -17.4%
───────────────────────────────────────────────────
```

**Chart (`pairs_trading.png`):**

| Panel | Description |
|---|---|
| (a) | Normalised KO and PEP prices |
| (b) | Spread over time with mean line |
| (c) | Rolling z-score with entry/exit/stop-loss threshold lines and shaded position regions |
| (d) | Cumulative strategy returns |

## Project Structure

```
.
├── main.py              # Full strategy implementation
├── pairs_trading.png    # Generated chart (after running)
└── README.md
```

## Notes

- The strategy performs best when KO and PEP are cointegrated (high R²). A low R² signals the pair has decoupled and the mean-reversion assumption may not hold.
- `matplotlib.use("Agg")` is set so the script saves the chart to file and exits cleanly without requiring a display. Remove that line if you want an interactive pop-up window instead.
- All returns are normalised by the initial spread level to express P&L in percentage terms.
