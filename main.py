"""
Pairs Trading Mean Reversion Strategy: KO vs PEP

Steps:
  1. Download 2 years of adjusted close prices via yfinance
  2. Estimate hedge ratio (beta) via OLS regression
  3. Compute spread and rolling z-score
  4. Generate entry/exit/stop-loss signals
  5. Simulate P&L day by day
  6. Visualize with 4 subplots
"""

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — renders to file, never blocks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── 1. DATA ──────────────────────────────────────────────────────────────────

END   = datetime.today()
START = END - timedelta(days=2 * 365)

print("Downloading price data …")
# threads=False avoids a yfinance SQLite timezone-cache lock issue on some systems
raw = yf.download(["KO", "PEP"], start=START, end=END, auto_adjust=True,
                  progress=False, threads=False)
prices = raw["Close"][["KO", "PEP"]].dropna()

if prices.empty:
    raise RuntimeError("Price download returned no data. Check your internet connection.")

ko  = prices["KO"]
pep = prices["PEP"]

# ── 2. HEDGE RATIO (OLS: KO ~ beta * PEP + alpha) ────────────────────────────

# Add a constant so OLS estimates both intercept and slope
X = sm.add_constant(pep)
model = sm.OLS(ko, X).fit()

alpha = model.params["const"]
beta  = model.params["PEP"]

print(f"OLS fit  →  alpha = {alpha:.4f},  beta = {beta:.4f}")
print(f"R²       →  {model.rsquared:.4f}")

# ── 3. SPREAD & ROLLING Z-SCORE ──────────────────────────────────────────────

WINDOW = 30  # days for rolling mean / std

spread = ko - beta * pep

roll_mean = spread.rolling(WINDOW).mean()
roll_std  = spread.rolling(WINDOW).std()
zscore    = (spread - roll_mean) / roll_std

# Drop leading NaNs produced by the rolling window
valid = zscore.dropna().index
spread = spread.loc[valid]
zscore = zscore.loc[valid]
ko     = ko.loc[valid]
pep    = pep.loc[valid]

# ── 4. TRADING SIGNALS ───────────────────────────────────────────────────────
#
# Convention (trading the spread = long KO / short PEP):
#   z > +2   → spread is high  → short spread  (short KO, long PEP)
#   z < -2   → spread is low   → long  spread  (long  KO, short PEP)
#   |z| < 0.5 → mean reversion achieved → exit
#   |z| > 3.5 → stop-loss → exit
#
# Position encoding:  +1 = long spread,  -1 = short spread,  0 = flat

ENTRY_Z     =  2.0
EXIT_Z      =  0.5
STOP_LOSS_Z =  3.5

position = 0          # current position
positions = []        # record of daily position

for z in zscore:
    if position == 0:
        # No open trade — look for an entry
        if z < -ENTRY_Z:
            position = 1    # z too low → long spread
        elif z > ENTRY_Z:
            position = -1   # z too high → short spread
    elif position == 1:
        # Long spread — exit on mean reversion or stop-loss
        if z > -EXIT_Z or abs(z) > STOP_LOSS_Z:
            position = 0
    elif position == -1:
        # Short spread — exit on mean reversion or stop-loss
        if z < EXIT_Z or abs(z) > STOP_LOSS_Z:
            position = 0

    positions.append(position)

positions = pd.Series(positions, index=zscore.index, name="position")

# ── 5. P&L SIMULATION ────────────────────────────────────────────────────────
#
# Daily spread return = daily change in (KO - beta * PEP).
# We trade yesterday's position against today's change (no look-ahead).

spread_returns = spread.diff()  # daily $ change in spread

# Shift positions by 1 so we act on yesterday's signal
strategy_returns = positions.shift(1) * spread_returns

# Normalise by the initial spread level so returns are percentage-like
initial_spread = abs(spread.iloc[0])
if initial_spread == 0:
    initial_spread = 1.0

strategy_returns_pct = strategy_returns / initial_spread

# Cumulative returns (compounded)
cumulative_returns = (1 + strategy_returns_pct).cumprod() - 1

# ── 6. VISUALISATION ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle("Pairs Trading: KO vs PEP  (Mean Reversion Strategy)", fontsize=14, fontweight="bold")

date_fmt = mdates.DateFormatter("%b '%y")

# ── (a) Normalised prices ─────────────────────────────────────────────────────
ax = axes[0]
ko_norm  = ko  / ko.iloc[0]
pep_norm = pep / pep.iloc[0]
ax.plot(ko_norm,  label="KO (normalised)",  color="#e63946")
ax.plot(pep_norm, label="PEP (normalised)", color="#457b9d")
ax.set_ylabel("Price (normalised)")
ax.set_title("(a) Normalised Stock Prices")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# ── (b) Spread ────────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(spread, color="#2d6a4f", linewidth=1, label="Spread = KO − β·PEP")
ax.axhline(spread.mean(), color="grey", linestyle="--", linewidth=0.8, label="Mean")
ax.fill_between(spread.index, spread, spread.mean(), alpha=0.15, color="#2d6a4f")
ax.set_ylabel("Spread ($)")
ax.set_title(f"(b) Spread  (β = {beta:.4f})")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# ── (c) Z-score with thresholds ───────────────────────────────────────────────
ax = axes[2]
ax.plot(zscore, color="#6d6875", linewidth=1, label="Z-score (30-day rolling)")
# Threshold lines
for level, style, label in [
    ( ENTRY_Z,     "--", f"+{ENTRY_Z} entry"),
    (-ENTRY_Z,     "--", f"−{ENTRY_Z} entry"),
    ( EXIT_Z,      ":",  f"+{EXIT_Z} exit"),
    (-EXIT_Z,      ":",  f"−{EXIT_Z} exit"),
    ( STOP_LOSS_Z, "-.", f"+{STOP_LOSS_Z} stop"),
    (-STOP_LOSS_Z, "-.", f"−{STOP_LOSS_Z} stop"),
]:
    ax.axhline(level, linestyle=style, color="black", linewidth=0.8, alpha=0.6, label=label)

# Shade regions where we hold a position
long_mask  = (positions == 1)
short_mask = (positions == -1)
ax.fill_between(zscore.index, zscore.min() - 0.5, zscore.max() + 0.5,
                where=long_mask,  alpha=0.10, color="green",  label="Long spread")
ax.fill_between(zscore.index, zscore.min() - 0.5, zscore.max() + 0.5,
                where=short_mask, alpha=0.10, color="red",    label="Short spread")

ax.set_ylabel("Z-score")
ax.set_title("(c) Rolling Z-score with Entry / Exit / Stop-loss Levels")
ax.legend(loc="upper left", ncol=3, fontsize=8)
ax.grid(alpha=0.3)

# ── (d) Cumulative returns ────────────────────────────────────────────────────
ax = axes[3]
pos_ret = cumulative_returns.copy()
neg_ret = cumulative_returns.copy()
pos_ret[pos_ret < 0] = np.nan
neg_ret[neg_ret >= 0] = np.nan

ax.plot(cumulative_returns, color="#333333", linewidth=1.2, label="Cumulative return")
ax.fill_between(cumulative_returns.index, 0, pos_ret, alpha=0.25, color="green")
ax.fill_between(cumulative_returns.index, 0, neg_ret, alpha=0.25, color="red")
ax.axhline(0, color="grey", linewidth=0.8)

final_ret = cumulative_returns.iloc[-1] * 100
ax.set_ylabel("Cumulative Return")
ax.set_title(f"(d) Strategy Cumulative Returns  (final: {final_ret:+.1f}%)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Shared x-axis formatting
for ax in axes:
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.tight_layout()
plt.savefig("pairs_trading.png", dpi=150, bbox_inches="tight")
print("Chart saved → pairs_trading.png")

# ── 7. SUMMARY STATS ─────────────────────────────────────────────────────────

total_days   = len(strategy_returns_pct.dropna())
trade_days   = (positions.shift(1) != 0).sum()
n_trades     = positions.diff().abs().gt(0).sum() // 2   # rough trade count

ann_return   = (1 + cumulative_returns.iloc[-1]) ** (252 / total_days) - 1
daily_vol    = strategy_returns_pct.std()
ann_vol      = daily_vol * np.sqrt(252)
sharpe       = ann_return / ann_vol if ann_vol > 0 else np.nan

running_max  = (1 + cumulative_returns).cummax()
drawdown     = (1 + cumulative_returns) / running_max - 1
max_drawdown = drawdown.min()

print("\n── Strategy Summary ───────────────────────────────")
print(f"  Period          : {valid[0].date()} → {valid[-1].date()}")
print(f"  Hedge ratio β   : {beta:.4f}")
print(f"  Total days      : {total_days}")
print(f"  Days in market  : {trade_days}  ({100*trade_days/total_days:.1f}%)")
print(f"  Approx trades   : {n_trades}")
print(f"  Final return    : {final_ret:+.1f}%")
print(f"  Ann. return     : {ann_return*100:+.1f}%")
print(f"  Ann. volatility : {ann_vol*100:.1f}%")
print(f"  Sharpe ratio    : {sharpe:.2f}")
print(f"  Max drawdown    : {max_drawdown*100:.1f}%")
print("───────────────────────────────────────────────────")
