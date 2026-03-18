"""
Pairs Trading Mean Reversion Strategy
======================================
Pick a stock pair from the menu (or run all of them) to:
  1. Download 2 years of adjusted close prices via yfinance
  2. Run an ADF test to check if the spread is stationary
  3. Estimate hedge ratio (beta) via OLS regression
  4. Compute spread and 30-day rolling z-score
  5. Generate entry / exit / stop-loss signals
  6. Simulate P&L day by day
  7. Produce a 4-panel chart saved as a PNG
  8. Print a summary (return, Sharpe, drawdown, ADF p-value)

When "all" is selected, a final comparison table ranks every pair by Sharpe ratio.
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — renders to file, never blocks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── PAIR CATALOGUE ─────────────────────────────────────────────────────────────
# Each entry: (display_name, ticker_a, ticker_b)
# Grouped by sector for the menu.

SECTORS = [
    ("Beverages",      [("KO/PEP",    "KO",   "PEP")]),
    ("Oil & Gas",      [("XOM/CVX",   "XOM",  "CVX")]),
    ("Big Tech",       [("MSFT/GOOGL","MSFT", "GOOGL")]),
    ("Banks",          [("JPM/BAC",   "JPM",  "BAC")]),
    ("Semiconductors", [("AMD/NVDA",  "AMD",  "NVDA")]),
    ("Retail",         [("WMT/TGT",   "WMT",  "TGT")]),
    ("Airlines",       [("DAL/UAL",   "DAL",  "UAL")]),
]

# Flatten to a numbered list for easy lookup by the user's choice
ALL_PAIRS: list[tuple[str, str, str]] = [
    pair for _, pairs in SECTORS for pair in pairs
]

# ── SIGNAL THRESHOLDS ──────────────────────────────────────────────────────────

WINDOW      = 30   # rolling z-score window (trading days)
ENTRY_Z     = 2.0
EXIT_Z      = 0.5
STOP_LOSS_Z = 3.5


# ── MENU ───────────────────────────────────────────────────────────────────────

def print_menu() -> None:
    """Print the numbered pair selection menu grouped by sector."""
    print("\n" + "=" * 55)
    print("  Pairs Trading  —  Select a Stock Pair")
    print("=" * 55)

    idx = 1
    for sector, pairs in SECTORS:
        print(f"\n  {sector}")
        for name, a, b in pairs:
            print(f"    [{idx}]  {name}  ({a} vs {b})")
            idx += 1

    print("\n  [all]  Run every pair sequentially")
    print("=" * 55)


def get_user_choice() -> list[tuple[str, str, str]]:
    """Prompt until the user enters a valid number or 'all'."""
    while True:
        raw = input("\nEnter your choice: ").strip().lower()

        if raw == "all":
            return ALL_PAIRS

        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(ALL_PAIRS):
                return [ALL_PAIRS[n - 1]]

        print(f"  Invalid choice. Enter a number between 1 and {len(ALL_PAIRS)}, or 'all'.")


# ── CORE ANALYSIS ──────────────────────────────────────────────────────────────

def run_pairs_analysis(ticker_a: str, ticker_b: str, name: str) -> dict | None:
    """
    Run the full pairs-trading pipeline for one pair.

    Parameters
    ----------
    ticker_a : str   Primary ticker (dependent variable in OLS)
    ticker_b : str   Secondary ticker (independent variable)
    name     : str   Display label, e.g. "KO/PEP"

    Returns
    -------
    dict with summary stats, or None if data could not be loaded.
    """
    print(f"\n{'─' * 55}")
    print(f"  Analysing pair: {name}  ({ticker_a} vs {ticker_b})")
    print(f"{'─' * 55}")

    # ── 1. DATA ────────────────────────────────────────────────────────────────

    end   = datetime.today()
    start = end - timedelta(days=2 * 365)

    print("  Downloading price data ...")
    # threads=False avoids a yfinance SQLite timezone-cache lock on some systems
    raw = yf.download(
        [ticker_a, ticker_b],
        start=start, end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty or "Close" not in raw.columns:
        print(f"  ERROR: No data returned for {ticker_a}/{ticker_b}. Skipping.")
        return None

    prices = raw["Close"][[ticker_a, ticker_b]].dropna()

    if prices.empty:
        print(f"  ERROR: Price data is empty after dropping NaNs. Skipping.")
        return None

    price_a = prices[ticker_a]
    price_b = prices[ticker_b]

    # ── 2. OLS HEDGE RATIO ─────────────────────────────────────────────────────
    # Regress ticker_a on ticker_b: ticker_a ~ alpha + beta * ticker_b

    X     = sm.add_constant(price_b)
    model = sm.OLS(price_a, X).fit()
    alpha = model.params["const"]
    beta  = model.params[ticker_b]

    print(f"  OLS  ->  alpha = {alpha:.4f},  beta = {beta:.4f},  R2 = {model.rsquared:.4f}")

    # ── 3. SPREAD & ADF STATIONARITY TEST ─────────────────────────────────────
    # Spread: residual of the OLS regression = ticker_a - beta * ticker_b
    # A stationary spread is required for mean-reversion to be a valid assumption.

    spread_full = price_a - beta * price_b

    adf_result    = adfuller(spread_full.dropna(), autolag="AIC")
    adf_pvalue    = adf_result[1]
    adf_stat      = adf_result[0]
    is_stationary = adf_pvalue < 0.05

    print(f"\n  ADF Test on Spread:")
    print(f"    Statistic : {adf_stat:.4f}")
    print(f"    p-value   : {adf_pvalue:.4f}")

    if is_stationary:
        print("    Result    : STATIONARY (p < 0.05) -- mean reversion is a reasonable assumption.")
    else:
        print("    Result    : NON-STATIONARY (p >= 0.05) -- WARNING: the spread may not mean-revert.")
        print("                Interpret results for this pair with caution.")

    # ── 4. ROLLING Z-SCORE ─────────────────────────────────────────────────────

    roll_mean = spread_full.rolling(WINDOW).mean()
    roll_std  = spread_full.rolling(WINDOW).std()
    zscore    = (spread_full - roll_mean) / roll_std

    # Trim leading NaNs from the rolling window
    valid   = zscore.dropna().index
    spread  = spread_full.loc[valid]
    zscore  = zscore.loc[valid]
    price_a = price_a.loc[valid]
    price_b = price_b.loc[valid]

    # ── 5. TRADING SIGNALS ─────────────────────────────────────────────────────
    #
    # Position encoding:  +1 = long spread,  -1 = short spread,  0 = flat
    #
    # z < -ENTRY_Z  -> spread is low   -> long  spread (buy ticker_a, short ticker_b)
    # z > +ENTRY_Z  -> spread is high  -> short spread (short ticker_a, buy ticker_b)
    # |z| < EXIT_Z  -> mean reversion achieved -> exit
    # |z| > STOP_LOSS_Z -> stop-loss -> exit

    position  = 0
    positions = []

    for z in zscore:
        if position == 0:
            # Flat -- look for entry
            if z < -ENTRY_Z:
                position = 1    # z too low  -> long spread
            elif z > ENTRY_Z:
                position = -1   # z too high -> short spread
        elif position == 1:
            # Long spread -- exit when z reverts or breaches stop
            if z > -EXIT_Z or abs(z) > STOP_LOSS_Z:
                position = 0
        elif position == -1:
            # Short spread -- exit when z reverts or breaches stop
            if z < EXIT_Z or abs(z) > STOP_LOSS_Z:
                position = 0

        positions.append(position)

    positions = pd.Series(positions, index=zscore.index, name="position")

    # ── 6. P&L SIMULATION ─────────────────────────────────────────────────────
    #
    # Daily spread return = daily change in (ticker_a - beta * ticker_b).
    # Shift positions by 1 so we act on yesterday's signal (no look-ahead).

    spread_returns = spread.diff()                     # daily $ change in spread
    strategy_returns = positions.shift(1) * spread_returns

    # Normalise by initial spread so returns are percentage-like
    initial_spread = abs(spread.iloc[0]) or 1.0
    strategy_returns_pct = strategy_returns / initial_spread

    # Cumulative compounded returns
    cumulative_returns = (1 + strategy_returns_pct).cumprod() - 1

    # ── 7. SUMMARY STATS ──────────────────────────────────────────────────────

    total_days   = len(strategy_returns_pct.dropna())
    trade_days   = int((positions.shift(1) != 0).sum())
    n_trades     = int(positions.diff().abs().gt(0).sum()) // 2  # approx round-trips

    final_ret    = float(cumulative_returns.iloc[-1])
    ann_return   = (1 + final_ret) ** (252 / total_days) - 1
    daily_vol    = strategy_returns_pct.std()
    ann_vol      = daily_vol * np.sqrt(252)
    sharpe       = ann_return / ann_vol if ann_vol > 0 else float("nan")

    running_max  = (1 + cumulative_returns).cummax()
    drawdown     = (1 + cumulative_returns) / running_max - 1
    max_drawdown = float(drawdown.min())

    # ── 8. CHART ──────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"Pairs Trading: {name}  ({ticker_a} vs {ticker_b})  -- Mean Reversion Strategy",
        fontsize=14, fontweight="bold",
    )

    date_fmt = mdates.DateFormatter("%b '%y")

    # (a) Normalised prices
    ax = axes[0]
    ax.plot(price_a / price_a.iloc[0], label=f"{ticker_a} (normalised)", color="#e63946")
    ax.plot(price_b / price_b.iloc[0], label=f"{ticker_b} (normalised)", color="#457b9d")
    ax.set_ylabel("Price (normalised)")
    ax.set_title(f"(a) Normalised Stock Prices -- {name}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # (b) Spread
    ax = axes[1]
    ax.plot(spread, color="#2d6a4f", linewidth=1,
            label=f"Spread = {ticker_a} - beta*{ticker_b}")
    ax.axhline(spread.mean(), color="grey", linestyle="--", linewidth=0.8, label="Mean")
    ax.fill_between(spread.index, spread, spread.mean(), alpha=0.15, color="#2d6a4f")
    ax.set_ylabel("Spread ($)")
    ax.set_title(f"(b) Spread  (beta = {beta:.4f}  |  ADF p = {adf_pvalue:.4f})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # (c) Z-score with thresholds and position shading
    ax = axes[2]
    ax.plot(zscore, color="#6d6875", linewidth=1,
            label=f"Z-score ({WINDOW}-day rolling)")

    for level, style, lbl in [
        ( ENTRY_Z,     "--", f"+{ENTRY_Z} entry"),
        (-ENTRY_Z,     "--", f"-{ENTRY_Z} entry"),
        ( EXIT_Z,      ":",  f"+{EXIT_Z} exit"),
        (-EXIT_Z,      ":",  f"-{EXIT_Z} exit"),
        ( STOP_LOSS_Z, "-.", f"+{STOP_LOSS_Z} stop"),
        (-STOP_LOSS_Z, "-.", f"-{STOP_LOSS_Z} stop"),
    ]:
        ax.axhline(level, linestyle=style, color="black",
                   linewidth=0.8, alpha=0.6, label=lbl)

    z_min, z_max = zscore.min() - 0.5, zscore.max() + 0.5
    ax.fill_between(zscore.index, z_min, z_max,
                    where=(positions == 1),  alpha=0.10,
                    color="green", label="Long spread")
    ax.fill_between(zscore.index, z_min, z_max,
                    where=(positions == -1), alpha=0.10,
                    color="red",   label="Short spread")

    ax.set_ylabel("Z-score")
    ax.set_title("(c) Rolling Z-score with Entry / Exit / Stop-loss Levels")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Cumulative returns
    ax = axes[3]
    pos_ret = cumulative_returns.where(cumulative_returns >= 0)
    neg_ret = cumulative_returns.where(cumulative_returns < 0)
    ax.plot(cumulative_returns, color="#333333", linewidth=1.2, label="Cumulative return")
    ax.fill_between(cumulative_returns.index, 0, pos_ret, alpha=0.25, color="green")
    ax.fill_between(cumulative_returns.index, 0, neg_ret, alpha=0.25, color="red")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"(d) Strategy Cumulative Returns  (final: {final_ret * 100:+.1f}%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Shared x-axis formatting
    for ax in axes:
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()

    safe_name  = name.replace("/", "_")
    chart_path = f"pairs_trading_{safe_name}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart saved -> {chart_path}")

    # ── 9. PRINT SUMMARY ──────────────────────────────────────────────────────

    dash = "─" * max(0, 35 - len(name))
    print(f"\n  -- Summary: {name} {dash}")
    print(f"  Period          : {valid[0].date()} -> {valid[-1].date()}")
    print(f"  Hedge ratio beta: {beta:.4f}")
    print(f"  Total days      : {total_days}")
    print(f"  Days in market  : {trade_days}  ({100 * trade_days / total_days:.1f}%)")
    print(f"  Approx trades   : {n_trades}")
    print(f"  Final return    : {final_ret * 100:+.1f}%")
    print(f"  Ann. return     : {ann_return * 100:+.1f}%")
    print(f"  Ann. volatility : {ann_vol * 100:.1f}%")
    print(f"  Sharpe ratio    : {sharpe:.2f}")
    print(f"  Max drawdown    : {max_drawdown * 100:.1f}%")
    stationary_str = "stationary" if is_stationary else "NON-STATIONARY -- caution"
    print(f"  ADF p-value     : {adf_pvalue:.4f}  ({stationary_str})")
    print(f"  {'─' * 47}")

    return {
        "name":         name,
        "ticker_a":     ticker_a,
        "ticker_b":     ticker_b,
        "adf_pvalue":   adf_pvalue,
        "stationary":   is_stationary,
        "beta":         beta,
        "n_trades":     n_trades,
        "final_ret":    final_ret,
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_drawdown": max_drawdown,
    }


# ── COMPARISON TABLE (used when "all" is selected) ─────────────────────────────

def print_comparison_table(results: list[dict]) -> None:
    """Print all pairs ranked by Sharpe ratio, highest first."""
    if not results:
        print("\n  No results to compare.")
        return

    ranked = sorted(
        results,
        key=lambda r: r["sharpe"] if not np.isnan(r["sharpe"]) else -999,
        reverse=True,
    )

    col = "{:<5} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}"
    header = col.format("Rank", "Pair", "Sharpe", "AnnRet%", "MaxDD%",
                        "Trades", "ADF-p", "Stationary?")
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  All-Pairs Comparison  --  Ranked by Sharpe Ratio")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for rank, r in enumerate(ranked, start=1):
        sharpe_str = f"{r['sharpe']:.2f}" if not np.isnan(r["sharpe"]) else "  N/A"
        print(col.format(
            rank,
            r["name"],
            sharpe_str,
            f"{r['ann_return'] * 100:+.1f}",
            f"{r['max_drawdown'] * 100:.1f}",
            r["n_trades"],
            f"{r['adf_pvalue']:.4f}",
            "Yes" if r["stationary"] else "No",
        ))

    print(sep)
    print()


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

def main() -> None:
    print_menu()
    selected = get_user_choice()

    results = []
    for name, ticker_a, ticker_b in selected:
        stats = run_pairs_analysis(ticker_a, ticker_b, name)
        if stats is not None:
            results.append(stats)

    # If the user ran all pairs, show the ranking table
    if len(selected) > 1 and results:
        print_comparison_table(results)


if __name__ == "__main__":
    main()
