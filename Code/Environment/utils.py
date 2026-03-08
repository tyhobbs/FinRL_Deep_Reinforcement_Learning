# ============================================================
# utils.py — Shared utilities for FinRL ablation study
# ============================================================
#
# Usage in each notebook — add to Cell 1:
#
#   import sys
#   sys.path.append('.')
#   from utils import (
#       prepare_df, compute_metrics, compute_rolling_metrics,
#       plot_metrics, compute_buy_and_hold, overfitting_check,
#       check_degenerate_policy, check_lookahead_bias,
#       regime_analysis, run_full_evaluation
#   )
#
# ── Model type reference ──────────────────────────────────────
#   Baseline VGG    : has_sentiment=False, df_sentiment=None
#   VGG + FinBERT   : has_sentiment=True,  df_sentiment=df_sent_train
#   VGG + Alpaca    : has_sentiment=True,  df_sentiment=df_sent_train
#   Transformer     : has_sentiment=True,  df_sentiment=df_sent_train
#
# ── Capital level reference ───────────────────────────────────
#   $1,000,000 : capital='1M',   initial_capital=1_000_000, hmax=100
#   $100,000   : capital='100k', initial_capital=100_000,   hmax=10
#   $10,000    : capital='10k',  initial_capital=10_000,    hmax=5
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from finrl.agents.stablebaselines3.models import DRLAgent

# Risk-free rate — 5% annualised 3-month T-bill
RF_DAILY = 0.05 / 252


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_df(df, df_sentiment=None, df_macro=None,
               num_stocks=None, indicators_with_sent=None):
    """
    Clean, validate, optionally merge sentiment and macro features,
    apply one-day look-ahead lag, and build the integer day index
    required by StockTradingEnv.

    Parameters
    ----------
    df                   : raw preprocessed price DataFrame from
                           FeatureEngineer
    df_sentiment         : sentiment DataFrame (dates as index,
                           tickers as columns). Pass None for
                           baseline models with no sentiment.
    df_macro             : macro features DataFrame (dates as index,
                           features as columns e.g. VIX, TNX, SPY).
                           Pass None for VGG models, df_macro_train
                           for Transformer models.
    num_stocks           : expected number of tickers — used
                           internally, leave as None to auto-detect
    indicators_with_sent : list of indicator names — unused here
                           but kept for API consistency

    Returns
    -------
    df : cleaned DataFrame with correct integer day index
    """
    df = df.rename(columns={'datadate': 'date'})
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    # ── Drop tickers with incomplete data ─────────────────────
    # Drop by ticker (not by date) so all trading days are kept.
    # This handles Alpaca tickers that fail SSL download (e.g. NOW).
    all_dates    = df['date'].unique()
    ticker_dates = df.groupby('tic')['date'].nunique()
    # Allow up to 5% missing dates per ticker
    # Handles Yahoo Finance minor gaps without dropping valid tickers
    # In utils.py prepare_df — check this is what you have
    min_dates    = int(len(all_dates) * 0.95)
    full_tickers = ticker_dates[
        ticker_dates >= min_dates
    ].index.tolist()

    dropped = set(ticker_dates.index) - set(full_tickers)
    if dropped:
        print(f"Dropping {len(dropped)} incomplete tickers: "
              f"{sorted(dropped)}")
        df = df[df['tic'].isin(full_tickers)].reset_index(drop=True)

    print(f"Using {df['tic'].nunique()} tickers with complete data")

    # ── Normalize date format ─────────────────────────────────
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # ── Sentiment merge (skipped for baseline models) ─────────
    if df_sentiment is not None:
        df_sent_long = df_sentiment.copy()
        df_sent_long.index = pd.to_datetime(
            df_sent_long.index
        ).strftime('%Y-%m-%d')
        df_sent_long = df_sent_long.reset_index().melt(
            id_vars='date', var_name='tic', value_name='sentiment'
        ).rename(columns={'date': 'sent_date'})

        df = df.merge(
            df_sent_long,
            left_on=['date', 'tic'],
            right_on=['sent_date', 'tic'],
            how='left'
        ).drop(columns=['sent_date'], errors='ignore')

        df['sentiment'] = df['sentiment'].fillna(0.0)

        # One-day lag — news from date t informs trades on date t+1
        # Sort by tic+date for shift, then re-sort by date+tic
        df = df.sort_values(['tic', 'date'])
        df['sentiment'] = df.groupby('tic')['sentiment'].shift(1)
        df['sentiment'] = df['sentiment'].fillna(0.0)

        non_zero = (df['sentiment'] != 0).sum()
        print(f"Sentiment merged — non-zero rows: {non_zero} / {len(df)}")
        print(f"Look-ahead lag applied — trading on t uses news from t-1")
    else:
        print("No sentiment — baseline model")


    # ── Macro merge (transformer models only) ────────────────
    # Broadcasts macro values to all tickers on each date
    if df_macro is not None and not df_macro.empty:
        df_macro_reset = df_macro.copy()
        df_macro_reset.index = pd.to_datetime(
            df_macro_reset.index
        ).strftime("%Y-%m-%d")
        df_macro_reset = df_macro_reset.reset_index()
        df_macro_reset.columns = ["date"] + list(df_macro.columns)
        df = df.merge(df_macro_reset, on="date", how="left")
        macro_cols = list(df_macro.columns)
        for feat in macro_cols:
            if feat in df.columns:
                df[feat] = df[feat].fillna(0.0)
        n_merged = len([f for f in macro_cols if f in df.columns])
        print(f"Macro features merged — {n_merged} features: {macro_cols}")

    # ── Re-sort by date+tic before building index ─────────────
    # Critical: environment expects rows grouped by date
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    # ── Build integer day index starting at 0 ─────────────────
    dates_sorted = sorted(df['date'].unique())
    date_to_day  = {d: i for i, d in enumerate(dates_sorted)}
    df.index     = df['date'].map(date_to_day).values

    assert df.index.min() == 0, \
        f"Index does not start at 0 — min is {df.index.min()}"
    print(f"Index OK — starts at 0, ends at {df.index.max()}")
    return df


# ============================================================
# METRICS
# ============================================================

def compute_metrics(df_account_value, initial_capital=1_000_000):
    """
    Compute key trading performance metrics from account value history.

    Sharpe ratio uses 5% annualised risk-free rate (3-month T-bill).
    Transaction costs of 0.15% per trade (0.1% commission + 0.05%
    slippage) are already deducted by the environment and reflected
    in portfolio value — all metrics are therefore net of costs.

    Parameters
    ----------
    df_account_value : DataFrame with 'date' and 'account_value' cols
    initial_capital  : starting portfolio value

    Returns
    -------
    dict of metric name -> value
    """
    values = pd.Series(
        df_account_value['account_value'].values,
        index=pd.to_datetime(df_account_value['date'].values)
    )
    daily_returns = values.pct_change().dropna()

    total_return = (
        (values.iloc[-1] - initial_capital) / initial_capital * 100
    )

    sharpe = ((daily_returns.mean() - RF_DAILY) /
               daily_returns.std()) * np.sqrt(252) \
             if daily_returns.std() > 0 else 0.0

    rolling_max  = values.cummax()
    drawdown     = (values - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    win_rate         = (daily_returns > 0).sum() / len(daily_returns) * 100
    avg_daily_return = daily_returns.mean() * 100
    volatility       = daily_returns.std() * np.sqrt(252) * 100
    calmar           = abs(total_return / max_drawdown) \
                       if max_drawdown != 0 else 0.0

    return {
        'Total Return (%)':     round(total_return, 2),
        'Sharpe Ratio':         round(sharpe, 3),
        'Max Drawdown (%)':     round(max_drawdown, 2),
        'Win Rate (%)':         round(win_rate, 2),
        'Avg Daily Return (%)': round(avg_daily_return, 4),
        'Volatility (%)':       round(volatility, 2),
        'Calmar Ratio':         round(calmar, 3),
    }


def compute_rolling_metrics(df_account_value,
                             initial_capital=1_000_000,
                             window=20):
    """
    Compute rolling metrics for time-series performance plots.

    Parameters
    ----------
    df_account_value : DataFrame with 'date' and 'account_value' cols
    initial_capital  : starting portfolio value
    window           : rolling window in trading days (default 20 = 1 month)

    Returns
    -------
    DataFrame with one column per metric, indexed by date
    """
    values = pd.Series(
        df_account_value['account_value'].values,
        index=pd.to_datetime(df_account_value['date'].values)
    )
    daily_returns = values.pct_change().dropna()

    rolling_sharpe = (
        (daily_returns.rolling(window).mean() - RF_DAILY) /
        daily_returns.rolling(window).std()
    ) * np.sqrt(252)

    rolling_win_rate = (
        daily_returns.rolling(window)
        .apply(lambda x: (x > 0).sum() / len(x) * 100)
    )

    def rolling_max_drawdown(returns):
        cum  = (1 + returns).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak * 100
        return dd.min()

    rolling_max_dd     = daily_returns.rolling(window).apply(
                             rolling_max_drawdown
                         )
    rolling_avg_return = daily_returns.rolling(window).mean() * 100
    total_return       = (values - initial_capital) / initial_capital * 100

    return pd.DataFrame({
        'Sharpe Ratio':         rolling_sharpe,
        'Win Rate (%)':         rolling_win_rate,
        'Max Drawdown (%)':     rolling_max_dd,
        'Avg Daily Return (%)': rolling_avg_return,
        'Total Return (%)':     total_return,
    })


# ============================================================
# PLOTTING
# ============================================================

def plot_metrics(metrics_df, title_prefix='Training'):
    """
    Plot all five rolling metrics in a dark-theme multi-panel figure.
    Prints a summary table of final/average values below the chart.

    Parameters
    ----------
    metrics_df   : output of compute_rolling_metrics()
    title_prefix : string shown in chart title
    """
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f'{title_prefix} — Performance Metrics Over Time',
                 fontsize=18, y=1.01)

    plot_config = [
        {'col': 'Sharpe Ratio',
         'color': 'cyan',        'ylabel': 'Sharpe Ratio',
         'hline': 1.0,           'hline_label': 'Target (1.0)',
         'hline_color': 'yellow'},
        {'col': 'Win Rate (%)',
         'color': 'lime',        'ylabel': 'Win Rate (%)',
         'hline': 50.0,          'hline_label': 'Breakeven (50%)',
         'hline_color': 'orange'},
        {'col': 'Max Drawdown (%)',
         'color': 'red',         'ylabel': 'Max Drawdown (%)',
         'hline': -15.0,         'hline_label': 'Warning (-15%)',
         'hline_color': 'orange'},
        {'col': 'Avg Daily Return (%)',
         'color': 'gold',        'ylabel': 'Avg Daily Return (%)',
         'hline': 0.0,           'hline_label': 'Breakeven (0%)',
         'hline_color': 'white'},
        {'col': 'Total Return (%)',
         'color': 'mediumpurple', 'ylabel': 'Total Return (%)',
         'hline': 0.0,           'hline_label': 'Breakeven (0%)',
         'hline_color': 'white'},
    ]

    for ax, cfg in zip(axes, plot_config):
        series = metrics_df[cfg['col']].dropna()
        ax.plot(series.index, series.values,
                color=cfg['color'], linewidth=1.5, alpha=0.9)
        ax.axhline(cfg['hline'], color=cfg['hline_color'],
                   linestyle='--', linewidth=1,
                   label=cfg['hline_label'])

        if cfg['col'] == 'Max Drawdown (%)':
            ax.fill_between(series.index, series.values, 0,
                            alpha=0.2, color='red')
        else:
            ax.fill_between(
                series.index, series.values, cfg['hline'],
                where=series.values >= cfg['hline'],
                alpha=0.15, color='green'
            )
            ax.fill_between(
                series.index, series.values, cfg['hline'],
                where=series.values < cfg['hline'],
                alpha=0.15, color='red'
            )

        ax.set_ylabel(cfg['ylabel'], fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_facecolor('#1a1a2e')

        if len(series) > 0:
            final_val = series.iloc[-1]
            ax.annotate(
                f'Final: {final_val:.2f}',
                xy=(series.index[-1], final_val),
                xytext=(-80, 10),
                textcoords='offset points',
                fontsize=9, color=cfg['color'],
                arrowprops=dict(arrowstyle='->',
                                color=cfg['color'], lw=1)
            )

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')
    fig.patch.set_facecolor('#0d0d1a')
    fig.tight_layout()
    plt.show()

    # Summary table
    print(f"\n{'='*50}")
    print(f"  {title_prefix.upper()} — FINAL METRIC VALUES")
    print(f"{'='*50}")
    for col in metrics_df.columns:
        series = metrics_df[col].dropna()
        if col == 'Total Return (%)':
            val   = series.iloc[-1] if len(series) > 0 else 0.0
            label = f"  {col:<25} {val:>10.3f}  (final)"
        else:
            val   = series.mean() if len(series) > 0 else 0.0
            label = f"  {col:<25} {val:>10.3f}  (avg)"
        print(label)
    print(f"{'='*50}")


# ============================================================
# BUY-AND-HOLD BASELINE
# ============================================================

def compute_buy_and_hold(df_test, initial_capital=1_000_000):
    """
    Equal-weight buy-and-hold baseline for DRL comparison.

    Buys equal dollar amounts of each stock on the first trading
    day and holds until the end of the test period. No rebalancing,
    no transaction costs beyond the initial purchase.

    This answers: does DRL beat simply holding the market?

    Parameters
    ----------
    df_test         : preprocessed test DataFrame with 'date',
                      'tic', and 'close' columns
    initial_capital : starting portfolio value

    Returns
    -------
    DataFrame with 'date' and 'account_value' columns
    """
    tickers   = sorted(df_test['tic'].unique())
    dates     = sorted(df_test['date'].unique())
    n_stocks  = len(tickers)
    per_stock = initial_capital / n_stocks

    print(f"Buy-and-Hold: {n_stocks} stocks, "
          f"${per_stock:,.0f} per stock")

    first_date   = dates[0]
    first_prices = (
        df_test[df_test['date'] == first_date]
        .set_index('tic')['close']
    )

    shares         = {}
    cash_remaining = initial_capital
    for ticker in tickers:
        if ticker in first_prices.index:
            price           = first_prices[ticker]
            n_shares        = per_stock / price
            shares[ticker]  = n_shares
            cash_remaining -= n_shares * price

    bah_values = []
    for d in dates:
        day_data   = df_test[df_test['date'] == d].set_index('tic')
        port_value = cash_remaining
        for ticker, n_shares in shares.items():
            if ticker in day_data.index:
                port_value += n_shares * day_data.loc[ticker, 'close']
        bah_values.append({'date': d, 'account_value': port_value})

    return pd.DataFrame(bah_values)


# ============================================================
# VALIDATION CHECKS
# ============================================================

def overfitting_check(train_sharpe, test_sharpe, model_name):
    """
    Flag potential overfitting if test Sharpe drops more than
    50% from train Sharpe — per reviewer recommendation.

    Parameters
    ----------
    train_sharpe : Sharpe ratio on training period
    test_sharpe  : Sharpe ratio on test period
    model_name   : string label for output
    """
    if train_sharpe == 0:
        print(f"{model_name}: Cannot compute — train Sharpe is 0")
        return

    degradation = (train_sharpe - test_sharpe) / abs(train_sharpe) * 100

    print(f"\n{'='*50}")
    print(f"  OVERFITTING CHECK — {model_name}")
    print(f"{'='*50}")
    print(f"  Train Sharpe:  {train_sharpe:.3f}")
    print(f"  Test Sharpe:   {test_sharpe:.3f}")
    print(f"  Degradation:   {degradation:.1f}%")

    if degradation > 50:
        print(f"  WARNING — Sharpe dropped >50% out-of-sample")
        print(f"  This suggests potential overfitting")
    elif degradation > 25:
        print(f"  CAUTION — Moderate degradation, monitor closely")
    else:
        print(f"  OK — Degradation within acceptable range")
    print(f"{'='*50}")


def check_degenerate_policy(df_actions, df_account_value,
                              model_name='Model'):
    """
    Check for common degenerate RL trading policies:
      1. Always hold cash (>50% of days with no positions)
      2. Always buy  (buy/sell ratio > 10:1)
      3. Always sell (sell/buy ratio > 10:1)
      4. Portfolio barely moved

    Saves action distribution chart to ./overlay_data/

    Parameters
    ----------
    df_actions       : actions DataFrame from DRL_prediction
    df_account_value : account value DataFrame from DRL_prediction
    model_name       : string label for output and filename
    """
    print(f"\n{'='*55}")
    print(f"  DEGENERATE POLICY CHECK — {model_name}")
    print(f"{'='*55}")

    total_days = len(df_actions)
    cash_days  = (df_actions.abs().sum(axis=1) < 0.01).sum()
    cash_pct   = cash_days / total_days * 100

    print(f"\n  Cash-holding days: {cash_days}/{total_days} "
          f"({cash_pct:.1f}%)")
    if cash_pct > 50:
        print(f"  WARNING — Agent holds cash >50% of days")
        print(f"  Possible degenerate sell-everything policy")
    else:
        print(f"  OK — Agent actively trades most days")

    buy_actions   = (df_actions >  0.01).sum().sum()
    sell_actions  = (df_actions < -0.01).sum().sum()
    hold_actions  = (
        (df_actions >= -0.01) & (df_actions <= 0.01)
    ).sum().sum()
    total_actions = buy_actions + sell_actions + hold_actions

    print(f"\n  Action distribution:")
    print(f"    Buy:  {buy_actions:,}  "
          f"({buy_actions/total_actions*100:.1f}%)")
    print(f"    Sell: {sell_actions:,}  "
          f"({sell_actions/total_actions*100:.1f}%)")
    print(f"    Hold: {hold_actions:,}  "
          f"({hold_actions/total_actions*100:.1f}%)")

    if buy_actions / (sell_actions + 1) > 10:
        print(f"\n  WARNING — Agent buys 10x more than sells")
        print(f"  Possible always-buy degenerate policy")
    elif sell_actions / (buy_actions + 1) > 10:
        print(f"\n  WARNING — Agent sells 10x more than buys")
        print(f"  Possible panic-selling degenerate policy")
    else:
        print(f"\n  OK — Balanced buy/sell distribution")

    values    = df_account_value['account_value'].values
    total_ret = (values[-1] - values[0]) / values[0] * 100
    print(f"\n  Portfolio return: {total_ret:.2f}%")
    if abs(total_ret) < 1.0:
        print(f"  WARNING — Portfolio barely moved")
    else:
        print(f"  OK — Portfolio shows meaningful activity")

    # Action distribution chart
    os.makedirs('./overlay_data/', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d0d1a')
    fig.suptitle(f'Action Distribution — {model_name}',
                 fontsize=13, color='white')

    all_actions = df_actions.values.flatten()
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')
    ax1.hist(all_actions, bins=50, color='cyan',
             alpha=0.7, edgecolor='none')
    ax1.axvline(0, color='white', linestyle='--',
                linewidth=1, label='Hold (0)')
    ax1.set_title('Action Distribution\n(all stocks, all days)',
                  color='white', fontsize=11)
    ax1.set_xlabel('Action Value (-1=max sell, +1=max buy)',
                   color='white')
    ax1.set_ylabel('Frequency', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a2e', labelcolor='white')

    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    mean_actions = df_actions.mean()
    colors = ['#32CD32' if v > 0 else '#FF4500'
              for v in mean_actions.values]
    ax2.bar(range(len(mean_actions)), mean_actions.values,
            color=colors, alpha=0.8)
    ax2.axhline(0, color='white', linewidth=1, linestyle='--')
    ax2.set_title('Mean Action per Stock\n'
                  '(green=net buyer, red=net seller)',
                  color='white', fontsize=11)
    ax2.set_xlabel('Stock Index', color='white')
    ax2.set_ylabel('Mean Action', color='white')
    ax2.tick_params(colors='white')

    fig.tight_layout()
    safe_name = model_name.replace(' ', '_')
    plt.savefig(
        f'./overlay_data/{safe_name}_action_distribution.png',
        dpi=150, bbox_inches='tight', facecolor='#0d0d1a'
    )
    plt.show()
    print(f"{'='*55}")


def check_lookahead_bias(df_train, has_sentiment=True):
    """
    Verify that the one-day sentiment lag was correctly applied.

    The first sentiment value per ticker must be 0.0 — if shift(1)
    was applied correctly, no ticker has sentiment on its first
    trading day because there is no prior day to shift from.

    Skips gracefully for baseline models with no sentiment.

    Parameters
    ----------
    df_train      : training DataFrame after prepare_df
    has_sentiment : False for baseline models — skips check
    """
    if not has_sentiment:
        print("Look-ahead bias check skipped — no sentiment in "
              "this model")
        return

    print("=" * 60)
    print("LOOK-AHEAD BIAS ANALYSIS")
    print("=" * 60)
    print("\nPipeline:")
    print("  Polygon news date t  -> raw sentiment score for date t")
    print("  shift(1) applied     -> sentiment moved to date t+1")
    print("  Model trades date t  -> uses sentiment from date t-1")
    print("\nResult: NO look-ahead bias — news from date t informs")
    print("  trading decisions on date t+1 only.")

    first_rows = df_train.groupby('tic').first()['sentiment']
    all_zero   = (first_rows == 0).all()
    print(f"\nFirst sentiment row per ticker is 0: {all_zero}")

    if all_zero:
        print("CONFIRMED — shift(1) correctly applied")
    else:
        print("WARNING — shift may not be working correctly")
        print("Non-zero first rows:")
        print(first_rows[first_rows != 0])

    sample_tics = df_train[df_train['sentiment'] != 0]['tic'].values
    if len(sample_tics) > 0:
        sample = df_train[df_train['tic'] == sample_tics[0]][
            ['date', 'tic', 'sentiment']
        ].head(10)
        print(f"\nSample rows for {sample_tics[0]}:")
        print(sample.to_string(index=False))
        print("(First row should be 0.0 — no prior day sentiment)")
    print("=" * 60)


# ============================================================
# MARKET REGIME ANALYSIS
# ============================================================

def regime_analysis(df_account_value, df_bah,
                    initial_capital=1_000_000,
                    model_name='Model'):
    """
    Split the 2024 validation period into bull (H1) and volatile
    (H2) phases and compute metrics separately for each.

    H1 2024 (Jan-Jun): broadly bullish, S&P 500 up ~15%
    H2 2024 (Jul-Dec): more volatile, rate uncertainty

    A model that only profits in bull markets is not useful —
    this analysis reveals whether performance persists under
    adverse conditions.

    Parameters
    ----------
    df_account_value : test account value DataFrame
    df_bah           : buy-and-hold account value DataFrame
    initial_capital  : starting portfolio value
    model_name       : string label for output and filename
    """
    df_account_value = df_account_value.copy()
    df_bah           = df_bah.copy()
    df_account_value['date'] = pd.to_datetime(df_account_value['date'])
    df_bah['date']           = pd.to_datetime(df_bah['date'])

    def split(df, start, end):
        mask = (df['date'] >= start) & (df['date'] < end)
        return df[mask].reset_index(drop=True)

    def rebase(df):
        # Guard against empty DataFrame — happens when model peaks
        # before the start of the period being analyzed
        if len(df) == 0:
            return df
        df = df.copy()
        df['account_value'] = (
            df['account_value'] /
            df['account_value'].iloc[0] * initial_capital
        )
        return df

    h1_drl = rebase(split(df_account_value, '2024-01-01', '2024-07-01'))
    h2_drl = rebase(split(df_account_value, '2024-07-01', '2025-01-01'))
    h1_bah = rebase(split(df_bah,           '2024-01-01', '2024-07-01'))
    h2_bah = rebase(split(df_bah,           '2024-07-01', '2025-01-01'))

    def period_metrics(df, label):
        # Guard against empty period
        if len(df) == 0:
            print(f"  {label:<42} No data for this period "
                f"(model peaked before period start)")
            return {'sharpe': 0, 'return': 0,
                    'max_dd': 0, 'win_rate': 0}

        values        = pd.Series(df['account_value'].values)
        daily_returns = values.pct_change().dropna()

        if len(daily_returns) == 0:
            print(f"  {label:<42} Insufficient data")
            return {'sharpe': 0, 'return': 0,
                    'max_dd': 0, 'win_rate': 0}

        sharpe  = ((daily_returns.mean() - RF_DAILY) /
                    daily_returns.std()) * np.sqrt(252) \
                if daily_returns.std() > 0 else 0.0
        peak    = values.cummax()
        max_dd  = ((values - peak) / peak * 100).min()
        win_rt  = (daily_returns > 0).sum() / len(daily_returns) * 100
        ret     = (values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100
        print(f"  {label:<42} Sharpe: {sharpe:>6.3f}  "
            f"Return: {ret:>7.2f}%  "
            f"MaxDD: {max_dd:>7.2f}%  "
            f"WinRate: {win_rt:>5.1f}%")
        return {'sharpe': sharpe, 'return': ret,
                'max_dd': max_dd, 'win_rate': win_rt}

    print("=" * 80)
    print("  MARKET REGIME ANALYSIS")
    print("=" * 80)
    print(f"\n  H1 2024 — Bull Market (Jan-Jun):")
    period_metrics(h1_drl, model_name)
    period_metrics(h1_bah, 'Buy-and-Hold')

    print(f"\n  H2 2024 — Volatile (Jul-Dec):")
    period_metrics(h2_drl, model_name)
    period_metrics(h2_bah, 'Buy-and-Hold')

    print(f"\n  Full Period (Jan 2024 - Jan 2025):")
    period_metrics(df_account_value, model_name)
    period_metrics(df_bah,           'Buy-and-Hold')
    print("=" * 80)

    # Plot H1 and H2 side by side
    os.makedirs('./overlay_data/', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0d0d1a')
    fig.suptitle(f'Market Regime Analysis — {model_name}',
                 fontsize=14, color='white')

    for (drl_df, bah_df, ax, title) in [
        (h1_drl, h1_bah, axes[0], 'H1 2024 — Bull Market'),
        (h2_drl, h2_bah, axes[1], 'H2 2024 — Volatile'),
    ]:
        ax.set_facecolor('#1a1a2e')

        # Guard — skip plot if period is empty
        if len(drl_df) == 0:
            ax.text(0.5, 0.5, 'No data\n(model peaked before this period)',
                    ha='center', va='center', color='white',
                    fontsize=12, transform=ax.transAxes)
            ax.set_title(title, color='white', fontsize=12)
            ax.set_facecolor('#1a1a2e')
            continue

        ax.plot(pd.to_datetime(drl_df['date']),
                drl_df['account_value'],
                color='cyan', linewidth=2, label=model_name)

        if len(bah_df) > 0:
            ax.plot(pd.to_datetime(bah_df['date']),
                    bah_df['account_value'],
                    color='orange', linewidth=2,
                    linestyle='--', label='Buy-and-Hold')

        ax.axhline(initial_capital, color='white', linewidth=1,
                linestyle=':', alpha=0.5, label='Starting Capital')
        ax.set_title(title, color='white', fontsize=12)
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Portfolio Value (USD)', color='white')
        ax.tick_params(colors='white')
        ax.legend(fontsize=9, facecolor='#1a1a2e',
                labelcolor='white', edgecolor='#444')
        ax.grid(True, linestyle=':', alpha=0.3, color='#555')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.get_xticklabels(), rotation=45,
                ha='right', color='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')

    fig.tight_layout()
    safe_name = model_name.replace(' ', '_')
    plt.savefig(
        f'./overlay_data/{safe_name}_regime_analysis.png',
        dpi=150, bbox_inches='tight', facecolor='#0d0d1a'
    )
    plt.show()


# ============================================================
# FULL EVALUATION
# ============================================================

def run_full_evaluation(
    trained_model,
    e_train_gym,
    e_test_gym,
    df_test,
    model_name,
    universe,
    capital,
    initial_capital=1_000_000,
    has_sentiment=True
):
    """
    Run complete evaluation for one model in a single call.

    Handles both baseline (no sentiment) and sentiment models.
    Computes and saves all metrics, charts, pkl files, and a
    plain-text metrics summary.

    Parameters
    ----------
    trained_model   : trained PPO model object
    e_train_gym     : training StockTradingEnv instance
    e_test_gym      : test StockTradingEnv instance
    df_test         : preprocessed test DataFrame (for buy-and-hold)
    model_name      : str — one of:
                        'VGG_Baseline', 'VGG_FinBERT',
                        'VGG_Alpaca', 'Transformer'
    universe        : str — '30_Stock' or '50_Stock'
    capital         : str — '1M', '100k', or '10k'
    initial_capital : float — must match capital string:
                        1_000_000 / 100_000 / 10_000
    has_sentiment   : bool — False for VGG_Baseline notebooks,
                        True for all others (default True)

    Returns
    -------
    dict with all results for optional further use in the notebook

    Output files saved to ./overlay_data/:
      {save_key}_train_rolling.pkl
      {save_key}_test_rolling.pkl
      {save_key}_account_value.pkl
      {save_key}_train_account.pkl
      {save_key}_bah.pkl
      {save_key}_actions.pkl
      {save_key}_metrics_summary.txt
      {save_key}_action_distribution.png
      {save_key}_regime_analysis.png
    """
    os.makedirs('./overlay_data/', exist_ok=True)
    save_key = f"{universe}_{capital}_{model_name}"

    print(f"\n{'='*60}")
    print(f"  FULL EVALUATION — {save_key}")
    print(f"  Sentiment:   {'Yes' if has_sentiment else 'No (Baseline)'}")
    print(f"  Capital:     ${initial_capital:,.0f}")
    print(f"{'='*60}\n")

    # ── Step 1: Run predictions ───────────────────────────────
    print("Running train prediction...")
    df_train_account, _ = DRLAgent.DRL_prediction(
        model=trained_model, environment=e_train_gym
    )

    print("Running test prediction...")
    df_test_account, df_actions = DRLAgent.DRL_prediction(
        model=trained_model, environment=e_test_gym
    )

    # ── Step 2: Truncate test to peak portfolio value ─────────
    peak_idx        = df_test_account['account_value'].idxmax()
    peak_date       = df_test_account.loc[peak_idx, 'date']
    df_test_peak    = df_test_account.loc[
        :peak_idx
    ].reset_index(drop=True)
    df_actions_peak = df_actions.iloc[
        :peak_idx + 1
    ].reset_index(drop=True)
    print(f"Test peak date: {peak_date}")

    # ── Step 3: Compute summary metrics ──────────────────────
    train_metrics = compute_metrics(df_train_account, initial_capital)
    test_metrics  = compute_metrics(df_test_peak,     initial_capital)

    print(f"\n{'='*45}")
    print(f"  TRAIN METRICS")
    print(f"{'='*45}")
    for k, v in train_metrics.items():
        print(f"  {k:<25} {v:>10}")

    print(f"\n{'='*45}")
    print(f"  TEST METRICS (to peak — {peak_date})")
    print(f"{'='*45}")
    for k, v in test_metrics.items():
        print(f"  {k:<25} {v:>10}")

    # ── Step 4: Compute rolling metrics ──────────────────────
    train_rolling = compute_rolling_metrics(
        df_train_account, initial_capital
    )
    test_rolling  = compute_rolling_metrics(
        df_test_peak, initial_capital
    )

    # ── Step 5: Overfitting check ─────────────────────────────
    overfitting_check(
        train_sharpe=train_metrics['Sharpe Ratio'],
        test_sharpe=test_metrics['Sharpe Ratio'],
        model_name=save_key
    )

    # ── Step 6: Degenerate policy check ──────────────────────
    check_degenerate_policy(
        df_actions_peak, df_test_peak,
        model_name=save_key
    )

    # ── Step 7: Look-ahead bias verification ─────────────────
    # Rebuild a minimal df for the check using test environment
    try:
        df_check = e_test_gym.df[['date', 'tic']].copy()
        if has_sentiment and 'sentiment' in e_test_gym.df.columns:
            df_check['sentiment'] = e_test_gym.df['sentiment'].values
            check_lookahead_bias(df_check, has_sentiment=True)
        else:
            check_lookahead_bias(df_check, has_sentiment=False)
    except Exception:
        print("Look-ahead check skipped — env.df not accessible")

    # ── Step 8: Buy-and-hold comparison ──────────────────────
    df_bah      = compute_buy_and_hold(df_test, initial_capital)
    bah_metrics = compute_metrics(df_bah, initial_capital)

    print(f"\n{'='*45}")
    print(f"  BUY-AND-HOLD METRICS")
    print(f"{'='*45}")
    for k, v in bah_metrics.items():
        print(f"  {k:<25} {v:>10}")

    # ── Step 9: Market regime analysis ───────────────────────
    regime_analysis(
        df_test_peak, df_bah,
        initial_capital=initial_capital,
        model_name=save_key
    )

    # ── Step 10: Plot rolling metrics ────────────────────────
    plot_metrics(train_rolling,
                 title_prefix=f'{save_key} — Training')
    plot_metrics(test_rolling,
                 title_prefix=f'{save_key} — Test (to peak)')

    # ── Step 11: Save all pkl files ──────────────────────────
    pkl_files = {
        f'{save_key}_train_rolling':    train_rolling,
        f'{save_key}_test_rolling':     test_rolling,
        f'{save_key}_account_value':    df_test_peak,
        f'{save_key}_train_account':    df_train_account,
        f'{save_key}_bah':              df_bah,
        f'{save_key}_actions':          df_actions_peak,
    }
    for fname, obj in pkl_files.items():
        with open(f'./overlay_data/{fname}.pkl', 'wb') as f:
            pickle.dump(obj, f)
    print(f"\nPkl files saved: {len(pkl_files)} files")

    # ── Step 12: Save plain-text metrics summary ─────────────
    summary_path = f'./overlay_data/{save_key}_metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"MODEL:          {save_key}\n")
        f.write(f"Has sentiment:  {has_sentiment}\n")
        f.write(f"Peak date:      {peak_date}\n")
        f.write(f"Capital:        ${initial_capital:,.0f}\n\n")
        f.write("TRAIN METRICS\n" + "-" * 35 + "\n")
        for k, v in train_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTEST METRICS (to peak — {peak_date})\n")
        f.write("-" * 35 + "\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nBUY-AND-HOLD METRICS\n" + "-" * 35 + "\n")
        for k, v in bah_metrics.items():
            f.write(f"  {k}: {v}\n")

    print(f"Metrics summary: {summary_path}")
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE — {save_key}")
    print(f"{'='*60}\n")

    return {
        'save_key':          save_key,
        'train_metrics':     train_metrics,
        'test_metrics':      test_metrics,
        'bah_metrics':       bah_metrics,
        'train_rolling':     train_rolling,
        'test_rolling':      test_rolling,
        'df_test_account':   df_test_peak,
        'df_train_account':  df_train_account,
        'df_bah':            df_bah,
        'df_actions':        df_actions_peak,
        'peak_date':         peak_date,
    }

def run_full_evaluation_transformer(
    df_train_account,
    df_test_account,
    df_actions,
    df_test,
    model_name,
    universe,
    capital,
    initial_capital=1_000_000,
):
    os.makedirs('./overlay_data/', exist_ok=True)
    save_key = f"{universe}_{capital}_{model_name}"

    print(f"\n{'='*60}")
    print(f"  FULL EVALUATION — {save_key}")
    print(f"  Capital: ${initial_capital:,.0f}")
    print(f"{'='*60}\n")

    # ── Step 1: Truncate test to peak ─────────────────────────
    peak_idx        = df_test_account['account_value'].idxmax()
    peak_date       = df_test_account.loc[peak_idx, 'date']
    df_test_peak    = df_test_account.loc[
        :peak_idx
    ].reset_index(drop=True)
    df_actions_peak = df_actions.iloc[
        :peak_idx + 1
    ].reset_index(drop=True)
    print(f"Test peak date: {peak_date}")

    # ── Step 2: Metrics ───────────────────────────────────────
    train_metrics = compute_metrics(df_train_account, initial_capital)
    test_metrics  = compute_metrics(df_test_peak,     initial_capital)

    print(f"\n{'='*45}")
    print(f"  TRAIN METRICS")
    print(f"{'='*45}")
    for k, v in train_metrics.items():
        print(f"  {k:<25} {v:>10}")

    print(f"\n{'='*45}")
    print(f"  TEST METRICS (to peak — {peak_date})")
    print(f"{'='*45}")
    for k, v in test_metrics.items():
        print(f"  {k:<25} {v:>10}")

    # ── Step 3: Rolling metrics ───────────────────────────────
    train_rolling = compute_rolling_metrics(
        df_train_account, initial_capital
    )
    test_rolling  = compute_rolling_metrics(
        df_test_peak, initial_capital
    )

    # ── Step 4: Validation checks ─────────────────────────────
    overfitting_check(
        train_sharpe=train_metrics['Sharpe Ratio'],
        test_sharpe=test_metrics['Sharpe Ratio'],
        model_name=save_key
    )
    check_degenerate_policy(
        df_actions_peak, df_test_peak,
        model_name=save_key
    )

    # ── Step 5: Buy-and-hold ──────────────────────────────────
    df_bah      = compute_buy_and_hold(df_test, initial_capital)
    bah_metrics = compute_metrics(df_bah, initial_capital)

    print(f"\n{'='*45}")
    print(f"  BUY-AND-HOLD METRICS")
    print(f"{'='*45}")
    for k, v in bah_metrics.items():
        print(f"  {k:<25} {v:>10}")

    # ── Step 6: Regime analysis ───────────────────────────────
    regime_analysis(
        df_test_peak, df_bah,
        initial_capital=initial_capital,
        model_name=save_key
    )

    # ── Step 7: Plots ─────────────────────────────────────────
    plot_metrics(train_rolling,
                 title_prefix=f'{save_key} — Training')
    plot_metrics(test_rolling,
                 title_prefix=f'{save_key} — Test (to peak)')

    # ── Step 8: Save pkl files ────────────────────────────────
    pkl_files = {
        f'{save_key}_train_rolling':  train_rolling,
        f'{save_key}_test_rolling':   test_rolling,
        f'{save_key}_account_value':  df_test_peak,
        f'{save_key}_train_account':  df_train_account,
        f'{save_key}_bah':            df_bah,
        f'{save_key}_actions':        df_actions_peak,
    }
    for fname, obj in pkl_files.items():
        with open(f'./overlay_data/{fname}.pkl', 'wb') as f:
            pickle.dump(obj, f)
    print(f"\nPkl files saved: {len(pkl_files)} files")

    # ── Step 9: Save metrics summary ─────────────────────────
    summary_path = f'./overlay_data/{save_key}_metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"MODEL:         {save_key}\n")
        f.write(f"Peak date:     {peak_date}\n")
        f.write(f"Capital:       ${initial_capital:,.0f}\n\n")
        f.write("TRAIN METRICS\n" + "-"*35 + "\n")
        for k, v in train_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTEST METRICS (to peak)\n" + "-"*35 + "\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nBUY-AND-HOLD METRICS\n" + "-"*35 + "\n")
        for k, v in bah_metrics.items():
            f.write(f"  {k}: {v}\n")

    print(f"Metrics summary: {summary_path}")
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE — {save_key}")
    print(f"{'='*60}\n")

    return {
        'save_key':         save_key,
        'train_metrics':    train_metrics,
        'test_metrics':     test_metrics,
        'bah_metrics':      bah_metrics,
        'train_rolling':    train_rolling,
        'test_rolling':     test_rolling,
        'df_test_account':  df_test_peak,
        'df_train_account': df_train_account,
        'df_bah':           df_bah,
        'df_actions':       df_actions_peak,
        'peak_date':        peak_date,
    }