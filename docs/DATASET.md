# Dataset

---

## Time Periods

| Split | Start | End | Purpose |
|-------|-------|-----|---------|
| Training | 2020-01-01 | 2024-01-01 | Model training (4 years) |
| Validation | 2024-01-01 | 2025-01-01 | Out-of-sample evaluation (1 year) |

The training period (2020–2024) covers multiple market regimes including the COVID crash and recovery, the 2022 bear market, and the 2023 bull market. The validation period (2024) covers the S&P 500's strong H1 rally (+15%) and more volatile H2.

---

## Price Data

| Source | Models | Coverage |
|--------|--------|----------|
| Yahoo Finance | VGG Baseline, VGG + FinBERT | Daily OHLCV, 2020–2025 |
| Alpaca API | VGG + Alpaca, Transformer | Daily OHLCV, 2020–2025, adjustment='all' |

All price data uses split and dividend-adjusted closing prices. The Alpaca API is used for all non-baseline models as it provides more reliable institutional-grade data and enables live paper trading without a data source switch at deployment.

---

## Technical Indicators

Eight technical indicators are computed per stock per day using the FinRL `FeatureEngineer` pipeline (backed by `stockstats`):

| Indicator | Description |
|-----------|-------------|
| `macd` | Moving Average Convergence Divergence |
| `boll_ub` | Bollinger Band upper bound |
| `boll_lb` | Bollinger Band lower bound |
| `rsi_30` | Relative Strength Index (30-day) |
| `cci_30` | Commodity Channel Index (30-day) |
| `dx_30` | Directional Movement Index (30-day) |
| `close_30_sma` | 30-day Simple Moving Average |
| `close_60_sma` | 60-day Simple Moving Average |

---

## Sentiment Data

| Source | Coverage | Method |
|--------|----------|--------|
| Polygon.io | Full historical 2020–2025 | REST API, up to 1000 articles per ticker per period |
| FinBERT | Per-headline scoring | ProsusAI/finbert, scores: +1 bullish / -1 bearish |

Sentiment scores are computed as the mean FinBERT score across all headlines for a given ticker on a given day. A **one-day lag** is applied to all sentiment scores to eliminate look-ahead bias — today's model decision uses yesterday's news.

Sentiment coverage across training data:
- Mean score: ~0.02 (slightly bullish)
- Standard deviation: ~0.31
- Approximately 60% of ticker-day pairs have non-zero sentiment

---

## Macro Features

Macro features are included in the Transformer models only as additional state observations:

| Feature | Description |
|---------|-------------|
| VIX | CBOE Volatility Index daily return |
| TNX | 10-year Treasury yield daily return |
| SPY | S&P 500 ETF daily return |
| QQQ | NASDAQ-100 ETF daily return |
| XLK | Technology sector ETF daily return |

---

## Observation Space

The full observation vector per trading day is structured as:

```
[cash, prices(N), shares_held(N), indicators(N × K)]
```

Where:
- `N` = number of stocks (30 or 50)
- `K` = number of indicators (9 for VGG models including sentiment, 14 for Transformer models including sentiment + macro)
- VGG state space (30-stock): `1 + 30 + 30 + 30×9 = 331`
- Transformer state space (30-stock): `(1 + 30 + 30 + 30×14) × 10 = 4,810` (10-day lookback window)

---

## Transaction Costs

All models use realistic transaction costs:

| Component | Rate |
|-----------|------|
| Commission | 0.10% per trade |
| Slippage | 0.05% per trade |
| **Total** | **0.15% per trade** |

---

*Back to [README](../README.md)*
