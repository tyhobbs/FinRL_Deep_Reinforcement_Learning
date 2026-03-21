# Results — Full 24-Model Ablation Study

All metrics computed over the 2024 test period with a 5% annualised risk-free rate.
Returns reported to peak portfolio value. Transaction costs: 0.15% per trade (0.1% commission + 0.05% slippage).

---

## Full Results Table (Ranked by Test Sharpe)

| Model | Universe | Capital | Test Sharpe | Test Return | Max DD | vs BAH |
|-------|----------|---------|------------|-------------|--------|--------|
| VGG Baseline | 30-Stock | $10k | **3.111** | 98.70% | -11.27% | ▲ +1.137 |
| VGG + Alpaca | 50-Stock | $10k | 2.726 | 9.51% | -3.71% | ▲ +1.289 |
| VGG + Alpaca | 30-Stock | $100k | 2.531 | 57.16% | -9.50% | ▲ +0.556 |
| VGG + FinBERT | 50-Stock | $100k | 2.347 | 42.47% | -7.02% | ▲ +0.910 |
| VGG + FinBERT | 30-Stock | $100k | 2.350 | 48.92% | -8.95% | ▲ +0.376 |
| VGG Baseline | 30-Stock | $1M | 2.350 | 55.12% | -12.86% | ▲ +0.376 |
| VGG + FinBERT | 30-Stock | $1M | 2.349 | 49.02% | -10.84% | ▲ +0.375 |
| VGG Baseline | 30-Stock | $100k | 2.287 | 116.95% | -19.86% | ▲ +0.313 |
| VGG + Alpaca | 30-Stock | $10k | 2.111 | 123.85% | -20.21% | ▲ +0.136 |
| VGG + Alpaca | 50-Stock | $100k | 2.019 | 72.48% | -15.28% | ▲ +0.582 |
| Transformer | 50-Stock | $100k | 1.895 | 52.93% | -19.62% | ▲ +0.386 |
| Transformer | 50-Stock | $1M | 1.861 | 38.26% | -11.27% | ▲ +0.352 |
| Transformer | 50-Stock | $10k | 1.806 | 33.99% | -8.89% | ▲ +0.297 |
| Transformer | 30-Stock | $1M | 1.748 | 55.23% | -19.70% | ▲ +0.389 |
| VGG + FinBERT | 50-Stock | $10k | 1.695 | 68.48% | -21.66% | ▲ +0.258 |
| VGG + Alpaca | 50-Stock | $1M | 1.575 | 29.96% | -7.78% | ▲ +0.138 |
| VGG + Alpaca | 30-Stock | $1M | 1.570 | 31.41% | -12.52% | ▼ -0.405 |
| VGG Baseline | 50-Stock | $10k | 1.504 | 30.98% | -10.33% | ▲ +0.067 |
| Transformer | 30-Stock | $100k | 1.468 | 25.91% | -11.93% | ▲ +0.109 |
| VGG Baseline | 50-Stock | $100k | 1.381 | 32.11% | -10.75% | ▼ -0.056 |
| VGG + FinBERT | 30-Stock | $10k | 1.338 | 41.07% | -19.24% | ▼ -0.636 |
| VGG + FinBERT | 50-Stock | $1M | 1.273 | 23.61% | -8.42% | ▼ -0.164 |
| VGG Baseline | 50-Stock | $1M | 1.001 | 20.97% | -9.33% | ▼ -0.436 |
| Transformer | 30-Stock | $10k | 0.629 | 13.31% | -12.14% | ▼ -0.730 |

---

## By Architecture (Average Test Sharpe)

| Architecture | Avg Sharpe | Best | Worst |
|-------------|-----------|------|-------|
| VGG + Alpaca | **2.089** | 2.726 | 1.570 |
| VGG Baseline | 1.939 | 3.111 | 1.001 |
| VGG + FinBERT | 1.892 | 2.350 | 1.273 |
| Transformer | 1.568 | 1.895 | 0.629 |

---

## By Capital Level (Average Test Sharpe)

| Capital | Avg Sharpe | Best | Worst |
|--------|-----------|------|-------|
| $100k | **2.035** | 2.531 | 1.381 |
| $10k | 1.865 | 3.111 | 0.629 |
| $1M | 1.716 | 2.350 | 1.001 |

---

## Non-DRL Benchmark

| Benchmark | Universe | Sharpe | Return | Max DD |
|-----------|---------|--------|--------|--------|
| Buy-and-Hold (equal-weight) | 30-Stock | 1.974 | 39.34% | -10.82% |
| Buy-and-Hold (equal-weight) | 50-Stock | 1.509 | 25.14% | -7.06% |

---

## Completion Status

| Universe | Capital | VGG Baseline | VGG + FinBERT | VGG + Alpaca | Transformer |
|----------|---------|-------------|--------------|-------------|-------------|
| 30-Stock | $1,000,000 | ✅ | ✅ | ✅ | ✅ |
| 30-Stock | $100,000 | ✅ | ✅ | ✅ | ✅ |
| 30-Stock | $10,000 | ✅ | ✅ | ✅ | ✅ |
| 50-Stock | $1,000,000 | ✅ | ✅ | ✅ | ✅ |
| 50-Stock | $100,000 | ✅ | ✅ | ✅ | ✅ |
| 50-Stock | $10,000 | ✅ | ✅ | ✅ | ✅ |

---

## Key Findings

- **18 of 24 models beat buy-and-hold** on a risk-adjusted basis
- **$100k is the optimal capital level** — highest average test Sharpe (2.035) across all architectures
- **VGG + Alpaca outperforms the Transformer** (avg Sharpe 2.089 vs 1.568) — local convolutional feature extraction is more data-efficient than global attention for daily trading
- **30-stock universe outperforms 50-stock** on average (1.987 vs 1.629)
- **Best single model**: 30-Stock VGG Baseline $10k — Test Sharpe **3.111**, Return **+98.70%**
- **Paper trading**: The best-performing non-baseline model (30-Stock VGG + Alpaca $100k, Sharpe 2.531) is currently deployed live on Alpaca paper trading (deployed March 16, 2026)

---

## Market Regime Analysis

The 2024 test period splits into H1 (Jan–Jun, S&P 500 +15%) and H2 (Jul–Dec, volatile).

| Period | Best Model Sharpe | Buy-and-Hold Sharpe |
|--------|------------------|---------------------|
| H1 2024 | 3.343 | 3.411 |
| H2 2024 | 1.776 | 0.911 |

DRL adds more value in volatile periods — the model outperforms buy-and-hold more significantly in H2 than H1, confirming that the reward shaping and sentiment features provide meaningful downside protection during market stress.

---

*Full metric details available in [Metrics/METRICS.txt](../Metrics/METRICS.txt)*  
*Interactive dashboard: [GitHub Pages](https://tyhobbs.github.io/FinRL_Deep_Reinforcement_Learning/)*
