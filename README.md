# FinRL: Deep Reinforcement Learning for Automated Stock Trading

[![Results Dashboard](https://img.shields.io/badge/Results-GitHub_Pages-blue?logo=github)](https://tyhobbs.github.io/FinRL_Deep_Reinforcement_Learning/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This project integrates **Deep Reinforcement Learning (DRL)** with **LLM-driven sentiment analysis** to build an automated stock trading system using the [FinRL](https://github.com/AI4Finance-Foundation/FinRL) framework. Models are benchmarked across multiple architectures, data sources, stock universes, and starting capital levels to evaluate both institutional and retail-scale performance.

---

## Overview

We benchmark and compare DRL trading agents trained on **30 and 50 stocks across 9 sectors (2020-2025)**, progressively enhancing each model with live market data, NLP-based news sentiment, an expanded stock universe, and realistic capital constraints. Each variable is changed independently to isolate its contribution, forming a proper ablation study across architecture, data source, sentiment quality, universe size, and capital level.

The ablation follows a clean progressive structure:

| Step | Comparison | Variable Isolated |
|------|-----------|------------------|
| 1 | VGG Baseline to VGG + FinBERT | Effect of sentiment |
| 2 | VGG + FinBERT (Yahoo) to VGG + FinBERT(Polygon) | Effect of live data |
| 3 | VGG + FinBERT(Polygon) to VGG + Alpaca | Effect of architecture |
| 4 | Cross-Stock Transformer + FinBERT(Polygon) + Alpaca | Effect of Transformer |
| 5 | 1M to 100k to 10k and 30 stock and 50 stock | Effect of universe size and stock sizes |
| 6 | Best DRL model vs Buy-and-Hold | Justification for DRL complexity |

---

## Model Reference

| Model | Data Source | Sentiment | Architecture |
|-------|-------------|-----------|--------------|
| VGG Baseline | Yahoo Finance | None | VGG + PPO |
| VGG + FinBERT | Yahoo Finance | Polygon + FinBERT | VGG + PPO |
| VGG + FinBERT + Alpaca | Alpaca | Polygon + FinBERT | VGG + PPO |
| Transformer + FinBERT + Alpaca | Alpaca | Polygon + FinBERT | Transformer + PPO |

---

## Results

[![Results Dashboard](https://img.shields.io/badge/View_Interactive_Dashboard-GitHub_Pages-blue?logo=github)](https://tyhobbs.github.io/FinRL_Deep_Reinforcement_Learning/)

Detailed metrics for all completed models are available in [Metrics/METRICS.txt](Metrics/METRICS.txt).

### Key Findings

- **18 of 24 models beat buy-and-hold** on a risk-adjusted basis
- **$100k is the optimal capital level** — highest average test Sharpe (2.035) across all architectures
- **VGG + Alpaca outperforms the Transformer** (avg Sharpe 2.089 vs 1.568) — local convolutional feature extraction is more data-efficient than global attention for daily trading
- **30-stock universe outperforms 50-stock** on average (1.987 vs 1.629)
- **Best single model**: 30-Stock VGG Baseline $10k — Test Sharpe **3.111**, Return **+98.70%**


### Summary Results — All 24 Models (ranked by Test Sharpe)

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

*All Sharpe ratios computed with 5% annualised risk-free rate. Returns reported to peak portfolio value. Transaction costs: 0.15% per trade (0.1% commission + 0.05% slippage).*

### By Architecture (Average Test Sharpe)

| Architecture | Avg Sharpe | Best | Worst |
|-------------|-----------|------|-------|
| VGG + Alpaca | **2.089** | 2.726 | 1.570 |
| VGG Baseline | 1.939 | 3.111 | 1.001 |
| VGG + FinBERT | 1.892 | 2.350 | 1.273 |
| Transformer | 1.568 | 1.895 | 0.629 |

### By Capital Level (Average Test Sharpe)

| Capital | Avg Sharpe | Best | Worst |
|--------|-----------|------|-------|
| $100k | **2.035** | 2.531 | 1.381 |
| $10k | 1.865 | 3.111 | 0.629 |
| $1M | 1.716 | 2.350 | 1.001 |

### Non-DRL Benchmark

| Benchmark | Universe | Sharpe | Return | Max DD |
|-----------|---------|--------|--------|--------|
| Buy-and-Hold (equal-weight) | 30-Stock | 1.974 | 39.34% | -10.82% |
| Buy-and-Hold (equal-weight) | 50-Stock | 1.509 | 25.14% | -7.06% |

### Completion Status

| Universe | Capital | VGG Baseline | VGG + FinBERT | VGG + Alpaca | Transformer |
|----------|---------|-------------|--------------|-------------|-------------|
| 30-Stock | $1,000,000 | ✅ | ✅ | ✅ | ✅ |
| 30-Stock | $100,000 | ✅ | ✅ | ✅ | ✅ |
| 30-Stock | $10,000 | ✅ | ✅ | ✅ | ✅ |
| 50-Stock | $1,000,000 | ✅ | ✅ | ✅ | ✅ |
| 50-Stock | $100,000 | ✅ | ✅ | ✅ | ✅ |
| 50-Stock | $10,000 | ✅ | ✅ | ✅ | ✅ |

---

## Key Features

- **Data Sources**: Historical data via Yahoo Finance (baseline models) and live data via Alpaca API (all other models)
- **Sentiment Analysis**: Full historical financial news (2020-present) via [Polygon.io](https://polygon.io), scored by [FinBERT](https://huggingface.co/ProsusAI/finbert) (+1 bullish / -1 bearish), merged directly into training features
- **Macro Features**: VIX, 10-year treasury yield (TNX), SPY, QQQ, and XLK sector ETF returns as additional state features in transformer models
- **Technical Indicators**: MACD, Bollinger Bands, RSI, CCI, DX, SMA (30/60-day)
- **Architectures**: VGG-style CNN and Cross-Stock Transformer with two-stage attention (temporal per stock + cross-stock)
- **RL Algorithm**: Proximal Policy Optimization (PPO) via Stable Baselines 3
- **Training Improvements**: Linear learning rate schedule, VecNormalize observation normalisation, TrainSharpeSavingCallback peak weight preservation, CheckpointCallback periodic saves
- **Reward Shaping**: Simplified four-component reward - Sharpe reward, single drawdown penalty, sentiment reward/penalty, 15% per-stock concentration limit
- **Evaluation Metrics**: Sharpe Ratio, Win Rate, Maximum Drawdown, Calmar Ratio, Total Return, Volatility
- **Non-DRL Benchmark**: Equal-weight buy-and-hold portfolio as a reference baseline

---

## Ablation Study Design

Each model changes exactly one variable relative to the previous, allowing clean attribution of performance gains:
VGG Baseline (Yahoo Finance, no sentiment)
|
|  + Polygon sentiment via FinBERT
v
VGG + FinBERT (Yahoo Finance + sentiment)
|
|  + Live Alpaca data source
v
VGG + FinBERT + Alpaca (Alpaca + sentiment)
|
|  + Cross-Stock Transformer architecture
v
Transformer + FinBERT + Alpaca (best DRL model)
|
|  Capital level scaling (50-stock universe)
|---> $1,000,000 (institutional benchmark)
|---> $100,000   (realistic retail trader)
'---> $10,000    (small retail / max selectivity)
The best DRL model is then compared against a buy-and-hold benchmark to justify the complexity of the DRL approach.

---

## Stock Universes

### 30-Stock Universe

30 stocks drawn from NASDAQ-listed technology, healthcare, and consumer companies.

| Sector | Tickers |
|--------|---------|
| Technology | AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, PYPL, ADBE, NFLX |
| Semiconductors | INTC, CSCO, AVGO, QCOM, TXN, ADI |
| Software / Services | INTU, ADP, BKNG |
| Consumer Discretionary | PEP, COST, SBUX, MDLZ |
| Healthcare / Biotech | AMGN, GILD, ISRG, VRTX |
| Industrials | HON |
| Communication Services | CMCSA, TMUS |

---

### 50-Stock Universe

Core 30-stock universe expanded with 20 additional stocks across underrepresented sectors.

| Sector | Tickers |
|--------|---------|
| Technology | AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, PYPL, ADBE, NFLX |
| Semiconductors | INTC, CSCO, AVGO, QCOM, TXN, ADI |
| Software / Services | INTU, ADP, BKNG |
| Consumer Discretionary | PEP, COST, SBUX, MDLZ |
| Healthcare / Biotech | AMGN, GILD, ISRG, VRTX, JNJ, UNH, PFE, MRK, ABT |
| Industrials | HON, CAT, GE, RTX |
| Communication Services | CMCSA, TMUS |
| Financials | JPM, BAC, GS, MS, BLK |
| Energy | XOM, CVX, COP |
| Consumer Staples | WMT, PG, KO, MCD |
| Real Estate / Utilities | AMT, NEE, DUK |
| Materials | LIN, NEM, FCX |
| Growth / Emerging Tech | AMD, CRM, SNOW, PLTR |

---

## Dataset

- **Training**: 2020-01-01 to 2024-01-01
- **Validation**: 2024-01-01 to 2025-01-01
- **Sentiment**: Full historical coverage via Polygon.io (2020-present) - no 30-day limitation
- **Macro features**: VIX, TNX, SPY, QQQ, XLK daily returns (transformer models only)
- **Risk-free rate**: 5% annualised (3-month T-bill) used in Sharpe ratio calculation

---

## Capital Levels

Reducing starting capital forces the model to be selective. Capital scaling is evaluated across all architectures. $100k emerged as the optimal capital level with the highest average test Sharpe across all model types.

| Capital | hmax per stock | Purpose |
|---------|---------------|---------|
| $1,000,000 | 100 shares | Institutional benchmark |
| $100,000 | 10 shares | Realistic retail trader |
| $10,000 | 5 shares | Small retail / maximum selectivity |

---

## Goal

To evaluate whether combining cross-stock transformer architecture, live market data, full historical sentiment from Polygon.io, and realistic capital constraints produces measurably higher risk-adjusted returns compared to a VGG baseline and a buy-and-hold benchmark. Results show that capital constraint level is the dominant factor in performance — $100k models achieve the highest average Sharpe (2.035) — and that VGG-based architectures outperform the Cross-Stock Transformer on this dataset, suggesting local convolutional feature extraction is a more data-efficient inductive bias than global attention for daily stock trading with 4 years of training data.

---

## Installation

### Prerequisites

- Python 3.8+
- A [Polygon.io](https://polygon.io) account and API key (Stocks Starter plan or above recommended for full historical news coverage)
- An [Alpaca](https://alpaca.markets/) account and API key (for Alpaca-based models)

### Clone the Repository
```
bash
git clone https://github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning.git
cd FinRL_Deep_Reinforcement_Learning
```

### Install Dependencies
```
bash
pip install -r requirements.txt
```

Or install the core packages manually:
```
bash
pip install finrl
pip install stable-baselines3
pip install yfinance
pip install alpaca-py
pip install transformers
pip install torch
pip install polygon-api-client
pip install gymnasium
pip install pandas numpy matplotlib plotly
```

### Configure API Keys
```
Create a `.env` file in the root directory and add your API credentials:
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
POLYGON_API_KEY=your_polygon_api_key
```

### Download the Data

Historical price data is fetched automatically when running each notebook - via Yahoo Finance for baseline models and via Alpaca API for all other models. Polygon.io sentiment data is fetched and cached locally after the first run, so subsequent executions load from cache instantly.



## Repository Structure
```
FinRL_Deep_Reinforcement_Learning/
|
|--Environment/
|   |--requirements.txt
|   |--utils.py
|
|-- Code/
|   |-- 30-Stock Universe
|   |   |-- testFinRL.ipynb
|   |   |-- testFinRLwithNewsFetch_polygon.ipynb
|   |   |-- FinRL_Alpaca_polygon.ipynb
|   |   +-- FinRL_transformer_polygon.ipynb
|   |
|   +-- 50-Stock Universe
|       |-- BaselineVGG_50stocks_{1M,100k,10k}.ipynb
|       |-- VGG_yfinance_polygon_50stocks_{1M,100k,10k}.ipynb
|       |-- VGG_Alpaca_polygon_50stocks_{1M,100k,10k}.ipynb
|       +-- FinRL_transformer_polygon_50stocks_{1M,100k,10k}.ipynb
|
|-- Metrics/
|   +-- METRICS.txt
|
|-- Data/
|   +-- Training and validation CSV files per model
|
|-- Literature Paper/
|   +-- Reference papers
|
|-- docs/
|   |-- index.html               ← GitHub Pages results dashboard
|   |-- *.png                    ← Performance charts
|
|
+-- README.md
```

## Future Work

-- **Hyperparameter search**: Systematic grid search over PPO hyperparameters and reward component weights to improve Transformer performance at smaller capital levels
- **100-Stock Universe**: Extend the Cross-Stock Transformer to a 100-stock universe spanning all major S&P 500 sectors, evaluating cross-stock attention mechanisms at institutional scale
- **Live deployment**: Implement a kill-switch trading system for live Alpaca paper trading with drawdown-based stop conditions
- **Walk-forward validation**: Extend evaluation using rolling train/test windows to test generalization across different market regimes
- **Additional DRL algorithms**: Compare PPO against SAC and TD3 on the same universe and reward structure
