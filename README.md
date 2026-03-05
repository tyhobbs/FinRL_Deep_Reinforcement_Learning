# FinRL: Deep Reinforcement Learning for Automated Stock Trading

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

Detailed metrics for all completed models are available in [Metrics/METRICS.txt](Metrics/METRICS.txt).

### Status by Universe and Capital Level

| Universe | Capital | VGG Baseline | VGG + FinBERT | VGG + Alpaca | Transformer |
|----------|---------|-------------|--------------|-------------|-------------|
| 30-Stock | $1,000,000 | ✅ Complete | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 
| 30-Stock | $100,000 | ✅ Complete | 🔄 In Progress  | 🔄 In Progress  | 🔄 In Progress | 
| 30-Stock | $10,000 | ✅ Complete | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress |
| 50-Stock | $1,000,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 
| 50-Stock | $100,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 
| 50-Stock | $10,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 

### Non-DRL Benchmark

| Benchmark | Capital Level | Universe | Status |
|-----------|---------|--------|--------|
| Buy-and-Hold (equal-weight) | $1,000,000 | 30-Stock | 🔄 In Progress | 
| Buy-and-Hold (equal-weight) | $1,000,000 | 50-Stock | 🔄 In Progress | 

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

Reducing starting capital forces the model to be selective. Capital scaling is evaluated on the Transformer model only as it is the best-performing architecture.

| Capital | hmax per stock | Purpose |
|---------|---------------|---------|
| $1,000,000 | 100 shares | Institutional benchmark |
| $100,000 | 10 shares | Realistic retail trader |
| $10,000 | 5 shares | Small retail / maximum selectivity |

---

## Goal

To demonstrate that combining cross-stock transformer architecture, live market data, full historical sentiment from Polygon.io, and realistic capital constraints produces measurably higher risk-adjusted returns compared to a standard VGG baseline and a buy-and-hold benchmark - and to explore how performance scales across institutional and retail capital levels through a clean single-variable ablation study.

---

## Installation

### Prerequisites

- Python 3.8+
- A [Polygon.io](https://polygon.io) account and API key (Stocks Starter plan or above recommended for full historical news coverage)
- An [Alpaca](https://alpaca.markets/) account and API key (for Alpaca-based models)

### Clone the Repository
```bash
git clone https://github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning.git
cd FinRL_Deep_Reinforcement_Learning
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install the core packages manually:
```bash
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

Create a `.env` file in the root directory and add your API credentials:
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
POLYGON_API_KEY=your_polygon_api_key

### Download the Data

Historical price data is fetched automatically when running each notebook - via Yahoo Finance for baseline models and via Alpaca API for all other models. Polygon.io sentiment data is fetched and cached locally after the first run, so subsequent executions load from cache instantly.

```

## Repository Structure

FinRL_Deep_Reinforcement_Learning/
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
+-- README.md
```

## Future Work

- **100-Stock Universe**: Extend the Cross-Stock Transformer to a 100-stock universe spanning all major S&P 500 sectors, evaluating cross-stock attention mechanisms at institutional scale
- **Live deployment**: Implement a kill-switch trading system for live Alpaca paper trading with drawdown-based stop conditions
- **Walk-forward validation**: Extend evaluation using rolling train/test windows to test generalization across different market regimes
- **Additional DRL algorithms**: Compare PPO against SAC and TD3 on the same universe and reward structure
