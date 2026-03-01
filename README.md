# FinRL: Deep Reinforcement Learning for Automated Stock Trading

This project integrates **Deep Reinforcement Learning (DRL)** with **LLM-driven sentiment analysis** to build an automated stock trading system using the [FinRL](https://github.com/AI4Finance-Foundation/FinRL) framework. Models are benchmarked across multiple architectures, data sources, stock universes, and starting capital levels to evaluate both institutional and retail-scale performance.

---

## Overview

We benchmark and compare DRL trading agents trained on up to **100 stocks across 12 sectors (2020–2025)**, progressively enhancing each model with live market data, NLP-based news sentiment, expanded stock universes, and realistic capital constraints. Each variable is changed independently to isolate its contribution — forming a proper ablation study across architecture, data source, sentiment quality, universe size, and capital level.

---

## Model Reference

| Model | Data Source | Sentiment | Architecture |
|-------|-------------|-----------|--------------|
| VGG Baseline | Yahoo Finance | ❌ None | VGG + PPO |
| VGG + FinBERT | Yahoo Finance | ✅ Polygon + FinBERT | VGG + PPO |
| VGG + FinBERT + Alpaca | Alpaca | ✅ Polygon + FinBERT | VGG + PPO |
| Transformer + FinBERT + Alpaca | Alpaca | ✅ Polygon + FinBERT | Transformer + PPO |

---

## Results

Detailed metrics for all completed models are available in [`Results/METRICS.txt`](Results/METRICS.txt).

### Status by Universe and Capital Level

| Universe | Capital | VGG Baseline | VGG + FinBERT | VGG + Alpaca | Transformer |
|----------|---------|-------------|--------------|-------------|-------------|
| 30-Stock | $1,000,000 | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| 30-Stock | $100,000 | ✅ Complete | 🔄 In Progress | 🔄 In Progress | ✅ Complete |
| 30-Stock | $10,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | ✅ Complete |
| 50-Stock | $1,000,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | ✅ Complete |
| 50-Stock | $100,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | ✅ Complete |
| 50-Stock | $10,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | ✅ Complete |
| 100-Stock | $1,000,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress |
| 100-Stock | $100,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress |
| 100-Stock | $10,000 | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress | 🔄 In Progress |

---

## Key Features

- **Data Sources**: Historical data via Yahoo Finance (baseline models) and live data via Alpaca API (all other models)
- **Sentiment Analysis**: Full historical financial news (2020–present) via [Polygon.io](https://polygon.io), scored by [FinBERT](https://huggingface.co/ProsusAI/finbert) (+1 bullish / −1 bearish), merged directly into training features
- **Sentiment Gating**: The transformer model uses sentiment as a hard action gate — buying is blocked on negative sentiment, sell signals are amplified on bad news, and holding stocks with very negative sentiment is penalised
- **Macro Features**: VIX, 10-year treasury yield (TNX), SPY, QQQ, and XLK sector ETF returns as additional state features in transformer models
- **Technical Indicators**: MACD, Bollinger Bands, RSI, CCI, DX, SMA (30/60-day)
- **Architectures**: VGG-style CNN and Cross-Stock Transformer with two-stage attention (temporal per stock + cross-stock)
- **RL Algorithm**: Proximal Policy Optimization (PPO) via Stable Baselines 3
- **Training Improvements**: Linear learning rate schedule, VecNormalize observation normalisation, TrainSharpeSavingCallback peak weight preservation, CheckpointCallback periodic saves
- **Reward Shaping**: Direct annualised Sharpe reward, drawdown penalty, sentiment alignment reward and penalty, 15% per-stock concentration limit
- **Evaluation Metrics**: Sharpe Ratio, Win Rate, Maximum Drawdown, Calmar Ratio, Total Return, Peak Return, Volatility

---

## Stock Universes

### 30-Stock Universe (Original)
`AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, PYPL, ADBE, NFLX, INTC, CSCO, PEP, AVGO, COST, QCOM, CMCSA, TMUS, TXN, AMGN, HON, SBUX, INTU, MDLZ, GILD, ISRG, ADP, BKNG, VRTX, ADI`

### 50-Stock Universe
Core 30 NASDAQ stocks plus:

| Sector | Additions |
|--------|-----------|
| Healthcare | JNJ, UNH, PFE, MRK, ABT |
| Financials | JPM, BAC, GS, MS, BLK |
| Energy | XOM, CVX, COP |
| Consumer Staples | WMT, PG, KO, MCD |
| Industrials / Defence | CAT, GE, RTX |
| Real Estate / Utilities | AMT, NEE, DUK |
| Materials | LIN, NEM, FCX |
| Growth / Emerging Tech | AMD, CRM, SNOW, PLTR |

### 100-Stock Universe
50-stock universe expanded with additional coverage across all major S&P 500 sectors including energy majors (SLB, EOG, PSX, MPC, VLO), additional financials (WFC, C, AXP, SPGI, CME), biotech (REGN, BIIB, ILMN, IDXX, ALGN, MRNA), communication services (DIS, CHTR, T, VZ), real estate/utilities (PLD, SO), materials (APD, ECL, FCX), and emerging tech (COIN).

---

## Dataset

- **Training**: 2020-01-01 → 2024-01-01
- **Validation**: 2024-01-01 → 2025-01-01
- **Sentiment**: Full historical coverage via Polygon.io (2020–present) — no 30-day limitation
- **Macro features**: VIX, TNX, SPY, QQQ, XLK daily returns (transformer models only)

---

## Capital Levels

Reducing starting capital forces the model to be selective — it cannot hold meaningful positions across the entire stock universe simultaneously, encouraging concentration in the highest-conviction trades.

| Capital | hmax per stock | Purpose |
|---------|---------------|---------|
| $1,000,000 | 100 shares | Institutional benchmark |
| $100,000 | 10 shares | Realistic retail trader |
| $10,000 | 5 shares | Small retail / maximum selectivity |

---

## Goal

To demonstrate that combining cross-stock transformer architecture, live market data, full historical sentiment from Polygon.io, sector-diversified stock universes, and realistic capital constraints produces measurably higher risk-adjusted returns compared to a standard VGG baseline — and to explore how model performance scales across institutional and retail capital levels.

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

```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
POLYGON_API_KEY=your_polygon_api_key
```

### Download the Data

Historical price data is fetched automatically when running each notebook — via Yahoo Finance for baseline models and via Alpaca API for all other models. Polygon.io sentiment data is fetched and cached locally after the first run, so subsequent executions load from cache instantly.

---

## Repository Structure

```
FinRL_Deep_Reinforcement_Learning/
├── Code/
│   ├── 30-Stock Universe
│   │   ├── testFinRL.ipynb                              # VGG Baseline (Yahoo Finance, no sentiment)
│   │   ├── testFinRLwithNewsFetch_polygon.ipynb         # VGG + Polygon FinBERT (Yahoo Finance)
│   │   ├── FinRL_Alpaca_polygon.ipynb                   # VGG + Polygon FinBERT (Alpaca)
│   │   └── FinRL_transformer_polygon.ipynb              # Transformer + Polygon FinBERT (Alpaca)
│   │
│   ├── 50-Stock Universe
│   │   ├── BaselineVGG_50stocks_{1M,100k,10k}.ipynb            # Yahoo Finance, no sentiment
│   │   ├── VGG_yfinance_polygon_50stocks_{1M,100k,10k}.ipynb   # Yahoo Finance + Polygon FinBERT
│   │   ├── VGG_Alpaca_polygon_50stocks_{1M,100k,10k}.ipynb     # Alpaca + Polygon FinBERT
│   │   └── FinRL_transformer_polygon_50stocks.ipynb             # Transformer (Alpaca + Polygon)
│   │
│   └── 100-Stock Universe
│       ├── BaselineVGG_{1M,100k,10k}.ipynb                     # Yahoo Finance, no sentiment
│       ├── VGG_yfinance_polygon_{1M,100k,10k}.ipynb            # Yahoo Finance + Polygon FinBERT
│       ├── VGG_Alpaca_polygon_{1M,100k,10k}.ipynb              # Alpaca + Polygon FinBERT
│       └── FinRL_transformer_polygon_100stocks.ipynb            # Transformer (Alpaca + Polygon)
│
├── Results/
│   └── METRICS.txt                                      # Detailed metrics for all completed models
│
├── Data/
│   └── Training and validation CSV files per model
│
├── Literature Paper/
│   └── Reference papers
│
└── README.md
```
