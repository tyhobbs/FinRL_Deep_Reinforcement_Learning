# FinRL: Deep Reinforcement Learning for Automated Stock Trading

This project integrates **Deep Reinforcement Learning (DRL)** with **LLM-driven sentiment analysis** to build an automated stock trading system using the [FinRL](https://github.com/AI4Finance-Foundation/FinRL) framework.

## Overview

We benchmark and compare multiple DRL trading agents trained on the **top 30 S&P 500 companies (2020–2025)**, progressively enhancing each model with live market data and NLP-based news sentiment to maximize risk-adjusted returns (Sharpe Ratio).

## Model Comparison

| Model | Data Source | Sentiment | Architecture | Train Sharpe | Test Sharpe |
|-------|-------------|-----------|--------------|--------------|-------------|
| VGG Baseline | Yahoo Finance | ❌ | VGG + PPO | 3.58 | -2.761 |
| VGG + FinBERT | Yahoo Finance | ✅ NewsAPI + FinBERT | VGG + PPO | 9.15 | -4.63 |
| VGG + FinBERT + Alpaca | Alpaca (live) | ✅ NewsAPI + FinBERT | VGG + PPO | 5.76 | -7.31 |
| Transformer + FinBERT + Alpaca | Alpaca (live) | ✅ NewsAPI + FinBERT | Transformer + PPO | In Progress | In Progress |

## Key Features

- **Data**: Historical & live data via Yahoo Finance and Alpaca API
- **Sentiment Analysis**: Financial news headlines scored using [FinBERT](https://huggingface.co/ProsusAI/finbert) (+1 bullish / -1 bearish), merged directly into training features
- **Technical Indicators**: MACD, Bollinger Bands, RSI, CCI, DX, SMA (30/60-day)
- **RL Algorithm**: Proximal Policy Optimization (PPO) via Stable Baselines 3
- **Evaluation Metrics**: Sharpe Ratio, win rate, maximum drawdown

## Dataset

- **Training**: 2020-01-01 → 2024-01-01 (~30,000 time-series records)
- **Validation**: 2024-01-01 → 2025-01-01 (~7,500 time-series records)
- 80/20 train/validation split across 30 tickers

## Tickers

`AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, PYPL, ADBE, NFLX, INTC, CSCO, PEP, AVGO, COST, QCOM, CMCSA, TMUS, TXN, AMGN, HON, SBUX, INTU, MDLZ, GILD, ISRG, ADP, BKNG, VRTX, ADI`

## Goal

To demonstrate that combining transformer-based feature extraction, live market data, and FinBERT sentiment scoring produces a measurably higher Sharpe Ratio and lower drawdown compared to a standard VGG baseline — targeting a Sharpe Ratio in the range of **1.5–3.0**, consistent with top-performing FinRL models like FinRL-Podracer.

---

## Installation

### Prerequisites

- Python 3.8+
- A [NewsAPI](https://newsapi.org/) account and API key
- An [Alpaca](https://alpaca.markets/) account and API key (for live data)

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
pip install alpaca-trade-api
pip install transformers  # for FinBERT
pip install torch
pip install newsapi-python
pip install pandas numpy matplotlib
```

### Configure API Keys

Create a `.env` file in the root directory and add your API credentials:

```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
NEWS_API_KEY=your_newsapi_key
```

### Download the Data

The datasets are available in the `/Data` directory of this repo, or you can re-download them by running:

```bash
python src/download_data.py
```

This will pull historical data via `YahooDownloader` and live data via the Alpaca API for the top 30 S&P 500 tickers.
