# Installation

---

## Prerequisites

- Python 3.8+
- A [Polygon.io](https://polygon.io) account and API key (Stocks Starter plan or above recommended for full historical news coverage)
- An [Alpaca](https://alpaca.markets) account and API key (for Alpaca-based models and paper trading)

---

## Clone the Repository

```bash
git clone https://github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning.git
cd FinRL_Deep_Reinforcement_Learning
```

---

## Install Dependencies

```bash
pip install -r Environment/requirements.txt
```

Or install core packages manually:

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

---

## Configure API Keys

Create a `.env` file in the root directory:

```
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
POLYGON_API_KEY=your_polygon_api_key
```

---

## Running the Notebooks

Notebooks are organized by universe and architecture under `Code/`:

```
Code/
├── 30-Stock Universe/
│   ├── testFinRL_30stocks_{1M,100k,10k}.ipynb   ← VGG Baseline
│   ├── testFinRLwithNewsFetch_polygon_30stocks_{1M,100k,10k}.ipynb     ← VGG + FinBERT
│   ├── FinRL_Alpaca_polygon_30stocks_{1M,100k,10k}.ipynb ← VGG + Alpaca
│   └── FinRL_transformer_polygon_{1M,100k,10k}.ipynb          ← Transformer
└── 50-Stock Universe/
    ├── BaselineVGG_50stocks_{1M,100k,10k}.ipynb
    ├── VGG_yfinance_polygon_50stocks_{1M,100k,10k}.ipynb
    ├── VGG_Alpaca_polygon_50stocks_{1M,100k,10k}.ipynb
    └── FinRL_transformer_polygon_50stocks_{1M,100k,10k}.ipynb
```

Run each notebook top to bottom. Sentiment data is fetched and cached locally on first run (~30–45 minutes per model) and loaded instantly on subsequent runs.

---

## Paper Trading Setup

To run the live paper trading pipeline:

1. Ensure you have a paper trading account on Alpaca with $100,000 starting balance
2. Save the trained VGG + Alpaca model and ticker list:
   ```python
   trained_vgg.save('./best_vgg_alpaca_30stocks_100k')
   import json
   with open('./vgg_alpaca_30stocks_100k_tickers.json', 'w') as f:
       json.dump(TICKERS, f)
   ```
3. Open `Papertrading.ipynb` and run cells 0 through 4 in order
4. The scheduler fires daily at 9:45 AM ET with intraday stop-loss at 1:00 PM ET and end-of-day logging at 3:55 PM ET

---

## Hardware Notes

| Task | Recommended | Minimum |
|------|-------------|---------|
| VGG training | Mac M-series MPS or CUDA GPU | CPU (slow) |
| Transformer training | CUDA GPU (4GB+ VRAM) | CPU (very slow) |
| Paper trading inference | Any CPU | Any CPU |

Training times on Apple M4 MacBook Pro:
- VGG models: ~30–45 minutes per run
- Transformer models: ~3–5 hours per run

Paper trading inference runs in under 90 seconds per day and has negligible CPU impact.

---

*Back to [README](../README.md)*
