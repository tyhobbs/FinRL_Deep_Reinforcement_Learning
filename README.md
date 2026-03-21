# FinRL: Deep Reinforcement Learning for Automated Stock Trading

[![Results Dashboard](https://img.shields.io/badge/Results-GitHub_Pages-blue?logo=github)](https://tyhobbs.github.io/FinRL_Deep_Reinforcement_Learning/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Release](https://img.shields.io/github/v/release/tyhobbs/FinRL_Deep_Reinforcement_Learning?color=orange)](https://github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning/releases/tag/v1.0.0)
[![Visitors](https://hits.sh/github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning.svg?label=Visitors&color=blue)](https://hits.sh/github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning/)

This project integrates **Deep Reinforcement Learning (DRL)** with **LLM-driven sentiment analysis** to build an automated stock trading system using the [FinRL](https://github.com/AI4Finance-Foundation/FinRL) framework. Models are benchmarked across multiple architectures, data sources, stock universes, and starting capital levels to evaluate both institutional and retail-scale performance.

---

## Ablation Study Design

Each model changes exactly one variable relative to the previous, allowing clean attribution of performance gains:

| Step | Comparison | Variable Isolated |
|------|-----------|------------------|
| 1 | VGG Baseline → VGG + FinBERT | Effect of sentiment |
| 2 | VGG + FinBERT (Yahoo) → VGG + FinBERT (Polygon) | Effect of live data |
| 3 | VGG + FinBERT (Polygon) → VGG + Alpaca | Effect of architecture |
| 4 | VGG + Alpaca → Cross-Stock Transformer | Effect of Transformer |
| 5 | $1M → $100k → $10k, 30-stock → 50-stock | Effect of universe and capital |
| 6 | Best DRL model vs Buy-and-Hold | Justification for DRL complexity |

---

## Key Findings

- **18 of 24 models beat buy-and-hold** on a risk-adjusted basis
- **$100k is the optimal capital level** — highest average test Sharpe (2.035) across all architectures
- **VGG + Alpaca outperforms the Transformer** (avg Sharpe 2.089 vs 1.568) — local convolutional feature extraction is more data-efficient than global attention for daily trading
- **30-stock universe outperforms 50-stock** on average (Sharpe 1.987 vs 1.629)
- **Best single model**: 30-Stock VGG Baseline $10k — Test Sharpe **3.111**, Return **+98.70%**

---

## Results Summary

| Architecture | Avg Sharpe | Best | Worst |
|-------------|-----------|------|-------|
| VGG + Alpaca | **2.089** | 2.726 | 1.570 |
| VGG Baseline | 1.939 | 3.111 | 1.001 |
| VGG + FinBERT | 1.892 | 2.350 | 1.273 |
| Transformer | 1.568 | 1.895 | 0.629 |

| Capital | Avg Sharpe | Best | Worst |
|--------|-----------|------|-------|
| $100k | **2.035** | 2.531 | 1.381 |
| $10k | 1.865 | 3.111 | 0.629 |
| $1M | 1.716 | 2.350 | 1.001 |

→ **[Full 24-model results table](docs/RESULTS.md)**

---

## Quick Start

```bash
git clone https://github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning.git
cd FinRL_Deep_Reinforcement_Learning
pip install -r Environment/requirements.txt
```

→ **[Full installation and setup guide](INSTALLATION.md)**

---

## Repository Structure

```
FinRL_Deep_Reinforcement_Learning/
├── Environment/             ← requirements.txt, utils.py
├── Code/
│   ├── 30-Stock Universe/   ← 4 architectures × 3 capital levels
│   └── 50-Stock Universe/   ← 4 architectures × 3 capital levels
├── Metrics/                 ← METRICS.txt
├── Data/                    ← Training and validation CSVs
├── Literature Paper/        ← Reference papers
└── docs/                    ← GitHub Pages dashboard + RESULTS.md
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/RESULTS.md](docs/RESULTS.md) | Full 24-model results, regime analysis, benchmark comparison |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | VGG and Transformer architecture details, reward function |
| [docs/DATASET.md](docs/DATASET.md) | Data sources, indicators, sentiment pipeline, observation space |
| [docs/UNIVERSES.md](docs/UNIVERSES.md) | Full ticker lists for 30-stock and 50-stock universes |
| [INSTALLATION.md](INSTALLATION.md) | Setup, API keys, hardware requirements, paper trading |

---

## Future Work

- **Live deployment**: A kill-switch paper trading system is currently live on Alpaca (deployed March 16, 2026) with intraday stop-loss protection, automated daily execution, and end-of-day portfolio logging via the 30-Stock VGG + Alpaca \$100k model (test Sharpe 2.531). Minimum 63-day validation period targeting comparison against backtested risk-adjusted returns.

- **Open-source package**: The core components developed in this work — the four-component reward function, Cross-Stock Transformer architecture, VGG feature extractor, `TrainSharpeSavingCallback`, and evaluation framework — will be extracted and published as a standalone Python package on PyPI, providing a clean reproducible interface for DRL-based trading research.

---

