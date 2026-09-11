# FinRL: Deep Reinforcement Learning for Automated Stock Trading

[![Results Dashboard](https://img.shields.io/badge/Results-GitHub_Pages-blue?logo=github)](https://tyhobbs.github.io/FinRL_Deep_Reinforcement_Learning/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Release](https://img.shields.io/github/v/release/tyhobbs/FinRL_Deep_Reinforcement_Learning?color=orange)](https://github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning/releases/tag/v1.0.0)
[![Visitors](https://hits.sh/github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning.svg?label=Visitors&color=blue)](https://hits.sh/github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning/)
[![Dashboard Repo](https://img.shields.io/badge/Repo-Trading_Dashboard-gray?logo=github)](https://github.com/tyhobbs/finrl-dashboard)

This project integrates **Deep Reinforcement Learning (DRL)** with **LLM-driven sentiment analysis** to build an automated stock trading system using the [FinRL](https://github.com/AI4Finance-Foundation/FinRL) framework. Models are benchmarked across multiple architectures, data sources, stock universes, and starting capital levels to evaluate both institutional and retail-scale performance. Single-seed backtests looked strong, but under multi-seed evaluation over the full test period, no configuration reliably beat an equal-weight buy-and-hold benchmark.

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
| 7 | Multiple seeds per configuration, 30 → 50 stock universes | Robustness to seed and universe size |

---

## Key Findings

- **No configuration reliably beat buy-and-hold** over the full test year: every universe size from 30 to 50 stocks had a mean test Sharpe below its equal-weight benchmark, and only 6 of 17 runs beat theirs
- **Seed initialization was the dominant variable**: seeds within a single universe size spanned up to 2.0 Sharpe points, more than the 0.85 spread across universe-size means
- **Universe size (30–50 stocks) was within seed noise**: the mean ± std bands overlap at every size
- **An apparent edge dissolved under more seeds**: the 35-stock universe averaged 1.89 against a 1.64 benchmark after three seeds; two more seeds brought it to 1.35 ± 0.82
- **The original 24-model results overstated performance**: test metrics were computed only up to each model's peak portfolio value while buy-and-hold was measured over the full year, so "16 of 24 models beat buy-and-hold" and the 3.111 best Sharpe are not valid out-of-sample comparisons

---

## Results Summary

**Universe size** (VGG + Alpaca, $100k, full 2024 test year; 3 seeds per size, 5 at 35 stocks)

| Universe | Avg Sharpe | Best | Worst | Buy-and-Hold |
|----------|-----------|------|-------|--------------|
| 30-Stock | 0.679 | 1.775 | −0.043 | 1.540 |
| 35-Stock | 1.348 | 2.068 | 0.082 | 1.642 |
| 40-Stock | **1.525** | 1.972 | 1.200 | 1.678 |
| 45-Stock | 1.236 | 1.464 | 1.082 | 1.710 |
| 50-Stock | 1.329 | 1.722 | 1.059 | 1.668 |

**Architecture** (30-stock, $100k, 3 seeds each; measured to peak portfolio value, so useful for seed-to-seed variation but not comparable to buy-and-hold)

| Architecture | Avg Sharpe | Best | Worst |
|-------------|-----------|------|-------|
| VGG + Alpaca | **2.476** | 2.810 | 2.166 |
| VGG Baseline | 2.136 | 2.196 | 2.045 |
| VGG + FinBERT | 1.827 | 1.923 | 1.776 |
| Transformer | 1.690 | 1.815 | 1.455 |

→ **[Full results, including the original 24-model table](docs/RESULTS.md)**

---

## Quick Start

```bash
git clone https://github.com/tyhobbs/FinRL_Deep_Reinforcement_Learning.git
cd FinRL_Deep_Reinforcement_Learning
pip install -r Code/Environment/requirements.txt
```

→ **[Full installation and setup guide](INSTALLATION.md)**

---

## Repository Structure

```
FinRL_Deep_Reinforcement_Learning/
├── Code/
│   ├── Environment/         ← requirements.txt, utils.py
│   ├── 30 Stock Universe/   ← 4 architectures × 3 capital levels
│   └── 50 Stock Universe/   ← 4 architectures × 3 capital levels
├── Metrics/
│   ├── METRICS.txt                                     ← Original 24-model ablation metrics (to peak)
│   ├── Multi_seed Metrics.txt                          ← Multi-seed summary results (to peak)
│   ├── multi_seed_full_metrics_30stocks_100k.csv       ← Full per-seed metric breakdown
│   └── MultiSeed_Results_Collector_30stocks_100k.ipynb ← Multi-seed results notebook
├── Data/
│   ├── 30 Stock Universe/   ← Training and validation CSVs
│   ├── 50 Stock Universe/   ← Training and validation CSVs
│   └── Seed 1/, Seed 2/, Seed 3/  ← Multi-seed run outputs
├── Literature Paper/        ← Reference papers
└── docs/                    ← GitHub Pages dashboard, RESULTS.md, ARCHITECTURE.md, and other relevant figures
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/RESULTS.md](docs/RESULTS.md) | Universe-size and multi-seed results, original 24-model ablation, benchmark comparison |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | VGG and Transformer architecture details, reward function |
| [docs/DATASET.md](docs/DATASET.md) | Data sources, indicators, sentiment pipeline, observation space |
| [docs/UNIVERSES.md](docs/UNIVERSES.md) | Full ticker lists for 30-stock and 50-stock universes |
| [INSTALLATION.md](INSTALLATION.md) | Setup, API keys, hardware requirements, paper trading |
| [docs/trading_system_overview.pdf](docs/trading_system_overview.pdf) | System overview diagram |
| [Trading Dashboard Repo](https://github.com/tyhobbs/finrl-dashboard) | Paper trading monitor used during deployment (March–June 2026) |

---

## Future Work

- **Live deployment**: A kill-switch paper trading system ran on Alpaca starting March 16, 2026, with intraday stop-loss protection, automated daily execution, and end-of-day portfolio logging via the 30-Stock VGG + Alpaca \$100k model. Running it surfaced a short-selling bug, where two sell paths in `execute_trades` compounded to sell more than was held; this was fixed along with a 10% position cap and live buying-power re-fetching. The deployment was shut down in June 2026 after multi-seed evaluation found no model that reliably beat buy-and-hold.

- **Open-source package**: The core components developed in this work — the four-component reward function, Cross-Stock Transformer architecture, VGG feature extractor, `TrainSharpeSavingCallback`, and evaluation framework — will be extracted and published as a standalone Python package on PyPI, providing a clean reproducible interface for DRL-based trading research.

---
