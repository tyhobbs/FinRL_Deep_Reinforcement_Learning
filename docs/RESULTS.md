# Results

This document has three parts, in order of reliability:

1. **Universe-size sweep.** Multi-seed, full-period, per-run metrics. This is the project's result.
2. **Multi-seed architecture check.** Three seeds per architecture, but evaluated on a peak-truncated window.
3. **Original 24-model ablation.** Single seed, peak-truncated window. Kept for transparency, not as evidence.

---

## 1. Universe-size sweep

### Setup

| | |
|---|---|
| Model | VGG feature extractor + PPO, Alpaca price data, FinBERT news sentiment |
| Starting capital | $100,000 |
| Training period | 2020-01-01 → 2024-01-01 |
| Test period | 2024-01-01 → 2025-01-01 (full year, no truncation) |
| Universes | Nested: 30 ⊂ 35 ⊂ 40 ⊂ 45 ⊂ 50 stocks |
| Universe vetting | Gap-free price coverage over 2020–2025 required; GEV, SNDK, PLTR, and SHEL excluded |
| Seeds | 1–3 at every size; seeds 4–5 added at N=35 (see below) |
| Benchmark | Equal-weight buy-and-hold over the same universe and test window |

The vetted universes differ from the original ablation's universes, so the buy-and-hold values here differ from those in Section 3.

### Summary by universe size

| Universe | Runs | Sharpe (mean ± std) | Sharpe range | Return % (mean) | Max DD % (mean) | B&H Sharpe | B&H Return % | B&H Max DD % | Runs beating B&H |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 3 | 0.679 ± 0.965 | −0.043 to 1.775 | 16.58 | −17.52 | 1.540 | 32.41 | −12.57 | 1 / 3 |
| 35 | 5 | 1.348 ± 0.820 | 0.082 to 2.068 | 32.65 | −13.13 | 1.642 | 32.10 | −10.94 | 3 / 5 |
| 40 | 3 | 1.525 ± 0.400 | 1.200 to 1.972 | 36.48 | −11.68 | 1.678 | 31.43 | −10.03 | 1 / 3 |
| 45 | 3 | 1.236 ± 0.201 | 1.082 to 1.464 | 36.83 | −17.53 | 1.710 | 34.16 | −10.84 | 0 / 3 |
| 50 | 3 | 1.329 ± 0.348 | 1.059 to 1.722 | 40.17 | −15.39 | 1.668 | 33.61 | −10.70 | 1 / 3 |

Standard deviations are sample standard deviations. With three to five seeds per size, treat them as rough dispersion indicators.

### Per-run results

| N | Seed | Sharpe | Return % | Max DD % | B&H Sharpe | B&H Return % | B&H Max DD % | Beats B&H |
|---|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.306 | 9.84 | −19.35 | 1.540 | 32.41 | −12.57 | No |
| 30 | 2 | −0.043 | 0.21 | −22.27 | 1.540 | 32.41 | −12.57 | No |
| 30 | 3 | 1.775 | 39.69 | −10.93 | 1.540 | 32.41 | −12.57 | Yes |
| 35 | 1 | 1.872 | 55.30 | −12.83 | 1.642 | 32.10 | −10.94 | Yes |
| 35 | 2 | 2.068 | 40.75 | −7.63 | 1.642 | 32.10 | −10.94 | Yes |
| 35 | 3 | 1.740 | 37.49 | −11.74 | 1.642 | 32.10 | −10.94 | Yes |
| 35 | 4 | 0.082 | 4.67 | −18.00 | 1.642 | 32.10 | −10.94 | No |
| 35 | 5 | 0.977 | 25.05 | −15.46 | 1.642 | 32.10 | −10.94 | No |
| 40 | 1 | 1.972 | 47.52 | −11.23 | 1.678 | 31.43 | −10.03 | Yes |
| 40 | 2 | 1.200 | 31.80 | −13.99 | 1.678 | 31.43 | −10.03 | No |
| 40 | 3 | 1.402 | 30.13 | −9.83 | 1.678 | 31.43 | −10.03 | No |
| 45 | 1 | 1.163 | 20.42 | −8.25 | 1.710 | 34.16 | −10.84 | No |
| 45 | 2 | 1.082 | 37.93 | −23.04 | 1.710 | 34.16 | −10.84 | No |
| 45 | 3 | 1.464 | 52.13 | −21.31 | 1.710 | 34.16 | −10.84 | No |
| 50 | 1 | 1.206 | 44.01 | −19.96 | 1.668 | 33.61 | −10.70 | No |
| 50 | 2 | 1.059 | 27.41 | −11.62 | 1.668 | 33.61 | −10.70 | No |
| 50 | 3 | 1.722 | 49.09 | −14.58 | 1.668 | 33.61 | −10.70 | Yes |

"Beats B&H" compares Sharpe ratios only.

### Interpretation

**No universe size beats buy-and-hold on average.** Every size's mean Sharpe is below its benchmark, and only 6 of 17 runs clear it. That pattern is consistent with noise scattered around a mean slightly below buy-and-hold. The benchmark also had shallower drawdowns at every size (about −11% on average, versus about −15% for the models).

**Seed variance dominates design variance.** Within a size, seeds span up to 2.0 Sharpe points. Across sizes, the means span 0.85. The random initialization had a larger effect than the variable the sweep was designed to test.

**Universe size is within noise.** The mean ± std bands overlap across all five sizes. Size and composition are also entangled: each larger universe is the smaller one plus specific additional names.

**Why N=35 has five seeds.** After three seeds, N=35 was the only size whose runs all beat buy-and-hold (mean 1.89 vs. 1.64). Because that was the result the sweep's conclusion hinged on, I ran two more seeds instead of accepting it. They came in at 0.082 and 0.977, and the apparent edge disappeared. Had the model been deployed after three seeds, it would have been deployed on noise.

**Correction during the sweep.** The metrics first recorded for these runs were averages of evaluation outputs rather than canonical per-run results, which inflated them. Every run was re-evaluated, and all numbers above are the corrected per-run values.

---

## 2. Multi-seed architecture check (peak-truncated)

Three seeds × four architectures, 30-stock universe, $100k, 2024 test period. **These metrics were computed up to each run's peak portfolio value**, so they overstate performance and aren't comparable to full-year buy-and-hold. What they do show validly is how much single-seed results move across seeds.

| Architecture | Original single-seed Sharpe | Mean ± std across 3 seeds | Verdict |
|---|---|---|---|
| VGG + Alpaca | 2.531 | 2.476 ± 0.263 | Confirms |
| Transformer | 1.468 | 1.690 ± 0.166 | Confirms |
| VGG Baseline | 2.287 | 2.136 ± 0.065 | Diverges |
| VGG + FinBERT | 2.350 | 1.827 ± 0.068 | Diverges |

The Transformer's always-buy policy appeared in all three seeds. Full per-seed metrics are in [`Metrics/Multi_seed Metrics.txt`](../Metrics/Multi_seed%20Metrics.txt).

---

## 3. Original 24-model ablation (exploratory, peak-truncated)

Single seed per model, 2024 test period, 5% annualized risk-free rate, and 0.15% transaction costs per trade (0.1% commission + 0.05% slippage).

**Why these numbers are not evidence of performance.** Test metrics were computed only up to each model's peak portfolio value, while buy-and-hold was measured over the full year. Ending a measurement at the high-water mark flatters any strategy by construction. The earlier summary claims based on this table ("16 of 24 models beat buy-and-hold," "$100k is the optimal capital level," "30-stock outperforms 50-stock," and "DRL adds value in volatile periods") are withdrawn. The regime split was also computed on the truncated series, and the later sweep showed that single-seed differences of this size are within seed noise.

The table is kept so the repository's history is transparent. The "vs. buy-and-hold" column from the earlier version has been removed because it compared different windows.

| Model | Universe | Capital | Test Sharpe (to peak) | Test Return (to peak) | Max DD (to peak) |
|---|---|---|---|---|---|
| VGG Baseline | 30-Stock | $10k | 3.111 | 98.70% | −11.27% |
| VGG Baseline | 30-Stock | $1M | 3.086 | 48.22% | −5.00% |
| VGG + Alpaca | 50-Stock | $10k | 2.726 | 9.51% | −3.71% |
| VGG + Alpaca | 30-Stock | $100k | 2.531 | 57.16% | −9.50% |
| VGG + FinBERT | 50-Stock | $100k | 2.347 | 42.47% | −7.02% |
| VGG + FinBERT | 30-Stock | $100k | 2.350 | 48.92% | −8.95% |
| VGG + FinBERT | 30-Stock | $1M | 2.349 | 49.02% | −10.84% |
| VGG Baseline | 30-Stock | $100k | 2.287 | 55.37% | −19.86% |
| VGG + Alpaca | 30-Stock | $10k | 2.111 | 123.85% | −20.21% |
| VGG + Alpaca | 50-Stock | $100k | 2.019 | 72.48% | −15.28% |
| Transformer | 50-Stock | $1M | 1.861 | 38.26% | −11.27% |
| Transformer | 50-Stock | $100k | 1.895 | 52.93% | −19.62% |
| Transformer | 50-Stock | $10k | 1.806 | 33.99% | −8.89% |
| Transformer | 30-Stock | $1M | 1.748 | 55.23% | −19.70% |
| VGG + FinBERT | 50-Stock | $10k | 1.695 | 68.48% | −21.66% |
| VGG + Alpaca | 50-Stock | $1M | 1.575 | 29.96% | −7.78% |
| VGG + Alpaca | 30-Stock | $1M | 1.570 | 31.41% | −12.52% |
| VGG Baseline | 50-Stock | $10k | 1.504 | 30.98% | −10.33% |
| Transformer | 30-Stock | $100k | 1.468 | 25.91% | −11.93% |
| VGG Baseline | 50-Stock | $100k | 1.381 | 32.11% | −10.75% |
| VGG + FinBERT | 30-Stock | $10k | 1.338 | 41.07% | −19.24% |
| VGG + FinBERT | 50-Stock | $1M | 1.273 | 23.61% | −8.42% |
| VGG Baseline | 50-Stock | $1M | 1.001 | 20.97% | −9.33% |
| Transformer | 30-Stock | $10k | 0.629 | 13.31% | −12.14% |

### Buy-and-hold benchmarks for the original universes (full year)

| Universe | Sharpe | Return | Max DD |
|---|---|---|---|
| 30-Stock | 1.975 | 39.34% | −10.82% |
| 50-Stock | 1.437 | 25.53% | −7.80% |

Full original metrics are in [`Metrics/METRICS.txt`](../Metrics/METRICS.txt).

---

*Back to [README](../README.md)*
