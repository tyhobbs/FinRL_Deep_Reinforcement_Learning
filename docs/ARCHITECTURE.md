# Architecture

Four model architectures are evaluated in this ablation study, each building on the previous to isolate the contribution of each component.

---

## RL Algorithm

All models use **Proximal Policy Optimization (PPO)** via Stable Baselines 3 with the following shared hyperparameters:

| Parameter | Value |
|-----------|-------|
| `n_steps` | 2048 |
| `ent_coef` | 0.01 |
| `learning_rate` | 0.0001 (linear decay) |
| `batch_size` | 64 |
| `total_timesteps` | 500,000 (VGG) / 1,100,000 (Transformer) |

---

## VGG Feature Extractor

Used in VGG Baseline, VGG + FinBERT, and VGG + Alpaca models. The observation vector is reshaped into a 2D matrix of shape `(N_stocks × N_indicators)` and passed through two convolutional blocks, capturing cross-stock and cross-indicator relationships simultaneously.

```
Input: observation vector (331-dim for 30-stock)
  ↓
Reshape: (30 stocks × 9 indicators) → (1, 30, 9) 2D matrix
  ↓
BatchNorm2d(1)
  ↓
Conv2d(1→32, kernel=3, padding=1) → BatchNorm2d(32) → ReLU
  ↓
Conv2d(32→64, kernel=3, padding=1) → BatchNorm2d(64) → ReLU
  ↓
Flatten
  ↓
Linear(n_flatten → 512) → ReLU
  ↓
512-dimensional feature vector → PPO actor-critic head
```

**Key design choice**: 2D convolution treats stocks as rows and indicators as columns, allowing the network to learn cross-stock patterns (e.g. sector correlations) and cross-indicator patterns (e.g. RSI + MACD co-movement) simultaneously within a single convolutional pass.

---

## Cross-Stock Transformer

Used in the Transformer + FinBERT + Alpaca model. A two-stage attention architecture that first captures temporal patterns within each stock and then captures cross-stock dependencies.

```
Input: 10-day lookback window of observations
  ↓
Stage 1 — Temporal Self-Attention (per stock)
  For each of N stocks independently:
  Transformer encoder over 10-day sequence
  Captures intra-stock temporal patterns
  ↓
Stage 2 — Cross-Stock Attention
  At the most recent timestep:
  Transformer encoder across all N stocks simultaneously
  Captures inter-stock dependencies and correlations
  ↓
Flatten + Linear projection
  ↓
PPO actor-critic head
```

**VecNormalize**: The Transformer uses SB3's `VecNormalize` to normalize the observation space. At inference time this requires manual normalization using saved `obs_rms` statistics to work around a known SB3 internal copy issue:

```python
obs_normalized = np.clip(
    (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
    -clip_obs, clip_obs
).astype(np.float32)
```

---

## Reward Function

All models use a four-component reward function:

```
R_t = R_sharpe + R_sentiment − P_drawdown − P_concentration
```

| Component | Description |
|-----------|-------------|
| `R_sharpe` | Encourages risk-adjusted returns — reward proportional to rolling Sharpe ratio |
| `R_sentiment` | Bonus for aligning trades with FinBERT sentiment direction |
| `P_drawdown` | Penalty when unrealized drawdown exceeds −10% |
| `P_concentration` | Penalty for individual stock weights exceeding 15% of portfolio |

---

## Training Improvements

| Component | Description |
|-----------|-------------|
| `TrainSharpeSavingCallback` | Saves model weights at the episode with the highest training Sharpe ratio, preventing final-epoch overfit |
| `CheckpointCallback` | Periodic weight saves every 50,000 steps for recovery |
| Linear LR schedule | Learning rate decays linearly from 1e-4 to 0 over training |
| VecNormalize | Running mean/variance normalization of observations (Transformer only) |

---

## Model Comparison

| Architecture | State Space | Parameters | Training Steps | VecNormalize |
|-------------|-------------|------------|----------------|--------------|
| VGG Baseline | 331 | ~2.1M | 500,000 | No |
| VGG + FinBERT | 331 | ~2.1M | 500,000 | No |
| VGG + Alpaca | 331 | ~2.1M | 500,000 | No |
| Transformer | 4,810 | ~8.2M | 1,100,000 | Yes |

---

## Why VGG Outperforms the Transformer

Contrary to the initial hypothesis, the Cross-Stock Transformer underperforms VGG on this dataset (avg Sharpe 1.568 vs 2.089). Several factors likely contribute:

1. **Data efficiency**: 4 years of daily trading data (~1,000 trading days) is insufficient for a transformer to learn robust global attention patterns. VGG's local convolutional inductive bias requires less data to converge.

2. **Sequence length**: A 10-day lookback window may be too short for temporal attention to capture meaningful patterns beyond what the 8 technical indicators already encode.

3. **Training instability**: Transformer models exhibit higher variance during training and an early negative Sharpe spike that VGG models do not show, suggesting the attention mechanism struggles with the initial random policy phase of PPO.

---

*Back to [README](../README.md)*
