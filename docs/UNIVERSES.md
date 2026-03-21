# Stock Universes

Two stock universes are evaluated in this study — a focused 30-stock universe of NASDAQ-listed technology and healthcare companies, and an expanded 50-stock universe that adds coverage across financials, energy, consumer staples, and emerging tech sectors.

---

## 30-Stock Universe

30 stocks drawn from NASDAQ-listed technology, healthcare, and consumer companies. This universe forms the core of the ablation study and is used for all architecture comparisons.

| Sector | Tickers |
|--------|---------|
| Technology | AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, PYPL, ADBE, NFLX |
| Semiconductors | INTC, CSCO, AVGO, QCOM, TXN, ADI |
| Software / Services | INTU, ADP, BKNG |
| Consumer Discretionary | PEP, COST, SBUX, MDLZ |
| Healthcare / Biotech | AMGN, GILD, ISRG, VRTX |
| Industrials | HON |
| Communication Services | CMCSA, TMUS |

**Total: 30 stocks across 7 sectors**

---

## 50-Stock Universe

The core 30-stock universe expanded with 20 additional stocks across underrepresented sectors to evaluate whether a broader universe improves or degrades DRL performance. Results show the 30-stock universe outperforms the 50-stock universe on average (Sharpe 1.987 vs 1.629), suggesting that a focused high-liquidity universe is preferable for this model class.

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

**Total: 50 stocks across 13 sectors**

---

## Universe Selection Rationale

The 30-stock universe was chosen to focus on high-liquidity, high-sentiment-coverage names where FinBERT scoring is most reliable. All 30 stocks have consistent Polygon.io news coverage from 2020–2025, ensuring sentiment features are available throughout the full training and validation period.

The 50-stock expansion adds sector diversity but introduces several names with lower news coverage and higher volatility, which may explain the lower average Sharpe versus the focused 30-stock universe.

---

*Back to [README](../README.md)*
