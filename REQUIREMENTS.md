# Requirements Specification

## 1. Users and Environment

### 1.1 User Groups

| User Type          | Primary Goal                                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------------------|
| Data Curator       | Build and maintain reliable research datasets across market, fundamental, analytics, and alternative sources |
| Feature Analyst    | Create and evaluate predictive features for downstream modeling and signal generation                        |
| Strategist         | Design and refine investment strategies that can be tested and deployed                                      |
| Backtester         | Measure strategy behavior and robustness before live deployment                                              |
| Deployment Manager | Operate live trading workflows safely and reliably in production                                             |

### 1.2 Operating Environment

| Operating System | Interface |
|------------------|-----------|
| macOS            | CLI       |

## 2. Functional Requirements

### 2.1 Data Preprocessing

- Fundamental Data: Assets, Liabilities, Sales, Costs/earnings, Macro variables, …
- Market Data: Price/yield/implied volatility, Volume, Dividend/coupons, Open interest, Quotes/cancellations, Aggressor
  side, …
- Analytics: Analyst recommendations, Credit ratings, Earnings expectations, News sentiment, …
- Alternative Data: Satellite/CCTV images, Google searches, Twitter/chats, Metadata, …

### 2.2 Feature Analysis

- Implementation of Core Factors

### 2.3 Strategy Research

- Portfolio strategies: value investing, insider trading, long-short, trend following, mean reversion
- Arbitrage strategies: market making, statistical arbitrage, event-driven arbitrage

### 2.4 Backtesting

- Generate synthetic data
- Evaluate backtesting metrics in Jupyter notebooks
- Study reinforcement learning for HFT, where it may be effective

### 2.5 Automated Trading

- Enable online machine learning in live trading by supporting real-time data processing
- Connect to automated trading connectors for ultra-low-latency execution
- Supports Equity, Crypto trading in various platforms

### 2.6 Other Requirements

- User authentication
- Exception and failure handling
- 24/7 operation
- Log management
