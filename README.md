# Multi Asset Investing

## Project Description

This project is a system for financial research and automated trading. It is heavily inspired by Marcos de Prado's
Advances in Financial Machine Learning, and it tries to apply those ideas in a practical software system. It also
provides basic strategies to test out, which were inspired by Lasse Heje Pedersen's Efficiently Inefficient.

While most projects that follow Advances in Financial Machine Learning provide a list of core functions and solutions to
problems given in the book, this project aims to be a more complete software framework for the entire investment
research workflow by including data preprocessing, feature analysis, strategy templates, backtesting, and execution
features.

Also, most projects do not cover financial research and automated trading in the same project, so it often leads to
re-implementing the same algorithms in different software. This project tries to reduce that duplication by providing a
shared foundation that can be used across experiments and live trading systems.

## Directory Structure

The project is organized as follows:

```text
.
|-- alpha_models/
|-- data_preprocessing/
|-- feature_analysis/
|-- live_trading/
|-- model_backtest/
|-- REQUIREMENTS.md
|-- ARCHITECTURE.md
`-- DECISIONS.md
```

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy pyarrow
```

Set Alpaca credentials to fetch market data:

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

## Usage

Run a minimal preprocessing pipeline from raw trades to bar-based features:

```python
from data_preprocessing.fetch_market_data import fetch_historical_trades
from data_preprocessing.financial_data_structures import get_dollar_bars
from data_preprocessing.financial_data_labeling import get_daily_volatility
from data_preprocessing.fractionally_differentiate_features import (
    fractional_difference,
)

trades = fetch_historical_trades(
    ["AAPL"],
    start="2025-01-02T09:30:00Z",
    end="2025-01-02T16:00:00Z",
)

bars = get_dollar_bars(trades, threshold=50_000)
ohlcv = bars.ohlcv

close = ohlcv["close"]
daily_volatility = get_daily_volatility(close)
ffd_close = fractional_difference(ohlcv[["close"]], d=0.4, thres=0.01)
```
