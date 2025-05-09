# AI Hedge Fund

Inspired by https://github.com/virattt/ai-hedge-fund

A multi-agent AI system for financial analysis and portfolio management, featuring:

- **Investment Agents**: Specialized analysis agents modeled after famous investors
- **Portfolio Manager**: Makes buying/selling decisions based on agent signals
- **Risk Manager**: Sets risk parameters for portfolio management
- **Backtester**: (Coming soon) - Test strategies on historical data

## Getting Started

### Prerequisites

- Python 3.8+
- Alpha Vantage API key (for market data)
- DeepSeek API key (for agent models)

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### API Keys Configuration

This system requires two API keys to function properly:

1. **Alpha Vantage API Key**: Used to fetch market data
   - Get a free key at: https://www.alphavantage.co/support/#api-key
   - Set it as an environment variable: `export ALPHA_VANTAGE_API_KEY=your_key_here`

2. **DeepSeek API Key**: Used for LLM capabilities in agents
   - Get a key at: https://platform.deepseek.com/api-keys  
   - Set it as an environment variable: `export DEEPSEEK_API_KEY=your_key_here`

You can also set them in your terminal session before running commands:

```bash
# On Windows:
set ALPHA_VANTAGE_API_KEY=your_key_here
set DEEPSEEK_API_KEY=your_key_here

# On macOS/Linux:
export ALPHA_VANTAGE_API_KEY=your_key_here
export DEEPSEEK_API_KEY=your_key_here
```

## Usage

### 1. Run Individual Investment Agents

```bash
# Run Buffet-style analysis on a stock
SYMBOL=AAPL python ai_hedge_fund/buffet_agent.py

# Run Lynch-style analysis on a stock
SYMBOL=TSLA python ai_hedge_fund/lynch_agent.py
```

The agent analysis will be saved in the `data/` directory.

### 2. Use Portfolio Manager

Run the full pipeline (agents + portfolio manager):

```bash
# Analyze multiple stocks and make portfolio decisions (simulation only)
python -m ai_hedge_fund.run_portfolio_decision --tickers AAPL,MSFT,TSLA,NVDA

# Execute trades and update your portfolio
python -m ai_hedge_fund.run_portfolio_decision --tickers AAPL,MSFT,TSLA,NVDA --execute
```

Or test the portfolio manager directly:

```bash
# Run the portfolio manager with test data
python -m ai_hedge_fund.agents.portfolio_manager_agent
```

### 3. Portfolio Management

The system will:
1. Run agent analysis on each ticker
2. Extract investment signals
3. Generate risk parameters using the risk manager
4. Make portfolio decisions
5. Execute trades if requested
6. Save decisions to a timestamped file

## System Components

### Investment Agents

- **Buffet Agent**: Value investing approach focusing on business fundamentals
- **Lynch Agent**: Growth-oriented approach focusing on growth and potential

### Risk Manager

Calculates position sizing and risk parameters based on:
- Portfolio structure and cash levels
- LLM analysis of fundamental and technical risks
- Current market prices (from Alpha Vantage)

### Portfolio Manager

Consolidates signals from various analyst agents and applies risk parameters to make final trading decisions.

### Core Data Structures

- **AgentSignal**: Output from investment agents (bullish/bearish/neutral)
- **PortfolioDecision**: Final decision for a ticker (buy/sell/hold + quantity)
- **PortfolioRiskParameters**: Risk constraints for portfolio management

## Configuration

- Edit `run_portfolio_decision.py` to modify the risk parameters and logic
- Create a `portfolio.json` file to track your virtual portfolio

## Extending the System

- Add new agents by following the pattern in existing agent files
- Improve the LLM implementation in the portfolio manager
- Implement the backtester to evaluate strategies 
