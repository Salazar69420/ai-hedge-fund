import os
import subprocess
import json
from typing import List, Dict, Any
import argparse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Import our modules
from .agents.agent_signal_extractor import get_agent_signals
from .core.data_structures import (
    AgentSignal,
    PortfolioRiskParameters, 
    TickerRiskParameters,
    PortfolioManagerOutput,
    CurrentPortfolio,
    PortfolioPosition,
    Transaction
)
from .agents.portfolio_manager_agent import make_portfolio_decisions
from .agents.risk_manager_agent import generate_risk_parameters_for_tickers

# API configuration - load from environment variables or use defaults
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')

def run_agent_analysis(agent_file: str, ticker: str) -> bool:
    """Run an agent's analysis on a ticker"""
    print(f"Running {agent_file} analysis for {ticker}...")
    env = os.environ.copy()
    env["SYMBOL"] = ticker
    
    # Pass API keys to agent scripts
    if ALPHA_VANTAGE_API_KEY:
        env["ALPHA_VANTAGE_API_KEY"] = ALPHA_VANTAGE_API_KEY
    if DEEPSEEK_API_KEY:
        env["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    
    # Run agent script
    result = subprocess.run(
        ["python", agent_file],
        env=env,
        text=True,
        capture_output=True  # Capture output to avoid cluttering the console
    )
    
    if result.returncode != 0:
        print(f"Error running {agent_file}:")
        print(result.stderr)
        return False
    else:
        print(f"✓ {agent_file} analysis complete")
        return True

def load_portfolio(portfolio_file: str = "portfolio.json") -> CurrentPortfolio:
    """Load portfolio from file or create a default one, returns a CurrentPortfolio object."""
    if os.path.exists(portfolio_file):
        try:
            with open(portfolio_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Attempt to parse into the new Pydantic model
                # This allows for some flexibility if old format is slightly different
                # but will raise validation error if incompatible.
                return CurrentPortfolio(**data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {portfolio_file}: {e}. Creating default portfolio.")
            return CurrentPortfolio(cash=100000.0, positions={}, transactions=[])
        except Exception as e: # Catch Pydantic validation errors or other issues
            print(f"Error loading or validating portfolio from {portfolio_file}: {e}. Creating default portfolio.")
            return CurrentPortfolio(cash=100000.0, positions={}, transactions=[])
    else:
        # Default portfolio
        print(f"Portfolio file {portfolio_file} not found. Creating default portfolio.")
        return CurrentPortfolio(cash=100000.0, positions={}, transactions=[])

def save_portfolio(portfolio: CurrentPortfolio, portfolio_file: str = "portfolio.json") -> None:
    """Save CurrentPortfolio object to file."""
    with open(portfolio_file, 'w', encoding='utf-8') as f:
        json.dump(portfolio.model_dump(mode='json'), f, indent=2)

def update_portfolio_with_decisions(
    portfolio: CurrentPortfolio, 
    decisions: PortfolioManagerOutput,
    current_prices: Dict[str, float]
) -> CurrentPortfolio:
    """
    Update portfolio based on trading decisions, using Pydantic models.
    """
    updated_portfolio = CurrentPortfolio(
        cash=portfolio.cash,
        # Deepcopy positions to avoid modifying original portfolio's positions dict directly during iteration
        positions={k: v.model_copy() for k, v in portfolio.positions.items()},
        transactions=list(portfolio.transactions) # Shallow copy of list is fine, as Transaction objects are immutable once created or will be new
    )
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    for ticker, decision_obj in decisions.decisions.items(): # decision_obj is PortfolioDecision
        if ticker in ["DEFAULT_TICKER", "UNKNOWN_TICKER"]:
            continue
            
        current_price = current_prices.get(ticker, 0)
        if current_price == 0 and ticker in updated_portfolio.positions:
            current_price = updated_portfolio.positions[ticker].avg_price
        elif current_price == 0:
            current_price = 100.0  # Default if no other info

        # Ensure decision action is a string for direct comparison
        action_str = str(decision_obj.action) if decision_obj.action else "hold"
            
        if action_str == "buy" and decision_obj.quantity > 0:
            cost = current_price * decision_obj.quantity
            if cost <= updated_portfolio.cash:
                updated_portfolio.cash -= cost
                if ticker not in updated_portfolio.positions:
                    updated_portfolio.positions[ticker] = PortfolioPosition(
                        ticker=ticker, # Explicitly set ticker for new Position
                        shares=decision_obj.quantity,
                        avg_price=current_price
                    )
                else:
                    current_pos = updated_portfolio.positions[ticker]
                    total_shares = current_pos.shares + decision_obj.quantity
                    total_cost_basis = (current_pos.shares * current_pos.avg_price) + cost
                    new_avg_price = total_cost_basis / total_shares if total_shares > 0 else 0
                    current_pos.shares = total_shares
                    current_pos.avg_price = new_avg_price
                
                updated_portfolio.transactions.append(Transaction(
                    date=current_date, ticker=ticker, action="buy", 
                    quantity=decision_obj.quantity, price=current_price, total_value=cost
                ))
                print(f"Executed: BUY {decision_obj.quantity} shares of {ticker} at ${current_price:.2f}")
            else:
                print(f"Skipped BUY {ticker}: Insufficient cash. Need ${cost:.2f}, have ${updated_portfolio.cash:.2f}")
                
        elif action_str == "sell" and decision_obj.quantity > 0:
            if ticker in updated_portfolio.positions and updated_portfolio.positions[ticker].shares >= decision_obj.quantity:
                proceeds = current_price * decision_obj.quantity
                updated_portfolio.cash += proceeds
                
                current_pos = updated_portfolio.positions[ticker]
                current_pos.shares -= decision_obj.quantity
                
                if current_pos.shares == 0:
                    del updated_portfolio.positions[ticker]
                
                updated_portfolio.transactions.append(Transaction(
                    date=current_date, ticker=ticker, action="sell", 
                    quantity=decision_obj.quantity, price=current_price, total_value=proceeds
                ))
                print(f"Executed: SELL {decision_obj.quantity} shares of {ticker} at ${current_price:.2f}")
            else:
                current_shares = updated_portfolio.positions[ticker].shares if ticker in updated_portfolio.positions else 0
                print(f"Skipped SELL {ticker}: Insufficient shares. Tried to sell {decision_obj.quantity}, have {current_shares}")
    
    return updated_portfolio

def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get current prices for a list of tickers using Alpha Vantage API
    In a production system, this would use real-time data
    """
    prices = {}
    
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "demo":
        print("Warning: Using dummy prices because no Alpha Vantage API key provided")
        # Generate dummy prices
        for ticker in tickers:
            # Use portfolio price or a default
            prices[ticker] = 100.0  # Default price
        return prices
    
    import requests
    
    for ticker in tickers:
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            price_str = data.get("Global Quote", {}).get("05. price", "0")
            price = float(price_str)
            
            if price > 0:
                prices[ticker] = price
            else:
                print(f"Warning: Could not get price for {ticker}, using default")
                prices[ticker] = 100.0
                
        except Exception as e:
            print(f"Error getting price for {ticker}: {e}")
            prices[ticker] = 100.0
            
        # Alpha Vantage has rate limits, so add a small delay
        import time
        time.sleep(0.2)
    
    return prices

def save_decisions(decisions: PortfolioManagerOutput, ticker_list: List[str], history_folder: str = "data/decisions_history") -> str:
    """Save decisions to a file with timestamp in a dedicated history folder, returns filename."""
    os.makedirs(history_folder, exist_ok=True) # Ensure the history folder exists
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize ticker list for filename if it were to be very long, or just use a generic name
    # For now, keeping it simple, but for many tickers, this might be too long.
    # Consider f"decisions_portfolio_{timestamp}.json" or similar if ticker_list is huge.
    filename_base = f"decisions_run_{timestamp}.json" 
    filename = os.path.join(history_folder, filename_base)
    
    # decisions.decisions is already Dict[str, PortfolioDecision]
    # PortfolioDecision objects will be correctly serialized by model_dump
    decision_data_to_save = {
        "run_timestamp": timestamp,
        "analyzed_tickers": ticker_list,
        "portfolio_manager_output": decisions.model_dump(mode='json') # Save the whole output object
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(decision_data_to_save, f, indent=2)
    
    print(f"Decisions saved to {filename}")
    return filename

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run portfolio decisions based on agent signals')
    parser.add_argument('--tickers', type=str, required=True, 
                        help='Comma-separated list of ticker symbols to analyze')
    parser.add_argument('--portfolio', type=str, default='portfolio.json',
                        help='Path to CurrentPortfolio JSON file')
    parser.add_argument('--execute', action='store_true',
                        help='Execute trades and update portfolio (default: simulate only)')
    args = parser.parse_args()
    
    # Check API keys and warn if missing
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "demo":
        print("WARNING: Alpha Vantage API key not set. Using demo mode with limited functionality.")
        print("Set the ALPHA_VANTAGE_API_KEY environment variable for full functionality.")
    
    if not DEEPSEEK_API_KEY:
        print("WARNING: DeepSeek API key not set. Using default key from the code.")
        print("Set the DEEPSEEK_API_KEY environment variable for your own key.")
    
    # Parse tickers
    ticker_list = [ticker.strip().upper() for ticker in args.tickers.split(',')]
    if not ticker_list:
        print("Error: No tickers provided")
        return
    
    print(f"Analyzing tickers: {', '.join(ticker_list)}")
    
    # Load portfolio
    portfolio: CurrentPortfolio = load_portfolio(args.portfolio) # Typed for clarity
    print(f"Loaded portfolio with ${portfolio.cash:.2f} cash, {len(portfolio.positions)} positions, and {len(portfolio.transactions)} past transactions.")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # For each ticker, run agent analyses
    # (Agent scripts are expected to write to data/ folder)
    for ticker_symbol_to_analyze in ticker_list: # Renamed to avoid conflict with 'ticker' in loop
        run_agent_analysis("ai_hedge_fund/buffet_agent.py", ticker_symbol_to_analyze)
        run_agent_analysis("ai_hedge_fund/lynch_agent.py", ticker_symbol_to_analyze)
        if os.path.exists("ai_hedge_fund/wood_agent.py"):
            run_agent_analysis("ai_hedge_fund/wood_agent.py", ticker_symbol_to_analyze)
    
    # Generate risk parameters using risk manager agent
    # risk_manager_agent.py expects a Dict for portfolio, so we dump the model
    risk_params = generate_risk_parameters_for_tickers(ticker_list, portfolio.model_dump(mode='json'))
    print("Generated risk parameters using risk manager agent")
    
    # Collect signals for all tickers
    all_signals: List[AgentSignal] = []
    for ticker_symbol_to_analyze in ticker_list:
        signals_for_ticker = get_agent_signals(ticker_symbol_to_analyze, data_folder="data")
        all_signals.extend(signals_for_ticker)
    
    print(f"Collected {len(all_signals)} signals from agents")
    
    # Make portfolio decisions
    # portfolio_manager_agent.py expects a Dict for portfolio, so we dump the model
    decisions: PortfolioManagerOutput = make_portfolio_decisions(
        signals=all_signals,
        risk_params=risk_params,
        current_portfolio=portfolio.model_dump(mode='json'), 
        target_tickers=ticker_list
    )
    
    # Display portfolio decisions
    print("\n--- Portfolio Decisions ---")
    if decisions.decisions:
        for ticker_symbol_decision, decision_details in decisions.decisions.items():
            print(f"{ticker_symbol_decision}: {str(decision_details.action).upper()}, {decision_details.quantity} shares, confidence: {decision_details.confidence:.2f}")
            print(f"  Reasoning: {decision_details.reasoning}")
    else:
        print("No decisions made.")
    
    if decisions.overall_reasoning:
        print(f"\nOverall reasoning: {decisions.overall_reasoning}")
    
    # Save decisions with timestamp
    decisions_file = save_decisions(decisions, ticker_list)
    
    # Execute trades if requested
    if args.execute:
        print("\n--- Executing Trades ---")
        current_prices = get_current_prices(ticker_list)
        updated_portfolio = update_portfolio_with_decisions(portfolio, decisions, current_prices)
        save_portfolio(updated_portfolio, args.portfolio)
        print(f"Updated portfolio saved to {args.portfolio}. Now has {len(updated_portfolio.transactions)} total transactions.")
    else:
        print("\nTrades simulated only. Run with --execute to update portfolio.")

if __name__ == "__main__":
    main() 