import os
import subprocess
import json
from datetime import datetime, timedelta
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv()

# Assuming backtester.py is in the ai_hedge_fund directory or a similar level
# Adjust these imports if your directory structure is different or if you encounter ModuleNotFoundError
try:
    from .functions import fetch_historical_time_series, get_last_n_years_data, get_alpha_vantage_api_key
    from .agents.risk_manager_agent import generate_risk_parameters_for_tickers
    from .agents.portfolio_manager_agent import make_portfolio_decisions
    from .agents.agent_signal_extractor import get_agent_signals
    from .core.data_structures import (
        AgentSignal, PortfolioRiskParameters, TickerRiskParameters,
        PortfolioManagerOutput, CurrentPortfolio, PortfolioPosition, Transaction, PortfolioDecision
    )
except ImportError:
    print("Error: Could not import necessary modules. Make sure backtester.py is in the correct directory (e.g., ai_hedge_fund) and that your PYTHONPATH is set up if needed.")
    print("Attempting imports assuming script is run from parent of ai_hedge_fund...")
    from ai_hedge_fund.functions import fetch_historical_time_series, get_last_n_years_data, get_alpha_vantage_api_key
    from ai_hedge_fund.agents.risk_manager_agent import generate_risk_parameters_for_tickers
    from ai_hedge_fund.agents.portfolio_manager_agent import make_portfolio_decisions
    from ai_hedge_fund.agents.agent_signal_extractor import get_agent_signals
    from ai_hedge_fund.core.data_structures import (
        AgentSignal, PortfolioRiskParameters, TickerRiskParameters,
        PortfolioManagerOutput, CurrentPortfolio, PortfolioPosition, Transaction, PortfolioDecision
    )

DATA_FOLDER = "data" # For agent outputs and historical summaries
DECISIONS_HISTORY_FOLDER = os.path.join(DATA_FOLDER, "backtest_decisions_history")
HISTORICAL_PRICE_SUMMARY_FOLDER = os.path.join(DATA_FOLDER, "historical_price_summaries")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DECISIONS_HISTORY_FOLDER, exist_ok=True)
os.makedirs(HISTORICAL_PRICE_SUMMARY_FOLDER, exist_ok=True)

# Environment variables for API keys used by agents
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', get_alpha_vantage_api_key()) # Fallback to function.py's method
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')


def _run_agent_analysis_for_backtest(agent_script_path: str, ticker: str, current_backtest_date_str: str) -> bool:
    """
    Run an agent\'s analysis for a specific ticker and backtest date.
    Passes the path to the historical summary file via an environment variable.
    """
    print(f"Backtester: Running {agent_script_path} for {ticker} (as of {current_backtest_date_str})...")
    env = os.environ.copy()
    env["SYMBOL"] = ticker
    env["BACKTEST_MODE"] = "true" # Indicate to agent it's a backtest run
    env["CURRENT_BACKTEST_DATE"] = current_backtest_date_str # Make date available if needed

    # Agents will look for this summary file
    historical_summary_file = os.path.join(HISTORICAL_PRICE_SUMMARY_FOLDER, f"{ticker}_historical_price_summary.txt")
    env["HISTORICAL_SUMMARY_FILE"] = historical_summary_file

    if ALPHA_VANTAGE_API_KEY:
        env["ALPHA_VANTAGE_API_KEY"] = ALPHA_VANTAGE_API_KEY
    if DEEPSEEK_API_KEY:
        env["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

    # Construct the full path to the agent script if it's relative
    if not os.path.isabs(agent_script_path) and not agent_script_path.startswith("ai_hedge_fund/"):
         # Assuming agent scripts are in ai_hedge_fund/ or ai_hedge_fund/agents/
        base_path = "ai_hedge_fund" 
        if os.path.exists(os.path.join(base_path, agent_script_path)):
            agent_path = os.path.join(base_path, agent_script_path)
        elif os.path.exists(os.path.join(base_path, "agents", agent_script_path)):
            agent_path = os.path.join(base_path, "agents", agent_script_path)
        else:
            agent_path = agent_script_path # Fallback if path is already structured
    else:
        agent_path = agent_script_path

    # Check if the agent path needs to be prefixed with the module for `python -m`
    # This is a common way to run Python modules/scripts within a package.
    # For simplicity, we'll directly execute the .py file here.
    # If agents are part of a package and meant to be run with `python -m ai_hedge_fund.agents.some_agent`,
    # this would need adjustment. Current agent structure seems to be standalone scripts.

    try:
        result = subprocess.run(
            ["python", agent_path],
            env=env,
            text=True,
            capture_output=True,
            check=False  # Don't raise exception on non-zero exit, handle manually
        )
        if result.returncode != 0:
            print(f"Backtester: Error running {agent_script_path} for {ticker}:")
            print(f"  Stdout: {result.stdout}")
            print(f"  Stderr: {result.stderr}")
            return False
        # print(f"Backtester: Stdout from {agent_script_path} for {ticker}:\\n{result.stdout}") # Optional: for debugging agent output
        print(f"Backtester: ✓ {agent_script_path} analysis complete for {ticker} (as of {current_backtest_date_str})")
        return True
    except FileNotFoundError:
        print(f"Backtester: Error - Agent script not found: {agent_path}")
        print("Please ensure agent paths are correct (e.g., 'agents/buffet_agent.py' or 'buffet_agent.py' if in the same directory as this script or if ai_hedge_fund is in PYTHONPATH).")
        return False
    except Exception as e:
        print(f"Backtester: An unexpected error occurred running {agent_path}: {e}")
        return False


def _prepare_historical_summary_for_agent(symbol: str, current_backtest_date: datetime, years_of_history: int = 5) -> bool:
    """
    Fetches 5 years of weekly data ending at current_backtest_date,
    summarizes it, and saves it to a file for the agent.
    """
    print(f"Backtester: Preparing historical summary for {symbol} as of {current_backtest_date.strftime('%Y-%m-%d')}...")
    # Fetch data up to the current_backtest_date. Alpha Vantage TIME_SERIES functions return data ending before the query date.
    # So, fetching "full" and then filtering is more reliable for getting data *up to* a point.
    
    # To get data *up to* current_backtest_date, we might need to make sure our filtering logic is robust.
    # The get_last_n_years_data filters based on datetime.now(). We need to adapt this or filter manually.

    raw_weekly_data = fetch_historical_time_series(symbol, series_type="weekly", outputsize="full")
    if not raw_weekly_data:
        print(f"Backtester: Could not fetch raw weekly data for {symbol} to create summary.")
        return False

    series_key = "Weekly Adjusted Time Series"
    time_series = raw_weekly_data.get(series_key)
    if not time_series:
        print(f"Backtester: Missing '{series_key}' in historical data for {symbol}.")
        return False

    # Filter data for the N years *before or on* current_backtest_date
    cutoff_date_start = current_backtest_date - timedelta(days=years_of_history * 365.25)
    
    filtered_data_points = []
    # Sort dates to ensure correct processing if not already sorted (AlphaVantage usually is reverse chronological)
    sorted_dates = sorted(time_series.keys(), reverse=True)

    for date_str in sorted_dates:
        try:
            data_date = datetime.strptime(date_str, "%Y-%m-%d")
            if data_date <= current_backtest_date and data_date >= cutoff_date_start:
                entry = time_series[date_str]
                filtered_data_points.append({
                    "date": date_str,
                    "open": float(entry.get("1. open", 0)),
                    "high": float(entry.get("2. high", 0)),
                    "low": float(entry.get("3. low", 0)),
                    "close": float(entry.get("4. close", 0)),
                    "adjusted_close": float(entry.get("5. adjusted close", 0)),
                    "volume": int(entry.get("6. volume", 0))
                })
            elif data_date < cutoff_date_start:
                break # Past the N-year window
        except ValueError:
            print(f"Backtester: Warning - Could not parse date \'{date_str}\' in historical summary. Skipping.")
            continue
    
    if not filtered_data_points:
        summary_content = f"No weekly data found for {symbol} within the last {years_of_history} years up to {current_backtest_date.strftime('%Y-%m-%d')}."
    else:
        # Create a simple summary
        first_date = filtered_data_points[-1]['date'] # Oldest
        last_date = filtered_data_points[0]['date'] # Newest
        
        prices = [p['adjusted_close'] for p in filtered_data_points]
        volumes = [p['volume'] for p in filtered_data_points]

        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices) if prices else 0
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        start_price = filtered_data_points[-1]['adjusted_close']
        end_price = filtered_data_points[0]['adjusted_close']
        overall_change_pct = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0

        summary_content = (
            f"Historical Price Summary for {symbol} ({years_of_history} years weekly data, ending ~{current_backtest_date.strftime('%Y-%m-%d')}):\n"
            f"Data Range: {first_date} to {last_date}\n"
            f"Adjusted Close Price Range: ${min_price:.2f} - ${max_price:.2f}\n"
            f"Average Adjusted Close Price: ${avg_price:.2f}\n"
            f"Average Volume: {avg_volume:,.0f}\n"
            f"Overall Change: {overall_change_pct:.2f}%\n"
            f"Latest {len(filtered_data_points)} weekly data points provided (newest first):\n"
        )
        # Add a few recent data points to the summary for context
        for point in filtered_data_points[:min(5, len(filtered_data_points))]: # Show up to 5 most recent
            summary_content += f"  {point['date']}: Close=${point['adjusted_close']:.2f}, Vol={point['volume']:,}\n"

    summary_file_path = os.path.join(HISTORICAL_PRICE_SUMMARY_FOLDER, f"{symbol}_historical_price_summary.txt")
    try:
        with open(summary_file_path, "w") as f:
            f.write(summary_content)
        print(f"Backtester: ✓ Historical summary for {symbol} saved to {summary_file_path}")
        return True
    except IOError as e:
        print(f"Backtester: Error saving historical summary for {symbol}: {e}")
        return False


class Backtester:
    def __init__(self, tickers: list[str], start_date: str, end_date: str,
                 initial_capital: float = 100000.0,
                 agent_scripts: dict = None,
                 decision_frequency: str = "daily"): # "daily" or "weekly"
        self.tickers = [t.upper() for t in tickers]
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        self.decision_frequency = decision_frequency.lower()

        self.agent_scripts = agent_scripts or {
            "buffet": "buffet_agent.py", 
            "lynch": "lynch_agent.py",
            "wood": "wood_agent.py"
        }
        # Filter out agents not found
        valid_agent_scripts = {}
        for name, script_path in self.agent_scripts.items():
            # Attempt to construct a path relative to where backtester.py might be (e.g. ai_hedge_fund/)
            potential_path_from_module_root = os.path.join(os.path.dirname(__file__), script_path)
            if os.path.exists(script_path): # Check if path is absolute or directly accessible
                valid_agent_scripts[name] = script_path
            elif os.path.exists(potential_path_from_module_root):
                 valid_agent_scripts[name] = potential_path_from_module_root
            else:
                print(f"Warning: Agent script '{script_path}' for agent '{name}' not found. It will be skipped.")
        self.agent_scripts = valid_agent_scripts


        self.portfolio = CurrentPortfolio(cash=initial_capital, positions={}, transactions=[])
        self.portfolio_history = [] # List of (date, portfolio_value)
        self.trade_log = [] # List of trade details
        self.all_time_price_data = {} # Store all daily prices: {ticker: {date_str: close_price}}

    def _fetch_all_price_data(self):
        print("Backtester: Pre-fetching all required historical daily price data...")
        for ticker in self.tickers:
            print(f"  Fetching daily data for {ticker}...")
            # Fetch full history and then we'll pick dates as needed
            raw_data = fetch_historical_time_series(ticker, series_type="daily", outputsize="full")
            if raw_data and "Time Series (Daily)" in raw_data:
                self.all_time_price_data[ticker] = {}
                for date_str, daily_data in raw_data["Time Series (Daily)"].items():
                    try:
                        # Ensure date is valid and store close price
                        price_date = datetime.strptime(date_str, "%Y-%m-%d")
                        self.all_time_price_data[ticker][date_str] = float(daily_data.get("4. close", 0.0))
                    except ValueError:
                        print(f"Warning: Invalid date format \'{date_str}\' in price data for {ticker}")
                        continue
                print(f"  ✓ Fetched {len(self.all_time_price_data[ticker])} daily data points for {ticker}")
            else:
                print(f"  ✗ Failed to fetch or parse daily data for {ticker}. This ticker may be untradable.")
            time.sleep(1) # Be nice to Alpha Vantage if using free tier during multi-ticker fetch

        # Verify we have data for the backtest period
        current_date = self.start_date
        while current_date <= self.end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            for ticker in self.tickers:
                if ticker in self.all_time_price_data and date_str not in self.all_time_price_data[ticker]:
                    # Try to find nearest available previous day's price if exact match is missing (e.g. holiday)
                    temp_date = current_date - timedelta(days=1)
                    found_fallback = False
                    for i in range(7): # Check back up to 7 days
                        temp_date_str = temp_date.strftime("%Y-%m-%d")
                        if temp_date_str in self.all_time_price_data.get(ticker, {}):
                            self.all_time_price_data[ticker][date_str] = self.all_time_price_data[ticker][temp_date_str]
                            # print(f"Backtester: Using fallback price for {ticker} on {date_str} (from {temp_date_str})")
                            found_fallback = True
                            break
                        temp_date -= timedelta(days=1)
                    if not found_fallback:
                         print(f"Warning: Missing price data for {ticker} on {date_str} and no recent fallback. Decisions for this day might be affected.")
                         # Provide a default or skip? For now, let decision logic handle missing price.
                         # self.all_time_price_data.get(ticker, {})[date_str] = 0.0 # Or some other placeholder

            current_date += timedelta(days=1)
        print("Backtester: ✓ Price data pre-fetching complete.")


    def _get_current_prices_for_date(self, date_obj: datetime) -> Dict[str, float]:
        current_prices = {}
        date_str = date_obj.strftime("%Y-%m-%d")
        for ticker in self.tickers:
            if ticker not in self.all_time_price_data or not self.all_time_price_data[ticker]:
                print(f"Warning: No price data loaded for {ticker}. Cannot get price for {date_str}.")
                current_prices[ticker] = 0.0 # Or raise error
                continue

            price = self.all_time_price_data[ticker].get(date_str)
            if price is None:
                # Fallback: find most recent price before or on date_str
                ticker_prices = self.all_time_price_data[ticker]
                available_dates = sorted([d for d in ticker_prices.keys() if datetime.strptime(d, "%Y-%m-%d") <= date_obj], reverse=True)
                if available_dates:
                    price = ticker_prices[available_dates[0]]
                    # print(f"Using price from {available_dates[0]} for {ticker} on {date_str}")
                else:
                    print(f"Warning: No price found for {ticker} on or before {date_str}. Defaulting to 0.")
                    price = 0.0
            current_prices[ticker] = price
        return current_prices


    def run_backtest(self):
        print(f"Backtester: Starting backtest from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Decision frequency: {self.decision_frequency}")
        print(f"Analyzing tickers: {', '.join(self.tickers)}")
        print(f"Using agents: {', '.join(self.agent_scripts.keys())}")

        self._fetch_all_price_data()

        current_date = self.start_date
        days_processed = 0
        last_decision_date = None # To manage weekly/monthly decision frequency

        while current_date <= self.end_date:
            current_date_str = current_date.strftime("%Y-%m-%d")
            print(f"\\n--- Processing Date: {current_date_str} ---")

            # Get current market prices for all tickers for this date
            current_market_prices = self._get_current_prices_for_date(current_date)
            if not any(current_market_prices.values()): # If all prices are 0, maybe market closed or bad data
                print(f"Market data potentially unavailable for {current_date_str}. Advancing to next day.")
                current_date += timedelta(days=1)
                continue

            # ----- Decision Making Logic -----
            make_new_decisions_today = False
            if self.decision_frequency == "daily":
                make_new_decisions_today = True
            elif self.decision_frequency == "weekly":
                if last_decision_date is None or (current_date - last_decision_date).days >= 7 or current_date.weekday() == 0: # Monday
                    make_new_decisions_today = True
            # Add "monthly" or other frequencies if needed

            if make_new_decisions_today:
                print(f"Backtester: Making decisions for {current_date_str}...")
                # 1. Prepare historical summaries for agents for the current backtest date
                for ticker in self.tickers:
                    _prepare_historical_summary_for_agent(ticker, current_date)
                
                # 2. Run agent analyses for each ticker
                active_tickers_for_decision = []
                for ticker in self.tickers:
                    # Skip analysis if price is 0 (likely untradable or data issue for this day)
                    if current_market_prices.get(ticker, 0) == 0:
                        print(f"Skipping agent analysis for {ticker} on {current_date_str} due to zero price.")
                        continue
                    
                    # Clear previous agent run data for this ticker from DATA_FOLDER to avoid staleness
                    for agent_name in self.agent_scripts.keys():
                        agent_output_pattern = os.path.join(DATA_FOLDER, f"{ticker}_{agent_name}_*.json")
                        import glob
                        for f_path in glob.glob(agent_output_pattern):
                            try:
                                os.remove(f_path)
                            except OSError as e:
                                print(f"Warning: Could not remove old agent file {f_path}: {e}")
                                
                    # Run agents
                    for agent_name, script_path in self.agent_scripts.items():
                        _run_agent_analysis_for_backtest(script_path, ticker, current_date_str)
                    active_tickers_for_decision.append(ticker)
                
                if not active_tickers_for_decision:
                    print("No active tickers for decision making today.")
                else:
                    # 3. Collect signals
                    all_signals: List[AgentSignal] = []
                    for ticker in active_tickers_for_decision:
                        signals_for_ticker = get_agent_signals(ticker, data_folder=DATA_FOLDER)
                        all_signals.extend(signals_for_ticker)
                    print(f"Backtester: Collected {len(all_signals)} signals from agents.")

                    # 4. Generate risk parameters (using current portfolio state)
                    # The risk manager agent expects portfolio as dict
                    risk_params = generate_risk_parameters_for_tickers(
                        active_tickers_for_decision, self.portfolio.model_dump(mode='json')
                    )
                    print("Backtester: Generated risk parameters.")

                    # 5. Make portfolio decisions
                    decisions: PortfolioManagerOutput = make_portfolio_decisions(
                        signals=all_signals,
                        risk_params=risk_params,
                        current_portfolio=self.portfolio.model_dump(mode='json'),
                        target_tickers=active_tickers_for_decision
                    )
                    print("Backtester: Portfolio decisions made.")
                    if decisions.decisions:
                         for ticker_dec, dec_details in decisions.decisions.items():
                            print(f"  Decision for {ticker_dec}: {str(dec_details.action).upper()} {dec_details.quantity} shares. Confidence: {dec_details.confidence:.2f}. Reason: {dec_details.reasoning}")
                    if decisions.overall_reasoning:
                        print(f"  Overall Reasoning: {decisions.overall_reasoning}")

                    # Save decisions for this backtest iteration
                    self._save_backtest_decisions(decisions, active_tickers_for_decision, current_date_str)

                    last_decision_date = current_date # Mark that decisions were made today

                    # ----- Execute Trades based on decisions -----
                    for ticker, decision_obj in decisions.decisions.items():
                        if ticker not in current_market_prices or current_market_prices[ticker] == 0:
                            print(f"Skipping trade for {ticker}: No valid market price ({current_market_prices.get(ticker)}).")
                            continue

                        current_price = current_market_prices[ticker]
                        action_str = str(decision_obj.action).lower() # Ensure lowercase
                        quantity = int(decision_obj.quantity)

                        if action_str == "buy" and quantity > 0:
                            cost = current_price * quantity
                            if cost <= self.portfolio.cash:
                                self.portfolio.cash -= cost
                                current_pos = self.portfolio.positions.get(ticker)
                                if not current_pos:
                                    self.portfolio.positions[ticker] = PortfolioPosition(ticker=ticker, shares=quantity, avg_price=current_price)
                                else:
                                    total_cost_basis = (current_pos.shares * current_pos.avg_price) + cost
                                    current_pos.shares += quantity
                                    current_pos.avg_price = total_cost_basis / current_pos.shares if current_pos.shares > 0 else 0
                                
                                trx = Transaction(date=current_date_str, ticker=ticker, action="buy", quantity=quantity, price=current_price, total_value=cost)
                                self.portfolio.transactions.append(trx)
                                self.trade_log.append(trx.model_dump())
                                print(f"Executed: BUY {quantity} of {ticker} @ ${current_price:.2f}. Cash: ${self.portfolio.cash:.2f}")
                            else:
                                print(f"Skipped BUY {ticker}: Insufficient cash. Need ${cost:.2f}, have ${self.portfolio.cash:.2f}")
                        
                        elif action_str == "sell" and quantity > 0:
                            current_pos = self.portfolio.positions.get(ticker)
                            if current_pos and current_pos.shares >= quantity:
                                proceeds = current_price * quantity
                                self.portfolio.cash += proceeds
                                
                                realized_pnl = (current_price - current_pos.avg_price) * quantity # Simple PnL
                                current_pos.shares -= quantity
                                
                                if current_pos.shares == 0:
                                    del self.portfolio.positions[ticker]
                                
                                trx = Transaction(date=current_date_str, ticker=ticker, action="sell", quantity=quantity, price=current_price, total_value=proceeds)
                                self.portfolio.transactions.append(trx)
                                self.trade_log.append(trx.model_dump())
                                print(f"Executed: SELL {quantity} of {ticker} @ ${current_price:.2f}. Realized PnL: ${realized_pnl:.2f}. Cash: ${self.portfolio.cash:.2f}")
                            else:
                                shares_owned = current_pos.shares if current_pos else 0
                                print(f"Skipped SELL {ticker}: Insufficient shares. Tried {quantity}, have {shares_owned}")
            else: # Not a decision day
                 print(f"Not a decision day based on frequency ({self.decision_frequency}). Holding positions.")


            # ----- Record Portfolio Value for the day -----
            current_portfolio_value = self.portfolio.cash
            for ticker, position in self.portfolio.positions.items():
                price_for_value = current_market_prices.get(ticker, 0) # Use day's closing price for valuation
                if price_for_value == 0 and position.shares > 0: # If price is 0, use avg_price as fallback for valuation
                    price_for_value = position.avg_price
                    print(f"Warning: Using avg_price for valuation of {ticker} as current market price is 0.")
                current_portfolio_value += position.shares * price_for_value
            
            self.portfolio_history.append({
                "date": current_date, # datetime object
                "portfolio_value": current_portfolio_value,
                "cash": self.portfolio.cash,
                "positions_value": current_portfolio_value - self.portfolio.cash
            })
            print(f"End of Day {current_date_str}: Portfolio Value: ${current_portfolio_value:,.2f} (Cash: ${self.portfolio.cash:,.2f})")

            current_date += timedelta(days=1)
            days_processed += 1
            # Alpha Vantage free tier has rate limits (e.g., 5 calls/min, 100/day)
            # The historical summary call is one per ticker per decision period. Price fetching is once at start.
            # Agent calls might involve API calls if they are not using cached/local data.
            # Add a small delay if making many API calls in a loop.
            # time.sleep(0.5) # If agent calls or other logic makes many API calls per day.

        print(f"\\nBacktest finished. Processed {days_processed} days.")
        self.analyze_performance()

    def _save_backtest_decisions(self, decisions: PortfolioManagerOutput, ticker_list: List[str], current_date_str: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"decisions_backtest_{current_date_str.replace('-', '')}_{timestamp}.json"
        filename = os.path.join(DECISIONS_HISTORY_FOLDER, filename_base)
        
        decision_data_to_save = {
            "backtest_date": current_date_str,
            "run_timestamp": timestamp,
            "analyzed_tickers": ticker_list,
            "portfolio_manager_output": decisions.model_dump(mode='json')
        }
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(decision_data_to_save, f, indent=2)
            print(f"Backtest decisions for {current_date_str} saved to {filename}")
        except IOError as e:
            print(f"Error saving backtest decisions: {e}")


    def analyze_performance(self):
        if not self.portfolio_history:
            print("No portfolio history to analyze.")
            return

        perf_df = pd.DataFrame(self.portfolio_history)
        perf_df.set_index("date", inplace=True)

        print("\\n--- Backtest Performance Summary ---")
        initial_value = self.initial_capital
        final_value = perf_df["portfolio_value"].iloc[-1]
        total_return_pct = ((final_value - initial_value) / initial_value) * 100
        print(f"Initial Portfolio Value: ${initial_value:,.2f}")
        print(f"Final Portfolio Value:   ${final_value:,.2f}")
        print(f"Total Return:            {total_return_pct:.2f}%")

        # More metrics can be added here (Sharpe, Sortino, Max Drawdown, etc.)
        perf_df["daily_return"] = perf_df["portfolio_value"].pct_change()
        
        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity, can be parameterized)
        # Annualized assuming 252 trading days if data is daily
        if not perf_df["daily_return"].empty and perf_df["daily_return"].std() != 0 :
            avg_daily_return = perf_df["daily_return"].mean()
            std_daily_return = perf_df["daily_return"].std()
            # Approximation: if decision_frequency is weekly, there are ~52 periods. Daily ~252.
            annualization_factor = 252 if self.decision_frequency == "daily" else 52 
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(annualization_factor) if std_daily_return else 0
            print(f"Annualized Sharpe Ratio:   {sharpe_ratio:.2f} (approx, rf=0)")
        else:
            print("Sharpe Ratio:            N/A (not enough data or no volatility)")

        # Max Drawdown
        perf_df["rolling_max"] = perf_df["portfolio_value"].cummax()
        perf_df["drawdown"] = (perf_df["portfolio_value"] - perf_df["rolling_max"]) / perf_df["rolling_max"]
        max_drawdown = perf_df["drawdown"].min()
        if max_drawdown is not None and not np.isnan(max_drawdown): # Check for NaN
             print(f"Maximum Drawdown:        {max_drawdown:.2%}")
        else:
             print("Maximum Drawdown:        N/A")


        print(f"Total Trades Made:       {len(self.trade_log)}")

        # Plotting Portfolio Value
        plt.figure(figsize=(12, 6))
        plt.plot(perf_df.index, perf_df["portfolio_value"])
        plt.title(f"Portfolio Value Over Time ({', '.join(self.tickers)})")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.figtext(0.1, 0.01, f"Initial: ${initial_value:,.0f} | Final: ${final_value:,.0f} | Return: {total_return_pct:.2f}% | Max DD: {max_drawdown:.2% if max_drawdown and not np.isnan(max_drawdown) else 'N/A'}", fontsize=10)
        plt.show()

        # Print Trade Log
        print("\\n--- Trade Log ---")
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            print(trade_df.to_string(index=False))
        else:
            print("No trades were made.")

def main():
    parser = argparse.ArgumentParser(description="Run financial agent backtesting.")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of ticker symbols (e.g., AAPL,MSFT).")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for backtest (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date for backtest (YYYY-MM-DD).")
    parser.add_argument("--initial_capital", type=float, default=100000.0, help="Initial capital for the portfolio.")
    parser.add_argument("--decision_frequency", type=str, default="daily", choices=["daily", "weekly"], help="How often to make trading decisions.")
    
    # Optional: Allow specifying agent scripts if they are not in default locations or have different names
    parser.add_argument("--buffet_agent_script", type=str, default="agents/buffet_agent.py", help="Path to Buffet agent script.")
    parser.add_argument("--lynch_agent_script", type=str, default="agents/lynch_agent.py", help="Path to Lynch agent script.")
    parser.add_argument("--wood_agent_script", type=str, default="agents/wood_agent.py", help="Path to Wood agent script.")
    
    args = parser.parse_args()

    tickers_list = [t.strip().upper() for t in args.tickers.split(',')]
    
    agent_scripts_paths = {
        "buffet": args.buffet_agent_script,
        "lynch": args.lynch_agent_script,
        "wood": args.wood_agent_script
    }
    # You might want to add a check here to see if os.environ.get("ALPHA_VANTAGE_API_KEY") is set and not "demo"
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "demo":
        print("WARNING: ALPHA_VANTAGE_API_KEY is not set or is 'demo'. Backtester may have limited data access or hit rate limits quickly.")
    if not DEEPSEEK_API_KEY:
        print("WARNING: DEEPSEEK_API_KEY is not set. LLM-based agents may not function correctly.")


    backtester = Backtester(
        tickers=tickers_list,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        agent_scripts=agent_scripts_paths,
        decision_frequency=args.decision_frequency
    )
    backtester.run_backtest()

if __name__ == "__main__":
    main() 