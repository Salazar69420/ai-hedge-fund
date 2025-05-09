from typing import List, Dict, Any
from ..core.data_structures import (
    AgentSignal,
    PortfolioRiskParameters,
    PortfolioDecision,
    PortfolioManagerOutput,
    TickerRiskParameters,
)
from dotenv import load_dotenv
load_dotenv()

# Placeholder for actual LLM call utility
def call_llm_for_portfolio_decision(
    prompt: str,
    # llm_config: Dict[str, Any] # Model name, API keys, etc.
) -> Dict[str, Any]:
    """
    Improved stub for a function that would call an LLM and parse its response.
    This version tries to make intelligent decisions based on the signals in the prompt.
    """
    print(f"LLM Prompt (stubbed):\n{prompt}\n")
    
    # Extract tickers from the prompt
    tickers_in_prompt = []
    if "Tickers to Consider for Decisions" in prompt:
        try:
            tickers_section = prompt.split("Tickers to Consider for Decisions")[1].split("\n")[1].strip()
            if tickers_section != "None specified.":
                tickers_in_prompt = [t.strip() for t in tickers_section.split(',')]
        except Exception as e:
            print(f"Error extracting tickers: {e}")
            
    # Extract signals for each ticker
    signals_by_ticker = {}
    for line in prompt.split("\n"):
        if "Agent:" in line and "Ticker:" in line and "Signal:" in line:
            try:
                parts = line.split(", ")
                agent = parts[0].replace("Agent:", "").strip()
                ticker = parts[1].replace("Ticker:", "").strip()
                signal_type = parts[2].replace("Signal:", "").strip()
                confidence_str = parts[3].replace("Confidence:", "").strip()
                confidence = float(confidence_str)
                
                # Extract target price and stop loss if available
                target_price = None
                stop_loss_price = None
                for part in parts:
                    if "Target Price:" in part:
                        try:
                            target_price = float(part.replace("Target Price:", "").replace("$", "").strip())
                        except:
                            pass
                    if "Stop-Loss:" in part:
                        try:
                            stop_loss_price = float(part.replace("Stop-Loss:", "").replace("$", "").strip())
                        except:
                            pass
                
                if ticker not in signals_by_ticker:
                    signals_by_ticker[ticker] = []
                    
                signals_by_ticker[ticker].append({
                    "agent": agent,
                    "signal_type": signal_type,
                    "confidence": confidence,
                    "target_price": target_price,
                    "stop_loss_price": stop_loss_price
                })
            except Exception as e:
                print(f"Error parsing signal line: {e}")
    
    # Extract current prices from the portfolio section
    current_prices = {}
    for line in prompt.split("\n"):
        if "Current Prices:" in line:
            try:
                price_line = line.replace("Current Prices:", "").strip()
                price_items = price_line.split(", ")
                for item in price_items:
                    if ":" in item:
                        ticker, price_str = item.split(":")
                        ticker = ticker.strip()
                        price = float(price_str.replace("$", "").strip())
                        current_prices[ticker] = price
            except Exception as e:
                print(f"Error parsing current prices: {e}")
    
    # Make decisions for each ticker based on the signals
    decisions = {}
    overall_market_sentiment = "neutral"  # Default sentiment
    
    for ticker in tickers_in_prompt:
        # Use signals if available, otherwise default to hold
        current_price = current_prices.get(ticker, 100.0)  # Default price if not found
        avg_target_price = None
        avg_stop_loss = None
        
        if ticker in signals_by_ticker:
            ticker_signals = signals_by_ticker[ticker]
            
            # Calculate weighted sentiment
            bullish_score = sum(s["confidence"] for s in ticker_signals if s["signal_type"] == "bullish")
            bearish_score = sum(s["confidence"] for s in ticker_signals if s["signal_type"] == "bearish")
            neutral_score = sum(s["confidence"] for s in ticker_signals if s["signal_type"] == "neutral")
            
            # Calculate average target price and stop loss from signals
            target_prices = [s["target_price"] for s in ticker_signals if s["target_price"] is not None]
            stop_losses = [s["stop_loss_price"] for s in ticker_signals if s["stop_loss_price"] is not None]
            
            if target_prices:
                avg_target_price = sum(target_prices) / len(target_prices)
            
            if stop_losses:
                avg_stop_loss = sum(stop_losses) / len(stop_losses)
            
            total_score = bullish_score + bearish_score + neutral_score
            if total_score == 0:
                # No signal strength, default to hold
                action = "hold"
                quantity = 0
                confidence = 0.5
                reasoning = f"No clear signal for {ticker}"
            else:
                # Normalize scores
                bullish_normalized = bullish_score / total_score
                bearish_normalized = bearish_score / total_score
                neutral_normalized = neutral_score / total_score
                
                # Decision logic - simplified for the stub
                if bullish_normalized > 0.6:
                    action = "buy"
                    quantity = 10  # Placeholder - would be calculated based on cash/risk in real implementation
                    confidence = bullish_normalized
                    reasoning = f"Strong bullish signals for {ticker} ({bullish_normalized:.2f} bullish score)"
                elif bearish_normalized > 0.6:
                    action = "sell"
                    quantity = 5  # Placeholder 
                    confidence = bearish_normalized
                    reasoning = f"Strong bearish signals for {ticker} ({bearish_normalized:.2f} bearish score)"
                else:
                    action = "hold"
                    quantity = 0
                    confidence = max(bullish_normalized, bearish_normalized, neutral_normalized)
                    reasoning = f"Mixed signals for {ticker} (bullish: {bullish_normalized:.2f}, bearish: {bearish_normalized:.2f}, neutral: {neutral_normalized:.2f})"
        else:
            # No signals for this ticker
            action = "hold"
            quantity = 0
            confidence = 0.5
            reasoning = f"No signals available for {ticker}"
            
        decisions[ticker] = {
            "action": action,
            "quantity": quantity,
            "confidence": confidence,
            "reasoning": reasoning,
            "current_price": current_price,
            "target_price": avg_target_price,
            "stop_loss_price": avg_stop_loss
        }
    
    # If no decisions were made, provide a fallback
    if not decisions:
        decisions["DEFAULT_TICKER"] = {
            "action": "hold", 
            "quantity": 0, 
            "confidence": 0.0, 
            "reasoning": "No specific tickers identified for decision making.",
            "current_price": None,
            "target_price": None,
            "stop_loss_price": None
        }
    
    # Determine overall market sentiment based on all signals
    all_bullish = sum(1 for ticker in signals_by_ticker for signal in signals_by_ticker[ticker] if signal["signal_type"] == "bullish")
    all_bearish = sum(1 for ticker in signals_by_ticker for signal in signals_by_ticker[ticker] if signal["signal_type"] == "bearish")
    all_neutral = sum(1 for ticker in signals_by_ticker for signal in signals_by_ticker[ticker] if signal["signal_type"] == "neutral")
    
    if all_bullish > all_bearish + all_neutral:
        overall_reasoning = "Overall market sentiment is bullish based on analyst signals."
    elif all_bearish > all_bullish + all_neutral:
        overall_reasoning = "Overall market sentiment is bearish based on analyst signals."
    else:
        overall_reasoning = "Mixed market signals with no clear direction. Maintain cautious positioning."

    return {"decisions": decisions, "overall_reasoning": overall_reasoning}


def generate_portfolio_decision_prompt(
    signals: List[AgentSignal],
    risk_params: PortfolioRiskParameters,
    current_portfolio: Dict[str, Any], # Example: {"cash": 100000, "positions": {"AAPL": {"shares": 10, "avg_price": 150}}}
    target_tickers: List[str]
) -> str:
    """
    Generates a prompt for the LLM to make portfolio decisions.
    """
    prompt_lines = [
        "You are a Portfolio Manager for an AI-driven hedge fund.",
        "Your role is to consolidate signals from various analyst agents (considering their stated confidence levels) and risk parameters to make final trading decisions.",
        "Consider the overall market context, individual stock analysis, and portfolio risk management.",
        "Output your decisions in a strict JSON format.",
        "\n--- Current Portfolio State ---",
        f"Cash: ${current_portfolio.get('cash', 0):.2f}",
        "Current Positions:",
    ]
    for ticker, details in current_portfolio.get("positions", {}).items():
        prompt_lines.append(f"  {ticker}: {details.get('shares', 0)} shares @ avg price ${details.get('avg_price', 0):.2f}")

    # Add current prices section (will be populated by real-time data in the real implementation)
    current_prices = get_current_prices(target_tickers)
    prompt_lines.append("\nCurrent Prices:")
    price_strs = []
    for ticker in target_tickers:
        price = current_prices.get(ticker, None)
        if price:
            price_strs.append(f"{ticker}: ${price:.2f}")
    prompt_lines.append(", ".join(price_strs) if price_strs else "Not available")

    prompt_lines.append("\n--- Analyst Agent Signals ---")
    if not signals:
        prompt_lines.append("No specific agent signals provided for this cycle.")
    for signal in signals:
        signal_line = (
            f"Agent: {signal.agent_name}, Ticker: {signal.ticker}, Signal: {signal.signal_type}, "
            f"Confidence: {signal.confidence:.2f}"
        )
        if signal.target_price is not None:
            signal_line += f", Target Price: ${signal.target_price:.2f}"
        if signal.stop_loss_price is not None:
            signal_line += f", Stop-Loss: ${signal.stop_loss_price:.2f}"
        signal_line += f", Reasoning: {signal.reasoning or 'N/A'}"
        prompt_lines.append(signal_line)

    prompt_lines.append("\n--- Risk Management Parameters ---")
    prompt_lines.append(f"Cash Buffer Target: {risk_params.cash_buffer_percentage*100:.2f}%")
    if risk_params.total_portfolio_max_risk_exposure:
        prompt_lines.append(f"Max Portfolio Risk Exposure: ${risk_params.total_portfolio_max_risk_exposure:.2f}")
    for ticker, params in risk_params.ticker_specific_parameters.items():
        prompt_lines.append(
            f"Ticker {ticker}: Max Position Value: ${params.max_position_value or 'N/A'}, "
            f"Max Shares to Buy: {params.max_shares_to_buy or 'N/A'}, Max Shares to Short: {params.max_shares_to_short or 'N/A'}"
        )
    
    prompt_lines.append("\n--- Tickers to Consider for Decisions ---")
    prompt_lines.append(", ".join(target_tickers) if target_tickers else "None specified.")

    prompt_lines.append(
        "\n--- INSTRUCTIONS ---"
        "Based on all the above information, provide your trading decisions for the specified target tickers."
        "For each ticker, provide: action ('buy', 'sell', 'short', 'cover', 'hold'), quantity (integer), confidence (a float between 0.0 and 1.0 representing your conviction in this specific decision), and reasoning (string)."
        "Additionally, include current_price, target_price (if available), and stop_loss_price (if available)."
        "When deciding, weigh the signals from different agents, paying attention to their individual confidence scores."
        "If no action is to be taken for a ticker, use 'hold' with quantity 0 and an appropriate confidence level."
        "Format the output as a JSON object with a 'decisions' key, which is a dictionary mapping tickers to their decision objects, and an 'overall_reasoning' key for your general market assessment."
        "Example for 'decisions': {'AAPL': {'action': 'buy', 'quantity': 10, 'confidence': 0.8, 'reasoning': 'Strong bullish signals with high agent confidence, and favorable risk profile.', 'current_price': 150.0, 'target_price': 180.0, 'stop_loss_price': 140.0}}"
    )
    return "\n".join(prompt_lines)


def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Placeholder function to get current prices for tickers.
    In a real implementation, this would fetch real-time prices.
    """
    import os
    import requests
    import time
    
    prices = {}
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    
    if not api_key or api_key == "demo":
        print("Warning: Using dummy prices because no Alpha Vantage API key provided")
        for ticker in tickers:
            prices[ticker] = 100.0  # Default price
        return prices
    
    for ticker in tickers:
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
            response = requests.get(url)
            data = response.json()
            
            price_str = data.get("Global Quote", {}).get("05. price", "0")
            price = float(price_str) if price_str else 0
            
            if price > 0:
                prices[ticker] = price
            else:
                print(f"Warning: Could not get price for {ticker}, using default")
                prices[ticker] = 100.0
                
        except Exception as e:
            print(f"Error getting price for {ticker}: {e}")
            prices[ticker] = 100.0
            
        # Alpha Vantage has rate limits, so add a small delay
        time.sleep(0.2)
    
    return prices


def make_portfolio_decisions(
    signals: List[AgentSignal],
    risk_params: PortfolioRiskParameters,
    current_portfolio: Dict[str, Any], # Define this structure more formally later
    target_tickers: List[str],
    # llm_config: Dict[str, Any] # To be added for actual LLM calls
) -> PortfolioManagerOutput:
    """
    Orchestrates the generation of portfolio decisions.
    1. Generates a prompt based on inputs.
    2. Calls an LLM (currently stubbed) to get decisions.
    3. Parses LLM response and returns structured PortfolioManagerOutput.
    """
    # Ensure target_tickers is not empty for the stub to work, or handle appropriately
    if not target_tickers:
        # In a real system, might decide based on all tickers in signals or portfolio
        # For stub, if no target_tickers, we'll return empty decisions.
         return PortfolioManagerOutput(decisions={}, overall_reasoning="No target tickers specified for portfolio decisions.")

    prompt = generate_portfolio_decision_prompt(signals, risk_params, current_portfolio, target_tickers)

    # In a real implementation, you would pass llm_config here
    llm_response_dict = call_llm_for_portfolio_decision(prompt)

    # Parse and validate the LLM response into Pydantic models
    parsed_decisions: Dict[str, PortfolioDecision] = {}
    for ticker_symbol, decision_data in llm_response_dict.get("decisions", {}).items():
        try:
            # Ensure the ticker from decision_data is added to PortfolioDecision
            parsed_decisions[ticker_symbol] = PortfolioDecision(
                ticker=ticker_symbol, # Add ticker here
                action=decision_data.get("action"),
                quantity=decision_data.get("quantity"),
                confidence=decision_data.get("confidence"),
                reasoning=decision_data.get("reasoning"),
                current_price=decision_data.get("current_price"),
                target_price=decision_data.get("target_price"),
                stop_loss_price=decision_data.get("stop_loss_price")
            )
        except Exception as e:
            print(f"Error parsing decision for {ticker_symbol}: {e}. Skipping.")
            # Fallback or logging if a decision for a ticker is malformed
            # For the stub, we might add a default hold
            parsed_decisions[ticker_symbol] = PortfolioDecision(
                ticker=ticker_symbol,
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning=f"Error parsing LLM response for {ticker_symbol}, defaulting to hold."
            )


    overall_reasoning = llm_response_dict.get("overall_reasoning", "No overall reasoning provided by LLM.")

    return PortfolioManagerOutput(
        decisions=parsed_decisions,
        overall_reasoning=overall_reasoning
    )

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    print("--- Testing Portfolio Manager Agent ---")
    
    # Sample data (mimicking what other components would provide)
    sample_signals = [
        AgentSignal(agent_name="LynchAgent", ticker="AAPL", signal_type="bullish", confidence=0.8, reasoning="Strong earnings growth.", target_price=185.0, stop_loss_price=155.0),
        AgentSignal(agent_name="WoodAgent", ticker="TSLA", signal_type="bullish", confidence=0.9, reasoning="Innovation in battery tech.", target_price=950.0, stop_loss_price=750.0),
        AgentSignal(agent_name="BuffetAgent", ticker="MSFT", signal_type="neutral", confidence=0.6, reasoning="Fairly valued, holding.", target_price=350.0, stop_loss_price=300.0),
        AgentSignal(agent_name="LynchAgent", ticker="MSFT", signal_type="bullish", confidence=0.7, reasoning="Cloud segment expansion.", target_price=370.0, stop_loss_price=310.0)
    ]

    sample_risk_params = PortfolioRiskParameters(
        total_portfolio_max_risk_exposure=50000.0,
        cash_buffer_percentage=0.1,
        ticker_specific_parameters={
            "AAPL": TickerRiskParameters(ticker="AAPL", max_position_value=20000.0, max_shares_to_buy=100),
            "TSLA": TickerRiskParameters(ticker="TSLA", max_position_value=15000.0, max_shares_to_buy=50),
            "MSFT": TickerRiskParameters(ticker="MSFT", max_position_value=25000.0, max_shares_to_buy=80),
        }
    )

    sample_portfolio = {
        "cash": 80000.0,
        "positions": {
            "AAPL": {"shares": 50, "avg_price": 150.0},
            "GOOG": {"shares": 20, "avg_price": 2500.0} # A position not in current signals/risk_params
        }
    }
    
    sample_target_tickers = ["AAPL", "TSLA", "MSFT", "NVDA"] # NVDA not in signals/risk params, to test robustness
    
    # Make portfolio decisions
    decisions = make_portfolio_decisions(
        signals=sample_signals,
        risk_params=sample_risk_params,
        current_portfolio=sample_portfolio,
        target_tickers=sample_target_tickers
    )
    
    # Display the results
    print("\n--- Portfolio Manager Output ---")
    print(decisions.model_dump_json(indent=2))
    
    # Test with no target tickers
    print("\n--- Testing with no target tickers ---")
    empty_decisions = make_portfolio_decisions(
        signals=sample_signals,
        risk_params=sample_risk_params,
        current_portfolio=sample_portfolio,
        target_tickers=[]
    )
    print(empty_decisions.model_dump_json(indent=2))
    
    # Test with no signals
    print("\n--- Testing with no signals ---")
    no_signal_decisions = make_portfolio_decisions(
        signals=[],
        risk_params=sample_risk_params,
        current_portfolio=sample_portfolio,
        target_tickers=sample_target_tickers
    )
    print(no_signal_decisions.model_dump_json(indent=2)) 