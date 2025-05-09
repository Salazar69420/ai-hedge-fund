import json
import os
from openai import OpenAI
from typing import Dict, Any, List, Optional
import requests
from ..core.data_structures import PortfolioRiskParameters, TickerRiskParameters
from dotenv import load_dotenv
load_dotenv()

# Configuration - load from environment variables or use defaults
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')  # Replace with actual key
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-39cbd05d862749d8bca6c13f1bc234ee')  # Default from existing files
SYMBOL = os.environ.get('SYMBOL', None)  # For single stock analysis
DATA_FOLDER = "data"
MODEL = "deepseek-chat"

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize API client for LLM
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def get_price_data(ticker: str) -> Dict[str, Any]:
    """Get price data from Alpha Vantage API"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching price data for {ticker}: {response.status_code}")
        return {}

def load_agent_json_data(report_path: str) -> Optional[Dict[str, Any]]:
    """Load agent analysis from a JSON report file."""
    if not os.path.exists(report_path):
        # print(f"Report file not found: {report_path}") # Optional: more verbose logging
        return None
    
    try:
        with open(report_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        # We expect keys like 'recommendation', 'confidence_score', 'reasoning', etc.
        # 'agent' and 'ticker' should also be in the agent's output JSON.
        return data 
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {report_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading report {report_path}: {e}")
        return None

def calculate_risk_parameters(tickers: List[str], portfolio_data: Dict[str, Any]) -> PortfolioRiskParameters:
    """
    Calculate risk parameters for multiple tickers based on portfolio data and market prices.
    Accepts portfolio_data as a dictionary (e.g., from CurrentPortfolio.model_dump()).
    """
    # Calculate total portfolio value
    portfolio_value = portfolio_data.get("cash", 0.0)
    positions_data = portfolio_data.get("positions", {})
    
    for ticker, position_details in positions_data.items():
        # Assuming position_details is a dict with 'shares' and 'avg_price'
        portfolio_value += position_details.get("shares", 0) * position_details.get("avg_price", 0.0)
    
    # Default risk settings
    max_portfolio_risk_exposure = portfolio_value * 0.5  # 50% of portfolio max risk
    cash_buffer = 0.1  # 10% cash buffer
    
    # Calculate ticker-specific risk parameters
    ticker_specific_parameters = {}
    
    for ticker in tickers:
        # Get current price from Alpha Vantage
        price_data = get_price_data(ticker)
        current_price = 0
        
        try:
            current_price = float(price_data.get("Global Quote", {}).get("05. price", 0))
        except (ValueError, TypeError):
            # If price can't be fetched, use position avg_price or default to 100
            position = positions_data.get(ticker, {})
            current_price = position.get("avg_price", 100.0)
        
        # Calculate position value and limits
        max_position_value = portfolio_value * 0.2  # 20% max per position
        current_position = positions_data.get(ticker, {})
        current_shares = current_position.get("shares", 0)
        current_value = current_shares * (current_position.get("avg_price", 0) or current_price)
        
        # Calculate remaining risk capacity
        remaining_value = max(0, max_position_value - current_value)
        max_shares_to_buy = int(remaining_value / current_price) if current_price > 0 else 0
        
        # For stocks we already own more than the limit, don't allow buying more
        if current_value > max_position_value:
            max_shares_to_buy = 0
        
        # Calculate max shares to short - using 50% of max position value for shorts
        max_shares_to_short = int((max_position_value * 0.5) / current_price) if current_price > 0 else 0
        
        # Create TickerRiskParameters object
        ticker_specific_parameters[ticker] = TickerRiskParameters(
            ticker=ticker,
            max_position_value=max_position_value,
            max_shares_to_buy=max_shares_to_buy,
            max_shares_to_short=max_shares_to_short
        )
    
    return PortfolioRiskParameters(
        total_portfolio_max_risk_exposure=max_portfolio_risk_exposure,
        cash_buffer_percentage=cash_buffer,
        ticker_specific_parameters=ticker_specific_parameters
    )

def analyze_ticker_risk_with_llm(ticker: str) -> Dict[str, Any]:
    """
    Use LLM to analyze risk for a specific ticker based on agent JSON reports.
    Outputs a dictionary with keys like 'risk_level', 'max_position_percentage_adjustment', 'key_risk_summary'.
    """
    # Define file paths for agent JSON reports
    buffett_report_path = os.path.join(DATA_FOLDER, f"{ticker}_buffett_analysis.json")
    lynch_report_path = os.path.join(DATA_FOLDER, f"{ticker}_lynch_analysis.json")
    wood_report_path = os.path.join(DATA_FOLDER, f"{ticker}_wood_analysis.json")
    
    # Load analysis data from agent JSON reports
    buffett_data = load_agent_json_data(buffett_report_path)
    lynch_data = load_agent_json_data(lynch_report_path)
    wood_data = load_agent_json_data(wood_report_path)
    
    # Prepare summaries of agent reports for the LLM prompt
    def format_agent_data_for_prompt(agent_name: str, data: Optional[Dict[str, Any]]) -> str:
        if not data:
            return f"{agent_name} Agent Report for {ticker}: Not available or failed to load.\n"
        # Extract relevant fields. Add more as needed by the risk manager's LLM.
        recommendation = data.get('recommendation', 'N/A')
        confidence = data.get('confidence_score', 'N/A')
        reasoning = data.get('reasoning', 'No reasoning provided.')
        target = data.get('target_price')
        stop_loss = data.get('stop_loss_price')
        
        details = f"- Recommendation: {recommendation} (Confidence: {confidence})\n"
        details += f"- Reasoning: {reasoning}\n"
        if target is not None: details += f"- Target Price: {target}\n"
        if stop_loss is not None: details += f"- Stop-Loss Price: {stop_loss}\n"
        # Add agent-specific fields if they exist (Lynch/Wood)
        if 'key_growth_drivers' in data: details += f"- Key Growth Drivers: {data['key_growth_drivers']}\n"
        if 'risk_factors' in data: details += f"- Risk Factors: {data['risk_factors']}\n"
        if 'disruptive_theme' in data: details += f"- Disruptive Theme: {data['disruptive_theme']}\n"
        if 'catalysts' in data: details += f"- Catalysts: {data['catalysts']}\n"
        return f"{agent_name} Agent Report for {ticker}:\n{details}\n"

    buffett_prompt_summary = format_agent_data_for_prompt("Buffett", buffett_data)
    lynch_prompt_summary = format_agent_data_for_prompt("Lynch", lynch_data)
    wood_prompt_summary = format_agent_data_for_prompt("Wood", wood_data)

    context = (
        "You are a financial Risk Manager. Your task is to synthesize insights from multiple investment agent reports "
        "(which were generated based on financial data provided to them) to provide a risk assessment for a specific stock. "
        "Focus solely on the information within these reports."
    )
    
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": f"""Synthesize the following agent reports for {ticker} to assess its risk profile.

{buffett_prompt_summary}
{lynch_prompt_summary}
{wood_prompt_summary}

Based *only* on the provided agent reports, output a JSON object with the following keys:
- "risk_level": Your assessment of the overall risk (e.g., "Low", "Medium", "High") based on the synthesis of agent reports.
- "confidence_in_risk_assessment": Your confidence (0.0 to 1.0) in this risk_level assessment.
- "key_risk_summary": A brief bullet-point summary of the key risks highlighted or implied by the agent reports.
- "max_position_percentage_suggestion": A float suggesting a maximum percentage of a portfolio to allocate to this stock based on its risk profile (e.g., 0.1 for 10%). This should be derived from the synthesized risk, not a fixed rule.
- "overall_synthesis_notes": Any important notes on contradictions, consensus, or lack of clarity among agent reports.

**Important Constraints:**
- All outputs **must be derived *solely* from the content of the agent reports.**
- Do **not** introduce external information, market sentiment, or your own financial analysis beyond interpreting these specific reports.
- If reports are contradictory, reflect this in "overall_synthesis_notes".
- Output *only* the JSON object and nothing else.
Example JSON Output:
{{
  "risk_level": "Medium",
  "confidence_in_risk_assessment": 0.75,
  "key_risk_summary": [
    "High valuation if growth targets are not met (Lynch)",
    "Dependent on continued innovation (Wood)",
    "Market sentiment can be volatile for this sector (implied)"
  ],
  "max_position_percentage_suggestion": 0.15,
  "overall_synthesis_notes": "Buffett report is conservative, Wood is very bullish. Overall, a balanced approach is warranted if growth materializes."
}}
"""}
    ]
    
    llm_output_json_str = "{}" # Default to empty JSON string
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"} # Request JSON output
        )
        llm_output_json_str = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM for {ticker} risk analysis: {e}")
        # Fallback if LLM call fails
        return {
            "ticker": ticker,
            "risk_level": "Medium", 
            "confidence_in_risk_assessment": 0.5,
            "key_risk_summary": ["LLM call failed, risk assessment is a default."],
            "max_position_percentage_suggestion": 0.1, # Conservative default
            "overall_synthesis_notes": "LLM call failed."
        }
    
    # Save the raw LLM analysis to file for debugging/auditing
    output_file = os.path.join(DATA_FOLDER, f"{ticker}_Risk_Manager_LLM_synthesis.json")
    try:
        # Save the received JSON string directly
        with open(output_file, 'w', encoding='utf-8') as f:
            # Attempt to pretty-print if it's valid JSON, otherwise save as is
            try: 
                json.dump(json.loads(llm_output_json_str), f, indent=2)
            except json.JSONDecodeError:
                f.write(llm_output_json_str)
        print(f"Risk LLM synthesis saved to {output_file}")
    except Exception as e:
        print(f"Error saving risk LLM synthesis for {ticker} to {output_file}: {e}")

    # Parse the LLM JSON output
    try:
        parsed_llm_response = json.loads(llm_output_json_str)
        # Add ticker for completeness, though it should be in the context already
        parsed_llm_response['ticker'] = ticker 
        return parsed_llm_response
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM risk analysis for {ticker}: {e}. LLM output: {llm_output_json_str}")
        return { # Fallback if JSON parsing fails
            "ticker": ticker,
            "risk_level": "Medium", 
            "confidence_in_risk_assessment": 0.5,
            "key_risk_summary": ["Failed to parse LLM JSON output for risk assessment."],
            "max_position_percentage_suggestion": 0.1, # Conservative default
            "overall_synthesis_notes": f"LLM output was not valid JSON: {llm_output_json_str[:200]}..."
        }

def generate_risk_parameters_for_tickers(tickers: List[str], portfolio_data: Dict[str, Any]) -> PortfolioRiskParameters:
    """
    Generate risk parameters for all tickers by combining quantitative analysis with LLM insights.
    Accepts portfolio_data as a dictionary (e.g., from CurrentPortfolio.model_dump()).
    """
    # Start with quantitative risk parameters based on portfolio
    quantitative_risk_params = calculate_risk_parameters(tickers, portfolio_data)
    
    # Enhance with LLM analysis for each ticker
    for ticker in tickers:
        print(f"--- Performing LLM Risk Analysis for {ticker} ---")
        llm_risk_synthesis = analyze_ticker_risk_with_llm(ticker)
        print(f"LLM Risk Synthesis for {ticker}: {llm_risk_synthesis.get('risk_level', 'N/A')}, Max Pos Suggestion: {llm_risk_synthesis.get('max_position_percentage_suggestion', 'N/A')}")
        
        # Adjust quantitative risk parameters based on LLM synthesis
        if ticker in quantitative_risk_params.ticker_specific_parameters:
            ticker_params = quantitative_risk_params.ticker_specific_parameters[ticker]
            
            # Use the max_position_percentage_suggestion from LLM if available and valid
            llm_max_pos_perc = llm_risk_synthesis.get('max_position_percentage_suggestion')
            portfolio_total_value = portfolio_data.get("cash", 0.0) + sum(p.get("shares",0)*p.get("avg_price",0) for p in portfolio_data.get("positions",{}).values())
            if portfolio_total_value == 0: portfolio_total_value = 100000 # Avoid division by zero for new portfolio

            if isinstance(llm_max_pos_perc, (float, int)) and 0 < llm_max_pos_perc <= 1.0:
                # Override max_position_value based on LLM suggestion
                original_max_pos_val = ticker_params.max_position_value
                ticker_params.max_position_value = portfolio_total_value * llm_max_pos_perc
                print(f"  {ticker}: Adjusted max_position_value from ${original_max_pos_val:.2f} to ${ticker_params.max_position_value:.2f} based on LLM suggestion ({llm_max_pos_perc*100}%). ")
                
                # Recalculate max_shares_to_buy based on the new max_position_value
                # Need current price for this recalculation
                price_data = get_price_data(ticker)
                current_price = 0
                try:
                    current_price = float(price_data.get("Global Quote", {}).get("05. price", 0))
                except (ValueError, TypeError):
                    current_pos_data = portfolio_data.get("positions", {}).get(ticker, {})
                    current_price = current_pos_data.get("avg_price", 100.0) # Fallback price
                if current_price == 0: current_price = 100.0 # Final fallback

                current_value_in_portfolio = 0
                if ticker in portfolio_data.get("positions", {}):
                     pos = portfolio_data["positions"][ticker]
                     current_value_in_portfolio = pos.get("shares",0) * pos.get("avg_price",0)
                
                remaining_value_capacity = max(0, ticker_params.max_position_value - current_value_in_portfolio)
                ticker_params.max_shares_to_buy = int(remaining_value_capacity / current_price) if current_price > 0 else 0
                if current_value_in_portfolio > ticker_params.max_position_value : ticker_params.max_shares_to_buy = 0 # Do not buy if already over limit
                print(f"  {ticker}: Recalculated max_shares_to_buy to {ticker_params.max_shares_to_buy} at price ${current_price:.2f}")
            else:
                print(f"  {ticker}: LLM did not provide a valid max_position_percentage_suggestion or it was out of range (0-1). Using quantitative limits.")
                # Fallback to simpler risk level adjustment if direct percentage is not good
                risk_level = str(llm_risk_synthesis.get("risk_level", "Medium")).lower()
                adjustment_factor = 1.0
                if risk_level == "high": adjustment_factor = 0.7
                elif risk_level == "low": adjustment_factor = 1.2
                
                if adjustment_factor != 1.0:
                    original_max_pos_val = ticker_params.max_position_value
                    ticker_params.max_position_value *= adjustment_factor
                    # Also adjust max_shares_to_buy proportionally
                    ticker_params.max_shares_to_buy = int(ticker_params.max_shares_to_buy * adjustment_factor)
                    print(f"  {ticker}: Adjusted max_position_value from ${original_max_pos_val:.2f} to ${ticker_params.max_position_value:.2f} and shares by factor {adjustment_factor} based on risk_level '{risk_level}'.")

    return quantitative_risk_params

def load_portfolio(portfolio_path: str = "portfolio.json") -> Dict[str, Any]:
    """Load portfolio data from JSON file"""
    if not os.path.exists(portfolio_path):
        return {"cash": 100000.0, "positions": {}}
    
    try:
        with open(portfolio_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading portfolio from {portfolio_path}: {e}")
        return {"cash": 100000.0, "positions": {}}

def save_risk_parameters(risk_params: PortfolioRiskParameters, output_path: str = None) -> None:
    """Save risk parameters to a JSON file"""
    if output_path is None:
        output_path = os.path.join(DATA_FOLDER, "risk_parameters.json")
    
    # Convert to dict for serialization
    risk_params_dict = {
        "total_portfolio_max_risk_exposure": risk_params.total_portfolio_max_risk_exposure,
        "cash_buffer_percentage": risk_params.cash_buffer_percentage,
        "ticker_specific_parameters": {
            ticker: {
                "ticker": params.ticker,
                "max_position_value": params.max_position_value,
                "max_shares_to_buy": params.max_shares_to_buy,
                "max_shares_to_short": params.max_shares_to_short
            }
            for ticker, params in risk_params.ticker_specific_parameters.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(risk_params_dict, f, indent=2)
    
    print(f"Risk parameters saved to {output_path}")

def main(tickers: Optional[List[str]] = None, portfolio_path: str = "portfolio.json") -> PortfolioRiskParameters:
    """
    Main function to generate risk parameters for tickers
    """
    # Use SYMBOL environment variable if tickers not provided
    if tickers is None:
        if SYMBOL:
            tickers = [SYMBOL]
        else:
            raise ValueError("No tickers provided and SYMBOL environment variable not set")
    
    print(f"Generating risk parameters for: {', '.join(tickers)}")
    
    # Load portfolio
    portfolio = load_portfolio(portfolio_path)
    print(f"Loaded portfolio with ${portfolio.get('cash', 0):.2f} cash and {len(portfolio.get('positions', {}))} positions")
    
    # Generate risk parameters
    risk_params = generate_risk_parameters_for_tickers(tickers, portfolio)
    
    # Save risk parameters
    save_risk_parameters(risk_params)
    
    # Display a summary
    print("\n--- Risk Parameter Summary ---")
    print(f"Portfolio Max Risk Exposure: ${risk_params.total_portfolio_max_risk_exposure:.2f}")
    print(f"Cash Buffer: {risk_params.cash_buffer_percentage*100:.1f}%")
    
    for ticker, params in risk_params.ticker_specific_parameters.items():
        print(f"\n{ticker}:")
        print(f"  Max Position Value: ${params.max_position_value:.2f}")
        print(f"  Max Shares to Buy: {params.max_shares_to_buy}")
        print(f"  Max Shares to Short: {params.max_shares_to_short}")
    
    return risk_params

if __name__ == "__main__":
    # If run directly, use the SYMBOL environment variable
    if SYMBOL:
        main([SYMBOL])
    else:
        print("Please set the SYMBOL environment variable or provide tickers as arguments.") 