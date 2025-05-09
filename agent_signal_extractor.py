import os
import json
from typing import List, Optional, Dict, Any
from ..core.data_structures import AgentSignal
from dotenv import load_dotenv
load_dotenv()

def load_agent_analysis_from_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load agent analysis from a JSON file if it exists and is valid JSON."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return None # Or raise error, or return a dict with an error field
        except Exception as e:
            print(f"An unexpected error occurred while reading {file_path}: {e}")
            return None
    return None

def map_recommendation_to_signal_type(recommendation: Optional[str]) -> str:
    """Maps BUY/SELL/HOLD/AVOID to bullish/bearish/neutral."""
    if recommendation:
        rec_upper = recommendation.upper()
        if rec_upper == "BUY":
            return "bullish"
        elif rec_upper in ["SELL", "AVOID"]:
            return "bearish"
        elif rec_upper == "HOLD":
            return "neutral"
    return "neutral" # Default or if recommendation is None

def get_agent_signals(ticker: str, data_folder: str = "data") -> List[AgentSignal]:
    """
    Get signals for a ticker from all available agent JSON analysis files.
    """
    signals = []
    
    # Updated agent names to match potential output filenames (e.g., 'buffett' not 'BuffetAgent')
    # The agent name in the JSON file itself ('agent' key) will be the definitive source.
    agent_file_map = {
        "Buffett": os.path.join(data_folder, f"{ticker}_buffett_analysis.json"),
        "Lynch": os.path.join(data_folder, f"{ticker}_lynch_analysis.json"),
        "Wood": os.path.join(data_folder, f"{ticker}_wood_analysis.json"),
    }
    
    for agent_key, file_path in agent_file_map.items(): # agent_key here is just for finding the file
        analysis_data = load_agent_analysis_from_json_file(file_path)
        
        if analysis_data:
            # Validate that essential keys are present
            agent_name_from_json = analysis_data.get('agent')
            ticker_from_json = analysis_data.get('ticker')
            recommendation = analysis_data.get('recommendation')
            confidence = analysis_data.get('confidence_score') # Updated key
            reasoning = analysis_data.get('reasoning')
            target_price = analysis_data.get('target_price')
            stop_loss_price = analysis_data.get('stop_loss_price')

            if not all([agent_name_from_json, ticker_from_json, recommendation, confidence is not None]):
                print(f"Skipping file {file_path} due to missing essential data (agent, ticker, recommendation, or confidence_score).")
                continue

            if ticker_from_json.upper() != ticker.upper():
                print(f"Warning: Ticker in filename ({ticker}) does not match ticker in JSON content ({ticker_from_json}) for file {file_path}.")
                # Decide on handling: skip, use filename ticker, or use JSON ticker. For now, using JSON ticker.

            signal_type = map_recommendation_to_signal_type(recommendation)
            
            # Ensure numeric fields are float or None
            try:
                confidence_float = float(confidence) if confidence is not None else 0.5 # Default confidence if None
            except (ValueError, TypeError):
                print(f"Warning: Could not parse confidence '{confidence}' as float for {file_path}. Defaulting to 0.5.")
                confidence_float = 0.5
            
            try:
                target_price_float = float(target_price) if target_price is not None else None
            except (ValueError, TypeError):
                print(f"Warning: Could not parse target_price '{target_price}' as float for {file_path}. Setting to None.")
                target_price_float = None

            try:
                stop_loss_price_float = float(stop_loss_price) if stop_loss_price is not None else None
            except (ValueError, TypeError):
                print(f"Warning: Could not parse stop_loss_price '{stop_loss_price}' as float for {file_path}. Setting to None.")
                stop_loss_price_float = None

            signals.append(
                AgentSignal(
                    agent_name=str(agent_name_from_json), # Use agent name from JSON content
                    ticker=str(ticker_from_json),      # Use ticker from JSON content
                    signal_type=signal_type,
                    confidence=confidence_float,
                    reasoning=str(reasoning) if reasoning is not None else "No reasoning provided.",
                    target_price=target_price_float,
                    stop_loss_price=stop_loss_price_float
                    # Lynch/Wood specific fields could be added to AgentSignal or handled downstream
                )
            )
        else:
            print(f"No analysis data found or error loading for {agent_key} on {ticker} from {file_path}")
            
    return signals

# Testing function
if __name__ == "__main__":
    data_folder_test = "test_data_extractor"
    os.makedirs(data_folder_test, exist_ok=True)
    
    test_ticker = "TESTJSON"
    
    # Dummy Buffet JSON
    dummy_buffet_data = {
        "agent": "Buffett", "ticker": test_ticker, "recommendation": "BUY", 
        "confidence_score": 0.85, "target_price": 120.50, "stop_loss_price": 90.00,
        "reasoning": "Company has a strong moat."
    }
    with open(os.path.join(data_folder_test, f"{test_ticker}_buffett_analysis.json"), 'w') as f:
        json.dump(dummy_buffet_data, f, indent=2)
        
    # Dummy Lynch JSON
    dummy_lynch_data = {
        "agent": "Lynch", "ticker": test_ticker, "recommendation": "HOLD",
        "confidence_score": 0.6, "target_price": None, "stop_loss_price": None,
        "reasoning": "Growth is good, but valuation is fair.",
        "key_growth_drivers": ["New market entry"], "risk_factors": ["Competition"]
    }
    with open(os.path.join(data_folder_test, f"{test_ticker}_lynch_analysis.json"), 'w') as f:
        json.dump(dummy_lynch_data, f, indent=2)

    # Dummy Wood JSON (missing target price for testing robustness)
    dummy_wood_data = {
        "agent": "Wood", "ticker": test_ticker, "recommendation": "BUY",
        "confidence_score": "0.90", # Test string confidence
        "target_price": "N/A", # Test non-numeric target
        "stop_loss_price": 50.75,
        "reasoning": "It is innovative and disruptive.",
        "disruptive_theme": "AI", "catalysts": ["Product launch Q3"]
    }
    with open(os.path.join(data_folder_test, f"{test_ticker}_wood_analysis.json"), 'w') as f:
        json.dump(dummy_wood_data, f, indent=2)

    # Test case for a missing file
    # No _missing_analysis.json file will be created

    print(f"--- Testing Agent Signal Extractor for {test_ticker} ---")
    signals = get_agent_signals(test_ticker, data_folder=data_folder_test)
    
    assert len(signals) == 3, f"Expected 3 signals, got {len(signals)}"
    print(f"Successfully extracted {len(signals)} signals.")

    for signal in signals:
        print(f"Agent: {signal.agent_name}, Ticker: {signal.ticker}")
        print(f"  Signal: {signal.signal_type}, Confidence: {signal.confidence}")
        print(f"  Reasoning: {signal.reasoning}")
        print(f"  Target Price: {signal.target_price if signal.target_price is not None else 'N/A'}")
        print(f"  Stop-Loss Price: {signal.stop_loss_price if signal.stop_loss_price is not None else 'N/A'}")
        assert isinstance(signal.confidence, float)
        assert signal.ticker == test_ticker
        if signal.agent_name == "Wood":
            assert signal.target_price is None # Due to "N/A" in test data
        print("-" * 50)
    
    print("Signal extraction tests passed.")

    # Clean up dummy files
    for f_name in os.listdir(data_folder_test):
        os.remove(os.path.join(data_folder_test, f_name))
    os.rmdir(data_folder_test) 