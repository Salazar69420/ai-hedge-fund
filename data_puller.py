import os
import time
from ai_hedge_fund.functions import (
    get_income_statement_data,
    get_balance_sheet_data,
    get_cash_flow_data,
    get_ratio_data,
    get_news_data,
    get_earnings_transcripts_data,
    get_insider_trading_data,
    get_earnings_data
)

# Configuration
# List of tickers to process directly
DIRECT_TICKERS = []  # Updated tickers

# Path to the file containing additional tickers
TICKER_FILE_PATH = "nyse_list.csv"
DATA_FOLDER = "data"

# API Rate Limiting
# 8 functions are called per ticker.
# Max 75 requests per minute.
# Delay per ticker = (number_of_functions * 60 seconds) / max_requests_per_minute
REQUESTS_PER_TICKER = 8
MAX_REQUESTS_PER_MINUTE = 75
DELAY_PER_TICKER = (REQUESTS_PER_TICKER * 60) / MAX_REQUESTS_PER_MINUTE


def load_tickers_from_file(file_path):
    """Loads tickers from a CSV file, one symbol per line, skips header."""
    tickers = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 or not line:  # Skip header or empty lines
                    continue
                tickers.append(line)
    else:
        print(f"Ticker file not found: {file_path}")
    return tickers

def pull_all_data_for_ticker(symbol):
    """Pulls all available data for a given stock symbol."""
    print(f"--- Pulling data for {symbol} ---")
    try:
        get_income_statement_data(symbol)
        print(f"  Successfully pulled income statement for {symbol}")
    except Exception as e:
        print(f"  Error pulling income statement for {symbol}: {e}")

    try:
        get_balance_sheet_data(symbol)
        print(f"  Successfully pulled balance sheet for {symbol}")
    except Exception as e:
        print(f"  Error pulling balance sheet for {symbol}: {e}")

    try:
        get_cash_flow_data(symbol)
        print(f"  Successfully pulled cash flow for {symbol}")
    except Exception as e:
        print(f"  Error pulling cash flow for {symbol}: {e}")

    try:
        get_ratio_data(symbol)
        print(f"  Successfully pulled ratio data for {symbol}")
    except Exception as e:
        print(f"  Error pulling ratio data for {symbol}: {e}")

    try:
        get_news_data(symbol)
        print(f"  Successfully pulled news data for {symbol}")
    except Exception as e:
        print(f"  Error pulling news data for {symbol}: {e}")

    try:
        get_earnings_transcripts_data(symbol)
        print(f"  Successfully pulled earnings transcripts for {symbol}")
    except Exception as e:
        print(f"  Error pulling earnings transcripts for {symbol}: {e}")

    try:
        get_insider_trading_data(symbol)
        print(f"  Successfully pulled insider trading data for {symbol}")
    except Exception as e:
        print(f"  Error pulling insider trading data for {symbol}: {e}")
    
    try:
        get_earnings_data(symbol)
        print(f"  Successfully pulled earnings data for {symbol}")
    except Exception as e:
        print(f"  Error pulling earnings data for {symbol}: {e}")

    print(f"--- Finished pulling data for {symbol} ---")


def main():
    """Main function to load tickers and pull data."""
    # Ensure data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Load tickers from file
    file_tickers = load_tickers_from_file(TICKER_FILE_PATH)
    
    # Combine tickers and remove duplicates
    all_tickers = []
    seen = set()
    for ticker in DIRECT_TICKERS + file_tickers:
        if ticker not in seen:
            all_tickers.append(ticker)
            seen.add(ticker)
    
    if not all_tickers:
        print("No tickers specified. Please add tickers to DIRECT_TICKERS or to the ticker file.")
        return

    print(f"Processing the following tickers: {all_tickers}")
    
    for i, ticker in enumerate(all_tickers):
        pull_all_data_for_ticker(ticker)
        if i < len(all_tickers) - 1:  # Don't sleep after the last ticker
            print(f"Waiting {DELAY_PER_TICKER:.2f} seconds before next ticker to respect API rate limits...")
            time.sleep(DELAY_PER_TICKER)
            
    print("All data pulling complete.")

if __name__ == "__main__":
    main() 