import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
# API_KEY = "U8DWLP7NIQJDT6XL"
DATA_FOLDER = "data"

os.makedirs(DATA_FOLDER, exist_ok=True)


def get_alpha_vantage_api_key():
    """Retrieves the Alpha Vantage API key from environment variables."""
    return os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")

def fetch_historical_time_series(symbol: str, series_type: str = "weekly", outputsize: str = "full", api_key: str = None):
    """
    Fetch historical time series data (daily or weekly) for a given symbol from Alpha Vantage.

    Args:
        symbol (str): The stock symbol (e.g., "IBM").
        series_type (str): "daily" or "weekly". Defaults to "weekly".
        outputsize (str): "compact" (last 100 data points) or "full". Defaults to "full".
        api_key (str, optional): Alpha Vantage API key. If None, uses ALPHA_VANTAGE_API_KEY env var.

    Returns:
        dict: The JSON response from Alpha Vantage, or None if an error occurs.
              For weekly data, key is "Weekly Adjusted Time Series".
              For daily data, key is "Time Series (Daily)".
    """
    if api_key is None:
        api_key = get_alpha_vantage_api_key()

    if series_type == "weekly":
        function_name = "TIME_SERIES_WEEKLY_ADJUSTED"
        series_key = "Weekly Adjusted Time Series"
    elif series_type == "daily":
        function_name = "TIME_SERIES_DAILY"
        series_key = "Time Series (Daily)"
    else:
        print(f"Error: Invalid series_type '{series_type}'. Must be 'daily' or 'weekly'.")
        return None

    url = f"https://www.alphavantage.co/query?function={function_name}&symbol={symbol}&outputsize={outputsize}&apikey={api_key}"
    
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        
        if "Error Message" in data:
            print(f"API Error for {symbol} ({series_type}): {data['Error Message']}")
            return None
        if "Information" in data and "rate limit" in data["Information"].lower():
             print(f"API Rate limit hit for {symbol} ({series_type}): {data['Information']}")
             return None
        if not data.get(series_key):
            print(f"Unexpected API response structure for {symbol} ({series_type}). Missing key: {series_key}")
            return None
            
        return data
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {symbol} ({series_type}): {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response for {symbol} ({series_type}): {e}")
        return None

def get_last_n_years_data(historical_data: dict, years: int = 5, series_type: str = "weekly"):
    """
    Filters historical data for the last N years.
    Assumes data is sorted reverse chronologically (most recent first).

    Args:
        historical_data (dict): The raw JSON output from fetch_historical_time_series.
        years (int): Number of years of data to retrieve.
        series_type (str): "daily" or "weekly". Used to find the correct data key.

    Returns:
        dict: A dictionary containing data points for the last N years.
              Returns None if input data is invalid.
    """
    if not historical_data:
        return None

    if series_type == "weekly":
        series_key = "Weekly Adjusted Time Series"
    elif series_type == "daily":
        series_key = "Time Series (Daily)"
    else:
        print(f"Error: Invalid series_type '{series_type}' for filtering.")
        return None

    time_series = historical_data.get(series_key)
    if not time_series:
        print(f"Could not find time series data under key '{series_key}'.")
        return None

    cutoff_date = datetime.now() - timedelta(days=years * 365.25)
    filtered_data = {}

    for date_str, data_point in time_series.items():
        try:
            current_date = datetime.strptime(date_str, "%Y-%m-%d")
            if current_date >= cutoff_date:
                filtered_data[date_str] = data_point
            else:
                break 
        except ValueError:
            print(f"Warning: Could not parse date '{date_str}'. Skipping.")
            continue
            
    return filtered_data

def get_income_statement_data(symbol):
    """Fetch and save income statement data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_income_statement.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def get_balance_sheet_data(symbol):
    """Fetch and save balance sheet data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_balance_sheet.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def get_cash_flow_data(symbol):
    """Fetch and save cash flow statement for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_cash_flow.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def get_ratio_data(symbol):
    """Fetch and save valuation and ratio metrics (TTM) for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_ratio.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_news_data(symbol):
    """Fetch and save news data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_news.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_earnings_transcripts_data(symbol):
    """Fetch and save earnings transcripts data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPTS&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_earnings_transcripts.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_insider_trading_data(symbol):
    """Fetch and save insider trading data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_insider_trading.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_earnings_data(symbol):
    """Fetch and save earnings data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError:
        data = {"error": "No data or invalid response from API"}
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_earnings.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_etf_profile_holdings_data(symbol):
    """Fetch and save ETF profile and holdings data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=ETF_OVERVIEW&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_etf_profile.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_dividends_data(symbol):
    """Fetch and save dividend data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=DIVIDEND_HISTORY&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_dividends.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_splits_data(symbol):
    """Fetch and save stock split data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=STOCK_SPLITS&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_splits.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_listing_status_data(symbol):
    """Fetch and save listing & delisting status for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_listing_status.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_earnings_calendar_data(symbol):
    """Fetch and save earnings calendar data for a given symbol."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_earnings_calendar.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_ipo_calendar_data():
    """Fetch and save IPO calendar data (no symbol required)."""
    api_key = get_alpha_vantage_api_key()
    url = f"https://www.alphavantage.co/query?function=IPO_CALENDAR&apikey={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, "ipo_calendar.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

    

