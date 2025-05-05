import os
import json
import requests

API_KEY = "U8DWLP7NIQJDT6XL"
DATA_FOLDER = "data"

os.makedirs(DATA_FOLDER, exist_ok=True)


def get_income_statement_data(symbol):
    """Fetch and save income statement data for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_income_statement.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def get_balance_sheet_data(symbol):
    """Fetch and save balance sheet data for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_balance_sheet.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def get_cash_flow_data(symbol):
    """Fetch and save cash flow statement for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_cash_flow.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def get_ratio_data(symbol):
    """Fetch and save valuation and ratio metrics (TTM) for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_ratio.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_news_data(symbol):
    """Fetch and save news data for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_news.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_earnings_transcripts_data(symbol):
    """Fetch and save earnings transcripts data for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPTS&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_earnings_transcripts.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_insider_trading_data(symbol):
    """Fetch and save insider trading data for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_insider_trading.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

def get_earnings_data(symbol):
    """Fetch and save earnings data for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    out_file = os.path.join(DATA_FOLDER, f"{symbol}_earnings.json")
    with open(out_file, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data

    

