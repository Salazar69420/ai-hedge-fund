import json
import csv
import os
import glob
from datetime import datetime

# --- Helper Functions ---

def safe_get(data, keys, default='None'):
    """Safely get nested dictionary keys."""
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    # Replace None values from JSON explicitly with the string 'None' if needed,
    # or handle specific data types as required for CSV consistency.
    return data if data is not None else default

def write_csv(output_csv_file, headers, rows):
    """Writes data rows to a CSV file with tab delimiter."""
    try:
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Successfully wrote {len(rows)} rows to {output_csv_file}")
    except Exception as e:
        print(f"Error writing to {output_csv_file}: {str(e)}")

# --- Parsing Functions ---

def parse_financial_statement(input_json_file, output_csv_file, statement_type, headers_map):
    """Generic parser for Income Statement, Balance Sheet, Cash Flow."""
    print(f"Parsing {statement_type} from {input_json_file}")
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"  Successfully loaded JSON for {statement_type}")

        if not data or 'symbol' not in data:
            print(f"Skipping {input_json_file}: Invalid or empty data.")
            return

        all_reports = data.get('annualReports', []) + data.get('quarterlyReports', [])
        if not all_reports:
            print(f"Skipping {input_json_file}: No reports found.")
            return
        print(f"  Found {len(all_reports)} reports for {statement_type}")

        all_reports.sort(key=lambda x: x.get('fiscalDateEnding', '0000-00-00'), reverse=True)

        headers = list(headers_map.keys())
        rows = []
        print(f"  Processing reports for {statement_type}...")
        for i, report in enumerate(all_reports):
            # print(f"    Processing report {i+1}/{len(all_reports)}") # Potentially too verbose
            report_type = 'Annual' if report in data.get('annualReports', []) else 'Quarterly'
            row = [report.get('fiscalDateEnding', 'N/A'), report_type]
            for header, key in headers_map.items():
                 if header not in ['Report_Date', 'Report_Type (Annual/Quarterly)']: # Skip already added fields
                    value = safe_get(report, [key])
                    row.append(value)
            rows.append(row)
        
        print(f"  Generated {len(rows)} rows for {statement_type}")

        if rows:
            print(f"  Attempting to write {len(rows)} rows to {output_csv_file}")
            write_csv(output_csv_file, headers, rows)
        else:
            print(f"No data rows generated for {output_csv_file}")

    except FileNotFoundError:
        print(f"Skipping {input_json_file}: File not found.")
    except json.JSONDecodeError:
        print(f"Skipping {input_json_file}: Invalid JSON format.")
    except Exception as e:
        print(f"Error parsing {input_json_file}: {str(e)}")
        import traceback
        traceback.print_exc() # Ensure traceback is printed for any exception

def parse_overview(input_json_file, output_csv_file):
    """Parses Overview (Ratio) data."""
    print(f"Parsing Overview from {input_json_file}")
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data or 'Symbol' not in data:
             # Check for potential API error message
            if isinstance(data, dict) and data.get("Error Message"):
                 print(f"Skipping {input_json_file}: API Error - {data.get('Error Message')}")
            elif isinstance(data, dict) and data.get("Information"):
                 print(f"Skipping {input_json_file}: API Info - {data.get('Information')}")
            else:
                print(f"Skipping {input_json_file}: Invalid or empty data.")
            return

        # Define headers based on the available keys in the JSON, providing more descriptive names
        headers_map = {
            'Symbol': 'Symbol',
            'AssetType': 'AssetType',
            'Name': 'Name',
            'Description': 'Description',
            'CIK': 'CIK',
            'Exchange': 'Exchange',
            'Currency': 'Currency',
            'Country': 'Country',
            'Sector': 'Sector',
            'Industry': 'Industry',
            'Address': 'Address',
            'FiscalYearEnd': 'FiscalYearEnd',
            'LatestQuarter': 'LatestQuarter',
            'MarketCap/CompanyValuation (USD)': 'MarketCapitalization',
            'EBITDA (USD)': 'EBITDA',
            'PERatio/PriceEarningsRatio': 'PERatio',
            'PEGRatio/PriceEarningsToGrowthRatio': 'PEGRatio',
            'BookValuePerShare (USD)': 'BookValue',
            'DividendPerShare (USD)': 'DividendPerShare',
            'DividendYield/DividendReturn (%)': 'DividendYield',
            'EPS/EarningsPerShare (USD)': 'EPS',
            'RevenuePerShareTTM (USD)': 'RevenuePerShareTTM',
            'ProfitMargin (%)': 'ProfitMargin',
            'OperatingMarginTTM (%)': 'OperatingMarginTTM',
            'ReturnOnAssetsTTM (%)': 'ReturnOnAssetsTTM',
            'ReturnOnEquityTTM (%)': 'ReturnOnEquityTTM',
            'RevenueTTM (USD)': 'RevenueTTM',
            'GrossProfitTTM (USD)': 'GrossProfitTTM',
            'DilutedEPSTTM (USD)': 'DilutedEPSTTM',
            'QuarterlyEarningsGrowthYOY (%)': 'QuarterlyEarningsGrowthYOY',
            'QuarterlyRevenueGrowthYOY (%)': 'QuarterlyRevenueGrowthYOY',
            'AnalystTargetPrice (USD)': 'AnalystTargetPrice',
            'TrailingPE': 'TrailingPE',
            'ForwardPE': 'ForwardPE',
            'PriceToSalesRatioTTM': 'PriceToSalesRatioTTM',
            'PriceToBookRatio': 'PriceToBookRatio',
            'EVToRevenue': 'EVToRevenue',
            'EVToEBITDA': 'EVToEBITDA',
            'Beta/Volatility': 'Beta',
            '52WeekHigh (USD)': '52WeekHigh',
            '52WeekLow (USD)': '52WeekLow',
            '50DayMovingAverage (USD)': '50DayMovingAverage',
            '200DayMovingAverage (USD)': '200DayMovingAverage',
            'SharesOutstanding/ShareCount': 'SharesOutstanding',
            'DividendDate': 'DividendDate',
            'ExDividendDate': 'ExDividendDate'
            # Add more mappings if needed based on API output inspection
        }

        headers = list(headers_map.keys())
        row = [safe_get(data, [json_key]) for json_key in headers_map.values()]
        rows = [row] # Overview data is typically a single record

        write_csv(output_csv_file, headers, rows)

    except FileNotFoundError:
        print(f"Skipping {input_json_file}: File not found.")
    except json.JSONDecodeError:
        print(f"Skipping {input_json_file}: Invalid JSON format.")
    except Exception as e:
        print(f"Error parsing {input_json_file}: {str(e)}")


def parse_news(input_json_file, output_csv_file):
    """Parses News Sentiment data."""
    print(f"Parsing News from {input_json_file}")
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data or 'feed' not in data or not isinstance(data['feed'], list):
            if isinstance(data, dict) and data.get("Information"):
                 print(f"Skipping {input_json_file}: API Info - {data.get('Information')}")
            else:
                print(f"Skipping {input_json_file}: Invalid or empty news feed data.")
            return

        headers = [
            'Published_Time', 'Title', 'URL/SourceLink', 'Summary/Abstract',
            'Overall_Sentiment_Score (-1 to 1)', 'Overall_Sentiment (Bearish/Bullish/Neutral)',
            'Source', 'Source_Domain'
            # Add more headers if needed, e.g., for specific topics or tickers
        ]
        rows = []

        for item in data['feed']:
            row = [
                safe_get(item, ['time_published']),
                safe_get(item, ['title']),
                safe_get(item, ['url']),
                safe_get(item, ['summary']),
                safe_get(item, ['overall_sentiment_score']),
                safe_get(item, ['overall_sentiment_label']),
                safe_get(item, ['source']),
                safe_get(item, ['source_domain'])
                # Extract more fields like banner_image, authors, topics if needed
            ]
            rows.append(row)

        if rows:
            write_csv(output_csv_file, headers, rows)
        else:
            print(f"No data rows generated for {output_csv_file}")

    except FileNotFoundError:
        print(f"Skipping {input_json_file}: File not found.")
    except json.JSONDecodeError:
        print(f"Skipping {input_json_file}: Invalid JSON format.")
    except Exception as e:
        print(f"Error parsing {input_json_file}: {str(e)}")


def parse_insider_trading(input_json_file, output_csv_file):
    """Parses Insider Trading data."""
    print(f"Parsing Insider Trading from {input_json_file}")
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"  Successfully loaded JSON for Insider Trading")

        # Check the structure: expects a dictionary with a 'data' key containing a list
        if not isinstance(data, dict) or 'data' not in data or not isinstance(data['data'], list):
            if isinstance(data, dict) and data.get("Information"):
                 print(f"Skipping {input_json_file}: API Info - {data.get('Information')}")
            elif isinstance(data, dict) and data.get("Error Message"):
                 print(f"Skipping {input_json_file}: API Error - {data.get('Error Message')}")
            else:
                print(f"Skipping {input_json_file}: Invalid or unexpected insider trading data structure.")
            return

        transactions = data['data']
        if not transactions:
            print(f"Skipping {input_json_file}: No transactions found in 'data' list.")
            return
        print(f"  Found {len(transactions)} transactions")

        headers = [
            'Transaction_Date', 'Ticker/Symbol', 'Executive_Name/Insider', 'Executive_Title/Position',
            'Security_Type/Instrument', 'Transaction_Type (Acquisition/Disposal)', 'Shares_Traded/Volume',
            'Share_Price (USD)'
        ]
        rows = []
        print(f"  Processing transactions...")
        for transaction in transactions:
            # Map acquisition/disposal codes
            transaction_type_code = safe_get(transaction, ['acquisition_or_disposal'])
            transaction_type = 'Acquisition' if transaction_type_code == 'A' else ('Disposal' if transaction_type_code == 'D' else 'Unknown')

            row = [
                safe_get(transaction, ['transaction_date']),
                safe_get(transaction, ['ticker']),
                safe_get(transaction, ['executive']),
                safe_get(transaction, ['executive_title']),
                safe_get(transaction, ['security_type']),
                transaction_type,
                safe_get(transaction, ['shares']),
                safe_get(transaction, ['share_price'])
            ]
            rows.append(row)

        if rows:
            # Sort by date descending
            print(f"  Generated {len(rows)} rows for Insider Trading")
            rows.sort(key=lambda x: x[0] if x[0] != 'None' else '0000-00-00', reverse=True)
            print(f"  Attempting to write {len(rows)} rows to {output_csv_file}")
            write_csv(output_csv_file, headers, rows)
        else:
            print(f"No data rows generated for {output_csv_file}")

    except FileNotFoundError:
        print(f"Skipping {input_json_file}: File not found.")
    except json.JSONDecodeError:
        print(f"Skipping {input_json_file}: Invalid JSON format.")
    except Exception as e:
        print(f"Error parsing {input_json_file}: {str(e)}")

def parse_earnings(input_json_file, output_csv_file):
    """Parses Earnings data (annual and quarterly)."""
    print(f"Parsing Earnings from {input_json_file}")
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data or 'symbol' not in data:
             # Check for potential API error message
            if isinstance(data, dict) and data.get("Error Message"):
                 print(f"Skipping {input_json_file}: API Error - {data.get('Error Message')}")
            elif isinstance(data, dict) and data.get("Information"):
                 print(f"Skipping {input_json_file}: API Info - {data.get('Information')}")
            else:
                print(f"Skipping {input_json_file}: Invalid or empty data.")
            return

        headers = [
            'Fiscal_Date_Ending', 'Report_Type (Annual/Quarterly)', 'Reported_Date',
            'Reported_EPS (USD)', 'Estimated_EPS (USD)', 'Surprise (USD)', 'Surprise (%)'
        ]
        rows = []

        # Process annual earnings
        for report in data.get('annualEarnings', []):
            rows.append([
                safe_get(report, ['fiscalDateEnding']),
                'Annual',
                'N/A', # Reported date not typically available for annual summary
                safe_get(report, ['reportedEPS']),
                'N/A', 'N/A', 'N/A' # Estimates/Surprise not typically available for annual summary
            ])

        # Process quarterly earnings
        for report in data.get('quarterlyEarnings', []):
            rows.append([
                safe_get(report, ['fiscalDateEnding']),
                'Quarterly',
                safe_get(report, ['reportedDate']),
                safe_get(report, ['reportedEPS']),
                safe_get(report, ['estimatedEPS']),
                safe_get(report, ['surprise']),
                safe_get(report, ['surprisePercentage'])
            ])

        if rows:
            # Sort by date descending
            rows.sort(key=lambda x: x[0] if x[0] != 'N/A' else '0000-00-00', reverse=True)
            write_csv(output_csv_file, headers, rows)
        else:
             print(f"No data rows generated for {output_csv_file}")

    except FileNotFoundError:
        print(f"Skipping {input_json_file}: File not found.")
    except json.JSONDecodeError:
        print(f"Skipping {input_json_file}: Invalid JSON format.")
    except Exception as e:
        print(f"Error parsing {input_json_file}: {str(e)}")

# --- Main Execution ---

def read_nyse_symbols(csv_file):
    """Read NYSE symbols from CSV file."""
    symbols = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            symbols = [row[0] for row in reader if row[0].strip()]
        print(f"Successfully loaded {len(symbols)} symbols from {csv_file}")
        return symbols
    except Exception as e:
        print(f"Error reading NYSE symbols from {csv_file}: {str(e)}")
        return []

def main(symbol=None):
    """Process data for a single symbol or all NYSE symbols."""
    if symbol:
        # Process single symbol
        process_symbol(symbol)
    else:
        # Process all NYSE symbols
        nyse_symbols = read_nyse_symbols('nyse_list.csv')
        if not nyse_symbols:
            print("No symbols found in NYSE list. Exiting.")
            return

        total_symbols = len(nyse_symbols)
        print(f"Starting batch processing of {total_symbols} symbols...")
        
        for i, symbol in enumerate(nyse_symbols, 1):
            print(f"\nProcessing symbol {i}/{total_symbols}: {symbol}")
            try:
                process_symbol(symbol)
            except Exception as e:
                print(f"Error processing symbol {symbol}: {str(e)}")
                continue

def process_symbol(symbol):
    """Process all data types for a single symbol."""
    # Define the base directory for JSON files
    json_base_dir = 'data'  # Changed from 'data/json' to 'data'
    
    # Define the base directory for CSV output
    csv_base_dir = 'data'  # Changed from 'data/csv' to 'data'
    
    # Process each data type
    data_types = {
        'ratio': parse_overview,  # Changed from 'overview' to 'ratio' to match file naming
        'income_statement': lambda i, o: parse_financial_statement(i, o, 'Income Statement', {
            'Report_Date': 'fiscalDateEnding',
            'Report_Type (Annual/Quarterly)': 'reportType',
            'Total Revenue (USD)': 'totalRevenue',
            'Gross Profit (USD)': 'grossProfit',
            'Operating Income (USD)': 'operatingIncome',
            'Net Income (USD)': 'netIncome',
            'EPS (USD)': 'eps',
            'EBITDA (USD)': 'ebitda'
        }),
        'balance_sheet': lambda i, o: parse_financial_statement(i, o, 'Balance Sheet', {
            'Report_Date': 'fiscalDateEnding',
            'Report_Type (Annual/Quarterly)': 'reportType',
            'Total Assets (USD)': 'totalAssets',
            'Total Current Assets (USD)': 'totalCurrentAssets',
            'Cash And Cash Equivalents (USD)': 'cashAndCashEquivalentsAtCarryingValue',
            'Total Liabilities (USD)': 'totalLiabilities',
            'Total Shareholder Equity (USD)': 'totalShareholderEquity',
            'Retained Earnings (USD)': 'retainedEarnings'
        }),
        'cash_flow': lambda i, o: parse_financial_statement(i, o, 'Cash Flow', {
            'Report_Date': 'fiscalDateEnding',
            'Report_Type (Annual/Quarterly)': 'reportType',
            'Operating Cash Flow (USD)': 'operatingCashflow',
            'Capital Expenditures (USD)': 'capitalExpenditures',
            'Free Cash Flow (USD)': 'freeCashflow',
            'Dividend Payout (USD)': 'dividendPayout'
        }),
        'news': parse_news,
        'insider_trading': parse_insider_trading,
        'earnings': parse_earnings
    }
    
    for data_type, parser_func in data_types.items():
        input_file = os.path.join(json_base_dir, f'{symbol}_{data_type}.json')
        output_file = os.path.join(csv_base_dir, f'{symbol}_{data_type}.csv')
        
        if os.path.exists(input_file):
            print(f"Processing {data_type} for {symbol}")
            parser_func(input_file, output_file)
        else:
            print(f"Skipping {data_type} for {symbol}: Input file not found")

if __name__ == "__main__":
    main()  # Process all symbols from NYSE list 