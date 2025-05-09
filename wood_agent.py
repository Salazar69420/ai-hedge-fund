import json  # for parsing function arguments
from openai import OpenAI
import os
from functions import get_income_statement_data, get_balance_sheet_data, get_cash_flow_data, get_ratio_data
from dotenv import load_dotenv
load_dotenv()

# Configuration - load from environment variables or use defaults
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-39cbd05d862749d8bca6c13f1bc234ee')
SYMBOL = os.environ.get('SYMBOL', 'ASTS')

# Define folder path
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)  # Ensure data folder exists

# Initialize client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
model = "deepseek-chat"
context = "Analyze the investment quality of a company using a Cathie Wood/ARK Invest-style lens."

# Define tools for DeepSeek function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_income_statement_data",
            "description": "Retrieve income statement data for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The stock ticker symbol"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet_data",
            "description": "Retrieve balance sheet data for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The stock ticker symbol"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cash_flow_data",
            "description": "Retrieve cash flow data for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The stock ticker symbol"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ratio_data",
            "description": "Retrieve valuation and ratio metrics for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The stock ticker symbol"}
                },
                "required": ["symbol"]
            }
        }
    }
]

# Track tool usage
used_tools = {
    "get_ratio_data": False,
    "get_income_statement_data": False,
    "get_balance_sheet_data": False,
    "get_cash_flow_data": False
}

# CHUNKED ANALYSIS FLOW
messages = [
    {"role": "system", "content": context},
    {"role": "system", "content": "You are Cathie Wood, ARK Invest founder. Use a forward-looking, innovation-driven approach, focusing on disruptive technology, exponential growth, and visionary leadership."},
    {"role": "user", "content": f'First, call get_ratio_data for {SYMBOL} and summarize only the key ratios needed for ARK Invest-style analysis.'}
]

# Step 1: Get key ratios
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)
response_message = response.choices[0].message
tool_calls = getattr(response_message, 'tool_calls', None)
if tool_calls:
    messages.append(response_message)
    for tool in tool_calls:
        if tool.function.name == "get_ratio_data":
            data = get_ratio_data(SYMBOL)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_ratio_data"] = True
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    ratio_summary = response.choices[0].message.content
    messages = [
        {"role": "system", "content": context},
        {"role": "system", "content": "You are Cathie Wood, ARK Invest founder. Use a forward-looking, innovation-driven approach, focusing on disruptive technology, exponential growth, and visionary leadership."},
        {"role": "user", "content": f'Key ratio summary for {SYMBOL}: {ratio_summary}'}
    ]

# Step 2: Income statement
messages.append({"role": "user", "content": f'Now, call get_income_statement_data for {SYMBOL} and summarize only the key points needed for ARK Invest-style analysis.'})
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)
response_message = response.choices[0].message
tool_calls = getattr(response_message, 'tool_calls', None)
if tool_calls:
    messages.append(response_message)
    for tool in tool_calls:
        if tool.function.name == "get_income_statement_data":
            data = get_income_statement_data(SYMBOL)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_income_statement_data"] = True
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    income_summary = response.choices[0].message.content
    messages = [
        {"role": "system", "content": context},
        {"role": "system", "content": "You are Cathie Wood, ARK Invest founder. Use a forward-looking, innovation-driven approach, focusing on disruptive technology, exponential growth, and visionary leadership."},
        {"role": "user", "content": f'Key ratio summary for {SYMBOL}: {ratio_summary}\nIncome statement summary: {income_summary}'}
    ]

# Step 3: Balance sheet
messages.append({"role": "user", "content": f'Now, call get_balance_sheet_data for {SYMBOL} and summarize only the key points needed for ARK Invest-style analysis.'})
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)
response_message = response.choices[0].message
tool_calls = getattr(response_message, 'tool_calls', None)
if tool_calls:
    messages.append(response_message)
    for tool in tool_calls:
        if tool.function.name == "get_balance_sheet_data":
            data = get_balance_sheet_data(SYMBOL)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_balance_sheet_data"] = True
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    balance_summary = response.choices[0].message.content
    messages = [
        {"role": "system", "content": context},
        {"role": "system", "content": "You are Cathie Wood, ARK Invest founder. Use a forward-looking, innovation-driven approach, focusing on disruptive technology, exponential growth, and visionary leadership."},
        {"role": "user", "content": f'Key ratio summary for {SYMBOL}: {ratio_summary}\nIncome statement summary: {income_summary}\nBalance sheet summary: {balance_summary}'}
    ]

# Step 4: Cash flow
messages.append({"role": "user", "content": f'Now, call get_cash_flow_data for {SYMBOL} and summarize only the key points needed for ARK Invest-style analysis.'})
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)
response_message = response.choices[0].message
tool_calls = getattr(response_message, 'tool_calls', None)
if tool_calls:
    messages.append(response_message)
    for tool in tool_calls:
        if tool.function.name == "get_cash_flow_data":
            data = get_cash_flow_data(SYMBOL)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_cash_flow_data"] = True
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    cashflow_summary = response.choices[0].message.content
    messages = [
        {"role": "system", "content": context},
        {"role": "system", "content": "You are Cathie Wood, ARK Invest founder. Use a forward-looking, innovation-driven approach, focusing on disruptive technology, exponential growth, and visionary leadership."},
        {"role": "user", "content": f'Key ratio summary for {SYMBOL}: {ratio_summary}\nIncome statement summary: {income_summary}\nBalance sheet summary: {balance_summary}\nCash flow summary: {cashflow_summary}'}
    ]

# Step 5: Final analysis
messages.append({
    "role": "user",
    "content": f"""Given the above summaries, provide a final Cathie Wood-style investment analysis for {SYMBOL}.
Your analysis must be structured as a JSON object with the following keys: "recommendation", "confidence_score", "target_price", "stop_loss_price", "reasoning", "disruptive_theme", "catalysts".

- "recommendation": One of: BUY, HOLD, SELL, or AVOID.
- "confidence_score": A numerical value between 0.0 (low) and 1.0 (high) representing your confidence in the recommendation.
- "target_price": Optional. Your estimated target price, often with a longer time horizon (e.g., 3-5 years). If not applicable, use null.
- "stop_loss_price": Optional. Your estimated stop-loss price. If not applicable, use null.
- "reasoning": Brief reasoning for your recommendation, focusing on disruptive innovation, exponential growth potential, and market leadership.
- "disruptive_theme": Optional. The primary disruptive technology or theme the company is part of (e.g., AI, Genomics, Blockchain). If not applicable, use null.
- "catalysts": Optional. List of potential upcoming catalysts for the stock. If not applicable, use null or an empty list.

**Important Constraints:**
- If you provide a Target Price or Stop-Loss Price, you **must** state that it is **your persona's (Cathie Wood) estimation based *only* on the financial data and summaries provided in this interaction**. Do not cite external, unverified sources (e.g., 'analyst consensus') unless such information was explicitly part of the data given to you in previous steps.
- Your reasoning and any price levels must be directly justifiable from the provided financial summaries and your persona's investment thesis.
- Be concise and actionable.
- Output *only* the JSON object and nothing else. Example:
  {{
    "recommendation": "BUY",
    "confidence_score": 0.9,
    "target_price": 300.00,
    "stop_loss_price": null,
    "reasoning": "The company is a leader in AI-driven drug discovery, a massively disruptive theme with exponential growth potential over the next 5 years.",
    "disruptive_theme": "AI in Healthcare",
    "catalysts": ["Phase 3 trial results expected Q4", "New strategic partnership"]
  }}
"""
})
response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format={"type": "json_object"}
)
response_message = response.choices[0].message
structured_output_str = response_message.content

print("------------------------------------------------------------------------------------------------------- Final Answer (JSON) -------------------------------------------------------------------------------------------")
print(structured_output_str)

# Attempt to parse the JSON
try:
    parsed_json = json.loads(structured_output_str)
    parsed_json['agent'] = "Wood"
    parsed_json['ticker'] = SYMBOL
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from LLM: {e}")
    print(f"LLM output was: {structured_output_str}")
    parsed_json = {"error": "Failed to parse LLM JSON output", "raw_output": structured_output_str, "agent": "Wood", "ticker": SYMBOL}

# Save to file
output_file = os.path.join(data_folder, f"{SYMBOL}_wood_analysis.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(parsed_json, f, indent=2)

print(f"\nAnalysis saved to {output_file}")
print("--------------------------------------------------------------------------------------------------------- End of Analysis ------------------------------------------------------------------------------------------")
# Print tool usage summary
print("\nTool usage summary:")
for tool, used in used_tools.items():
    print(f"{tool}: {used}")

# End of wood_agent.py

