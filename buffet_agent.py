import json
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
function = "financials_agent"
model = "deepseek-chat"
context = "Analyze the investment quality of a company using a Buffett-style lens."

# Using functions module for dynamic data retrieval

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
# 1. Fetch and summarize key ratios first
messages = [
    {"role": "system", "content": context},
    {"role": "system", "content": "You are a conservative, long-term value investor like Warren Buffett. Use a disciplined, fundamental approach focused on durable competitive advantages, high returns on capital, and margin of safety."},
    {"role": "user", "content": f'First, call get_ratio_data for {SYMBOL} and summarize only the key ratios needed for Buffett-style analysis.'}
]

# Step 1: Get key ratios
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)
response_message = response.choices[0].message
print("DEBUG response_message (ratios):", response_message)
tool_calls = getattr(response_message, 'tool_calls', None)
if tool_calls:
    messages.append(response_message)
    for tool in tool_calls:
        if tool.function.name == "get_ratio_data":
            data = get_ratio_data(SYMBOL)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_ratio_data"] = True
    # Summarize key ratios
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    ratio_summary = response.choices[0].message.content
    # Keep only summary
    messages = [
        {"role": "system", "content": context},
        {"role": "system", "content": "You are a conservative, long-term value investor like Warren Buffett. Use a disciplined, fundamental approach focused on durable competitive advantages, high returns on capital, and margin of safety."},
        {"role": "user", "content": f'Key ratio summary for {SYMBOL}: {ratio_summary}'}
    ]

# Step 2: Repeat for income statement
messages.append({"role": "user", "content": f'Now, call get_income_statement_data for {SYMBOL} and summarize only the key points needed for Buffett-style analysis.'})
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
        {"role": "system", "content": "You are a conservative, long-term value investor like Warren Buffett. Use a disciplined, fundamental approach focused on durable competitive advantages, high returns on capital, and margin of safety."},
        {"role": "user", "content": f'Key ratio summary for {SYMBOL}: {ratio_summary}\nIncome statement summary: {income_summary}'}
    ]

# Step 3: Repeat for balance sheet
messages.append({"role": "user", "content": f'Now, call get_balance_sheet_data for {SYMBOL} and summarize only the key points needed for Buffett-style analysis.'})
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
        {"role": "system", "content": "You are a conservative, long-term value investor like Warren Buffett. Use a disciplined, fundamental approach focused on durable competitive advantages, high returns on capital, and margin of safety."},
        {"role": "user", "content": f'Key ratio summary for {SYMBOL}: {ratio_summary}\nIncome statement summary: {income_summary}\nBalance sheet summary: {balance_summary}'}
]

# Step 4: Repeat for cash flow
messages.append({"role": "user", "content": f'Now, call get_cash_flow_data for {SYMBOL} and summarize only the key points needed for Buffett-style analysis.'})
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
        {"role": "system", "content": "You are a conservative, long-term value investor like Warren Buffett. Use a disciplined, fundamental approach focused on durable competitive advantages, high returns on capital, and margin of safety."},
        {"role": "user", "content": f'Key ratio summary for {SYMBOL}: {ratio_summary}\nIncome statement summary: {income_summary}\nBalance sheet summary: {balance_summary}\nCash flow summary: {cashflow_summary}'}
    ]

# Step 5: Final analysis
messages.append({
    "role": "user",
    "content": f"""Given the above summaries, provide a final Buffett-style investment analysis for {SYMBOL}.
Your analysis must be structured as a JSON object with the following keys: "recommendation", "confidence_score", "target_price", "stop_loss_price", "reasoning".

- "recommendation": One of: BUY, HOLD, SELL, or AVOID.
- "confidence_score": A numerical value between 0.0 (low) and 1.0 (high) representing your confidence in the recommendation.
- "target_price": Optional. Your estimated target price (e.g., 150.00). If not applicable, use null.
- "stop_loss_price": Optional. Your estimated stop-loss price (e.g., 120.00). If not applicable, use null.
- "reasoning": Brief reasoning for your recommendation.

**Important Constraints:**
- If you provide a Target Price or Stop-Loss Price, you **must** state that it is **your persona's (Warren Buffett) estimation based *only* on the financial data and summaries provided in this interaction**. Do not cite external, unverified sources (e.g., 'analyst consensus') unless such information was explicitly part of the data given to you in previous steps.
- Your reasoning and any price levels must be directly justifiable from the provided financial summaries.
- Be concise and actionable.
- Output *only* the JSON object and nothing else. Example:
  {{
    "recommendation": "BUY",
    "confidence_score": 0.85,
    "target_price": 150.00,
    "stop_loss_price": 120.00,
    "reasoning": "The company shows strong fundamentals and a significant margin of safety at the current price."
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

# Attempt to parse the JSON to ensure it's valid before saving
try:
    parsed_json = json.loads(structured_output_str)
    # Add agent name and ticker for completeness, useful for the extractor
    parsed_json['agent'] = "Buffett"
    parsed_json['ticker'] = SYMBOL
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from LLM: {e}")
    print(f"LLM output was: {structured_output_str}")
    # Fallback to saving raw output if JSON parsing fails, or handle error appropriately
    parsed_json = {"error": "Failed to parse LLM JSON output", "raw_output": structured_output_str, "agent": "Buffett", "ticker": SYMBOL}


# Save to file
output_file = os.path.join(data_folder, f"{SYMBOL}_buffett_analysis.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(parsed_json, f, indent=2)

print(f"\nAnalysis saved to {output_file}")
print("--------------------------------------------------------------------------------------------------------- End of Analysis ------------------------------------------------------------------------------------------")
# Print tool usage summary
print("\nTool usage summary:")
for tool, used in used_tools.items():
    print(f"{tool}: {used}")
# End of financials_agent.py