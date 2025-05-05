import json
from openai import OpenAI
import os
from functions import get_income_statement_data, get_balance_sheet_data, get_cash_flow_data, get_ratio_data


# Define folder path
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)  # Ensure data folder exists
symbol = os.environ.get('SYMBOL', 'ASTS')  # Add default value

# Initialize client
client = OpenAI(api_key="sk-39cbd05d862749d8bca6c13f1bc234ee", base_url="https://api.deepseek.com")
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
    {"role": "user", "content": f'First, call get_ratio_data for {symbol} and summarize only the key ratios needed for Buffett-style analysis.'}
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
            data = get_ratio_data(symbol)
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
        {"role": "user", "content": f'Key ratio summary for {symbol}: {ratio_summary}'}
    ]

# Step 2: Repeat for income statement
messages.append({"role": "user", "content": f'Now, call get_income_statement_data for {symbol} and summarize only the key points needed for Buffett-style analysis.'})
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
            data = get_income_statement_data(symbol)
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
        {"role": "user", "content": f'Key ratio summary for {symbol}: {ratio_summary}\nIncome statement summary: {income_summary}'}
    ]

# Step 3: Repeat for balance sheet
messages.append({"role": "user", "content": f'Now, call get_balance_sheet_data for {symbol} and summarize only the key points needed for Buffett-style analysis.'})
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
            data = get_balance_sheet_data(symbol)
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
        {"role": "user", "content": f'Key ratio summary for {symbol}: {ratio_summary}\nIncome statement summary: {income_summary}\nBalance sheet summary: {balance_summary}'}
]

# Step 4: Repeat for cash flow
messages.append({"role": "user", "content": f'Now, call get_cash_flow_data for {symbol} and summarize only the key points needed for Buffett-style analysis.'})
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
            data = get_cash_flow_data(symbol)
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
        {"role": "user", "content": f'Key ratio summary for {symbol}: {ratio_summary}\nIncome statement summary: {income_summary}\nBalance sheet summary: {balance_summary}\nCash flow summary: {cashflow_summary}'}
    ]

# Step 5: Final analysis
messages.append({"role": "user", "content": f"Given the above summaries, provide a final Buffett-style investment analysis for {symbol}.\n\nYour FINAL RECOMMENDATION must be one of: BUY, HOLD, SELL, or AVOID (use only one, no synonyms, and use it only once in a clearly marked 'Final Recommendation' section at the end of your output). Do not use these words anywhere else in your output. Be concise, structured, and actionable."})
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)
response_message = response.choices[0].message
reasoning = getattr(response_message, 'reasoning_content', '')
content = getattr(response_message, 'content', '')

print("--------------------------------------------------------------------------------------------------------- Reasoning ---------------------------------------------------------------------------------------------")
print(reasoning)
print("\n------------------------------------------------------------------------------------------------------- Final Answer -------------------------------------------------------------------------------------------")
print(content)

# Save to file
output_file = os.path.join(data_folder, f"{symbol}_Warren_Buffet_analysis.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("---- Reasoning ----\n")
    f.write(reasoning + "\n\n")
    f.write("---- Final Answer ----\n")
    f.write(content)

print(f"\nAnalysis saved to {output_file}")
print("--------------------------------------------------------------------------------------------------------- End of Analysis ------------------------------------------------------------------------------------------")
# Print tool usage summary
print("\nTool usage summary:")
for tool, used in used_tools.items():
    print(f"{tool}: {used}")
# End of financials_agent.py