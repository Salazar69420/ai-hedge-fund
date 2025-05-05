import json
import os
from openai import OpenAI
from functions import get_ratio_data, get_news_data, get_earnings_transcripts_data, get_insider_trading_data

# Config
symbol = os.environ.get('SYMBOL', 'ASTS')
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)  # Ensure data folder exists
model = "deepseek-chat"
context = "You are a risk manager. Synthesize agent reports and tool data to output buy/hold/sell price levels and key fundamental/technical levels. Do not form your own opinion—use only the agent reports and tool data."

def extract_final_answer(report):
    if "---- Final Answer ----" in report:
        return report.split("---- Final Answer ----")[-1].strip()
    return report

# Read agent reports
buffet_report = extract_final_answer(open(os.path.join(data_folder, f"{symbol}_Warren_Buffet_analysis.txt"), encoding="utf-8").read())
lynch_report = extract_final_answer(open(os.path.join(data_folder, f"{symbol}_lynch_analysis.txt"), encoding="utf-8").read())
wood_report = extract_final_answer(open(os.path.join(data_folder, f"{symbol}_Wood_analysis.txt"), encoding="utf-8").read())

# Define tools (only ratios, news, earnings transcripts, insider trading)
TOOLS = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_data",
            "description": "Retrieve news sentiment data for a given symbol",
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
            "name": "get_earnings_transcripts_data",
            "description": "Retrieve earnings transcripts data for a given symbol",
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
            "name": "get_insider_trading_data",
            "description": "Retrieve insider trading data for a given symbol",
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

used_tools = {tool["function"]["name"]: False for tool in TOOLS}

client = OpenAI(api_key="sk-39cbd05d862749d8bca6c13f1bc234ee", base_url="https://api.deepseek.com")

# Start with agent summaries
messages = [
    {"role": "system", "content": context},
    {"role": "user", "content": f"You are a risk manager. Your job is to synthesize the following agent reports and all available tool data for {symbol}.\n\nBuffet Agent Report:\n{buffet_report}\n\nLynch Agent Report:\n{lynch_report}\n\nWood Agent Report:\n{wood_report}\n\nYou must analyze all fundamental and technical aspects from both the agent reports and the tool data.\n\nYour output at the end must be:\n- Buy, hold, and sell price levels for {symbol}, with a clear price consensus based on all data.\n- Key fundamental and technical levels to watch.\n- The analysis must be strictly based on the data from tools and agent reports, with NO personal opinion or speculation.\n- Output must be concise, structured, and actionable for a portfolio manager.\n- Do not add any extra commentary or disclaimers.\n\nYou will now call tools and summarize each result for synthesis."}
]

# Step 1: Ratios
messages.append({"role": "user", "content": f"First, call get_ratio_data for {symbol} and summarize only the key ratios needed for risk management synthesis."})
response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
response_message = response.choices[0].message
if getattr(response_message, 'tool_calls', None):
    messages.append(response_message)
    for tool in response_message.tool_calls:
        if tool.function.name == "get_ratio_data":
            data = get_ratio_data(symbol)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_ratio_data"] = True
    response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    ratio_summary = response.choices[0].message.content
    messages = messages[:2] + [{"role": "user", "content": f"Key ratio summary for {symbol}: {ratio_summary}"}]
else:
    ratio_summary = ""

# Step 2: News
messages.append({"role": "user", "content": f"Now, call get_news_data for {symbol} and summarize only the key points needed for risk management synthesis."})
response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
response_message = response.choices[0].message
if getattr(response_message, 'tool_calls', None):
    messages.append(response_message)
    for tool in response_message.tool_calls:
        if tool.function.name == "get_news_data":
            data = get_news_data(symbol)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_news_data"] = True
    response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    news_summary = response.choices[0].message.content
    messages = messages[:2] + [{"role": "user", "content": f"Key ratio summary for {symbol}: {ratio_summary}\nNews summary: {news_summary}"}]
else:
    news_summary = ""

# Step 3: Earnings Transcripts
messages.append({"role": "user", "content": f"Now, call get_earnings_transcripts_data for {symbol} and summarize only the key points needed for risk management synthesis."})
response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
response_message = response.choices[0].message
if getattr(response_message, 'tool_calls', None):
    messages.append(response_message)
    for tool in response_message.tool_calls:
        if tool.function.name == "get_earnings_transcripts_data":
            data = get_earnings_transcripts_data(symbol)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_earnings_transcripts_data"] = True
    response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    earnings_transcripts_summary = response.choices[0].message.content
    messages = messages[:2] + [{"role": "user", "content": f"Key ratio summary for {symbol}: {ratio_summary}\nNews summary: {news_summary}\nEarnings transcripts summary: {earnings_transcripts_summary}"}]
else:
    earnings_transcripts_summary = ""

# Step 4: Insider Trading
messages.append({"role": "user", "content": f"Now, call get_insider_trading_data for {symbol} and summarize only the key points needed for risk management synthesis."})
response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
response_message = response.choices[0].message
if getattr(response_message, 'tool_calls', None):
    messages.append(response_message)
    for tool in response_message.tool_calls:
        if tool.function.name == "get_insider_trading_data":
            data = get_insider_trading_data(symbol)
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": json.dumps(data, separators=( ',', ':' ))})
            used_tools["get_insider_trading_data"] = True
    response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    insider_trading_summary = response.choices[0].message.content
    messages = messages[:2] + [{"role": "user", "content": f"Key ratio summary for {symbol}: {ratio_summary}\nNews summary: {news_summary}\nEarnings transcripts summary: {earnings_transcripts_summary}\nInsider trading summary: {insider_trading_summary}"}]
else:
    insider_trading_summary = ""

# Final synthesis
messages.append({"role": "user", "content": f"Given all the above summaries, synthesize a concise, structured risk manager report for {symbol}. Output:\n- Buy, hold, and sell price levels\n- Key fundamental and technical levels to watch\n- Synthesize only from agent reports and tool data. Do NOT form your own opinion."})
response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
response_message = response.choices[0].message
reasoning = getattr(response_message, 'reasoning_content', '')
content = getattr(response_message, 'content', '')

print("--------------------------------------------------------------------------------------------------------- Reasoning ---------------------------------------------------------------------------------------------")
print(reasoning)
print("\n------------------------------------------------------------------------------------------------------- Final Answer -------------------------------------------------------------------------------------------")
print(content)

# Save to file
output_file = os.path.join(data_folder, f"{symbol}_Risk_Manager_analysis.txt")
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

print("Note: DeepSeek context caching is enabled automatically.") 