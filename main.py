import subprocess
import os

def extract_final_answer(report_path):
    if not os.path.exists(report_path):
        return None
    with open(report_path, encoding='utf-8') as f:
        content = f.read()
    if '---- Final Answer ----' in content:
        return content.split('---- Final Answer ----')[-1].strip().lower()
    return content.strip().lower()

def run_agent(script, symbol):
    os.environ['SYMBOL'] = symbol
    result = subprocess.run(['python', script], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 0

def main():
    # List of tickers (can be loaded from a file)
    tickers = [
    "ASTS"
    ]
    
    # Ensure data folder exists
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    
    # Fix the paths to the agent scripts
    agent_scripts = {
        'buffet': 'ai-hedge-fund - Backup - Copy/buffet_agent.py',
        'lynch': 'ai-hedge-fund - Backup - Copy/lynch_agent.py',
        'wood': 'ai-hedge-fund - Backup - Copy/wood_agent.py',
        'risk_manager': 'ai-hedge-fund - Backup - Copy/risk_manager_agent.py',
    }
    
    for symbol in tickers:
        print(f'\n================= Processing {symbol} =================')
        os.environ['SYMBOL'] = symbol
        avoid_hold_count = 0
        
        # 1. Run Buffet agent
        print(f'Running Buffet agent for {symbol}...')
        run_agent(agent_scripts['buffet'], symbol)
        buffet_report = extract_final_answer(os.path.join(data_folder, f'{symbol}_Warren_Buffet_analysis.txt'))
        if buffet_report is None:
            print('Buffet report missing, skipping ticker.')
            continue
        if 'sell' in buffet_report:
            print(f'Buffet agent says SELL. Skipping to next ticker.')
            continue
        if any(x in buffet_report for x in ['avoid', 'hold']):
            avoid_hold_count += 1
            
        # 2. Run Lynch agent
        print(f'Running Lynch agent for {symbol}...')
        run_agent(agent_scripts['lynch'], symbol)
        lynch_report = extract_final_answer(os.path.join(data_folder, f'{symbol}_lynch_analysis.txt'))
        if lynch_report is None:
            print('Lynch report missing, skipping ticker.')
            continue
        if 'sell' in lynch_report:
            print(f'Lynch agent says SELL. Skipping to next ticker.')
            continue
        if any(x in lynch_report for x in ['avoid', 'hold']):
            avoid_hold_count += 1
        if avoid_hold_count >= 2:
            print(f'Two agents said AVOID or HOLD. Skipping to next ticker.')
            continue
            
        # 3. Run Wood agent
        print(f'Running Wood agent for {symbol}...')
        run_agent(agent_scripts['wood'], symbol)
        wood_report = extract_final_answer(os.path.join(data_folder, f'{symbol}_Wood_analysis.txt'))
        if wood_report is None:
            print('Wood report missing, skipping ticker.')
            continue
        if 'sell' in wood_report:
            print(f'Wood agent says SELL. Skipping to next ticker.')
            continue
        if any(x in wood_report for x in ['avoid', 'hold']):
            avoid_hold_count += 1
        if avoid_hold_count >= 2:
            print(f'Two agents said AVOID or HOLD. Skipping to next ticker.')
            continue
            
        # 4. Run Risk Manager agent
        print(f'Running Risk Manager agent for {symbol}...')
        run_agent(agent_scripts['risk_manager'], symbol)
        print(f'Completed full workflow for {symbol}.')

if __name__ == '__main__':
    main() 