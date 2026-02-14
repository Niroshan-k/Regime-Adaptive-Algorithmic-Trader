from src.data_loader import fetch_data
from src.features import add_technical_indicators, detect_regimes
from src.train import train_models
from src.backtest import run_backtest

def main():
    # 1. Get Data
    df = fetch_data(['^GSPC', '^VIX'], '2010-01-01', '2024-01-01')
    
    # 2. Add Intelligence
    df = add_technical_indicators(df)
    df = detect_regimes(df)
    
    # 3. Train Brains
    models = train_models(df)
    
    # 4. Prove it works
    final_pnl = run_backtest(df, models)
    print(f"System Finished. Final PnL: {final_pnl}")

if __name__ == "__main__":
    main()