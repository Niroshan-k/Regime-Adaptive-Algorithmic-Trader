from src.data_loader import fetch_data
from src.features import add_technical_indicators, detect_regimes
from src.train import train_models
from src.backtest import run_backtest
import pandas as pd

def main():
    # 1. Get Data
    tickers = ['^GSPC', '^VIX', '^TNX', 'CL=F']
    data = fetch_data(tickers, '2010-01-01', '2024-01-01')
    df = pd.DataFrame(index=data.index)
    
    # 2. Add Intelligence
    df_macro = add_technical_indicators(data, df)
    df_pca = detect_regimes(df_macro)
    final_df = pd.concat([df_macro, df_pca.drop(columns=['PC1','PC2'])], axis=1)
    final_df.dropna(inplace=True)
    #print("\n Final Dataset for Model \n",final_df)
    
    # 3. Train Brains
    models = train_models(final_df)
    
    # 4. Prove it works
    final_pnl = run_backtest(df, models)
    #print(f"System Finished. Final PnL: {final_pnl}")

if __name__ == "__main__":
    main()