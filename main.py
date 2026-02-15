from src.data_loader import fetch_data
from src.features import add_technical_indicators, detect_regimes
from src.train import train_models
from src.backtest import run_backtest
import pandas as pd

def main():
    # 1. Get Data
    tickers = ['^GSPC', '^VIX', '^TNX', 'CL=F']
    # Ensure this matches your data_loader logic
    data = fetch_data(tickers, '2010-01-01', '2024-01-01')
    
    df = pd.DataFrame(index=data.index)
    
    # 2. Add Intelligence
    # df_macro now technicals + macro
    df_macro = add_technical_indicators(data, df)
    
    # df_pca now Clusters + PC1/PC2 + Target
    df_pca = detect_regimes(df_macro)
    
    # Combine them. 
    # drop PC1/PC2 because XGBoost doesn't need them (it uses raw features)
    # We need 'Cluster' from df_pca and features from df_macro
    final_df = pd.concat([df_macro, df_pca[['Cluster', 'Target']]], axis=1)
    final_df.dropna(inplace=True)
    
    # 3. Train
    models = train_models(final_df)
    
    # 'final_df' has the 'Cluster' column required by the backtester.
    final_pnl = run_backtest(final_df, models)
    
    print(f"System Finished. Final PnL: {final_pnl:,.2f}")

if __name__ == "__main__":
    main()