import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def run_backtest(df, models):
    print("--- RUNNING VECTORIZED BACKTEST ---")
    
    # Create a copy to avoid SettingWithCopy warnings on the original df
    df = df.copy() 
    df['Signal'] = 0
    
    for cluster_id, model in models.items():
        idxs = df.index[df['Cluster'] == cluster_id]
        if len(idxs) == 0: continue

        # FIXED: Added errors='ignore' in case 'SMA_50' isn't in this specific dataframe view
        X_regime = df.loc[idxs].drop(columns=['Target','Cluster','SMA_50','Signal'], errors='ignore')

        # Predict
        preds = model.predict(X_regime)

        # FIXED: Corrected accuracy_score usage
        acc = accuracy_score(df.loc[idxs, 'Target'], preds)
        
        strategy = "INVERT" if acc < 0.50 else "FOLLOW"
        print(f"Regime {cluster_id} ({strategy}): Accuracy {acc*100:.2f}%")

        if strategy == "FOLLOW":
            signals = np.where(preds == 1, 1, 0)
        else:
            signals = np.where(preds == 1, -1, 1)

        df.loc[idxs, 'Signal'] = signals
    
    # Move signals forward by 1 day
    df['Signal_Shifted'] = df['Signal'].shift(1)

    # Strategy Return
    df['Strategy_Return'] = df['Signal_Shifted'] * df['SP500_Return']

    # Equity Curve
    initial_capital = 10000
    df['Equity_Curve'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
    df['Benchmark'] = initial_capital * (1 + df['SP500_Return']).cumprod()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Equity_Curve'], label='Algorithm', color='green', linewidth=2)
    plt.plot(df['Benchmark'], label='S&P 500', color='gray', linestyle='--', alpha=0.5)
    plt.title('Corrected Backtest (Vectorized)')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final Stats
    final_val = df['Equity_Curve'].iloc[-1]
    total_ret = ((final_val - initial_capital) / initial_capital) * 100
    print(f"Final Value: ${final_val:,.2f}")
    print(f"Total Return: {total_ret:.2f}%")
    
    return final_val