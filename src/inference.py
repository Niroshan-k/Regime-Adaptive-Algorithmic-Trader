import pandas as pd
import joblib
import os
from datetime import datetime
from src.data_loader import fetch_data
from src.features import add_technical_indicators

def get_latest_signal():
    # 1. LOAD THE SAVED BRAINS
    # We load the exact same models you trained in the notebook
    print("Loading models...")
    if not os.path.exists('models/scaler.gz'):
        print("❌ Error: Models not found! Run main.py first to train them.")
        return

    scaler = joblib.load('models/scaler.gz')
    pca = joblib.load('models/pca.gz')
    kmeans = joblib.load('models/kmeans.gz')
    
    # Load the 2 XGBoost Traders
    traders = {}
    for i in range(2):
        traders[i] = joblib.load(f'models/xgb_regime_{i}.gz')

    # 2. GET FRESH DATA (Live from Yahoo)
    print("Fetching live market data...")
    # We fetch the last 365 days to ensure we have enough data for the 50-day SMA
    # auto_adjust=False ensures we get the raw Close price, same as training
    df_raw = fetch_data(['^GSPC', '^VIX', '^TNX', 'CL=F'], '2023-01-01', datetime.now().strftime('%Y-%m-%d'))
    
    # 3. CALCULATE FEATURES (Exact same math as training)
    df_macro = pd.DataFrame(index=df_raw.index)
    df_macro = add_technical_indicators(df_raw, df_macro)
    
    # We only care about the very last row (Today's Market Close)
    latest_data = df_macro.tail(1)
    today_date = latest_data.index[0].strftime('%Y-%m-%d')
    print(f"Analyzing Market State for: {today_date}")

    # 4. DETECT REGIME (The Manager)
    # We filter only the macro columns, just like in training
    macro_cols = ['SP500_Return', 'SP500_Vol', 'VIX', 'Bond_Yield', 'Oil_Change']
    X_macro = latest_data[macro_cols]
    
    # Transform (Do NOT fit!) using the saved scaler and PCA
    X_scaled = scaler.transform(X_macro) 
    X_pca = pca.transform(X_scaled)
    current_regime = kmeans.predict(X_pca)[0]
    
    print(f"Detected Regime: {current_regime}")

    # 5. GET PREDICTION (The Specialist)
    # Prepare features for XGBoost (Drop non-features)
    features = latest_data.drop(columns=['SMA_50'], errors='ignore')
    
    # Select the correct Specialist for this regime
    model = traders[current_regime]
    prediction = model.predict(features)[0] # 1 = Up, 0 = Down
    
    # 6. INTERPRET SIGNAL (The Strategy Logic)
    # logic: Regime 0 (Bull) -> Follow, Regime 1 (Bear/Choppy) -> Invert
    # (Adjust this based on your final backtest results!)
    if current_regime == 0: # Bull
        signal = "BUY (Long)" if prediction == 1 else "CASH (Flat)"
    else: # Bear / Choppy
        signal = "SELL (Short)" if prediction == 1 else "BUY (Long)" # Inverted
    
    print(f"AI Prediction: {prediction}")
    print(f"FINAL SIGNAL: {signal}")
    
    # 7. SAVE TO LOG (The Database)
    log_entry = {
        'Date': today_date,
        'Regime': current_regime,
        'Raw_AI_Pred': prediction,
        'Final_Signal': signal,
        'Close_Price': df_raw['^GSPC'].iloc[-1]
    }
    
    log_df = pd.DataFrame([log_entry])
    
    # Append to 'trade_log.csv'
    log_file = 'trade_log.csv'
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
        
    print("✅ Saved to trade_log.csv")

if __name__ == "__main__":
    get_latest_signal()