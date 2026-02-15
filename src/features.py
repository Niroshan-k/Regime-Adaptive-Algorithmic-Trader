import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

def add_technical_indicators(data, df, price_col='^GSPC'):
    df = df.copy()
    
    df['SP500_Return'] = data['^GSPC'].pct_change()
    df['SP500_Vol'] = df['SP500_Return'].rolling(20).std()
    df['VIX'] = data['^VIX'] / 100
    df['Bond_Yield'] = data['^TNX'] / 100
    df['Oil_Change'] = data['CL=F'].pct_change()
    
    # Technicals
    df['SMA_50'] = data['^GSPC'].rolling(window=50).mean()
    df['Above_SMA'] = (data['^GSPC'] > df['SMA_50']).astype(int)
    df['Momentum'] = data['^GSPC'].pct_change(periods=5)
    df['RSI'] = calculate_rsi(data['^GSPC'])

    df.dropna(inplace=True)
    return df

def detect_regimes(df):
    macro_cols = ['SP500_Return', 'SP500_Vol', 'VIX', 'Bond_Yield', 'Oil_Change']
    X_macro = df[macro_cols]

    scaler = StandardScaler()
    # Fit on X_macro
    scaled_data = scaler.fit_transform(X_macro)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=df.index)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Captured Variance PCA: {explained_variance*100:.2f}%")

    kmeans_pca = KMeans(n_clusters=2, random_state=42)
    df_pca['Cluster'] = kmeans_pca.fit_predict(df_pca)
    
    # Add Target (Next Day Return > 0)
    df_pca['Target'] = (df['SP500_Return'].shift(-1) > 0).astype(int)

    print("Saving models...")
    joblib.dump(scaler, 'models/scaler.gz')      
    joblib.dump(pca, 'models/pca.gz')            
    joblib.dump(kmeans_pca, 'models/kmeans.gz')
    print("Phase 1 Complete: Models Saved!")

    return df_pca

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))