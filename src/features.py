import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

def add_technical_indicators(data, df, price_col='^GSPC'):
    # ... Paste your RSI, SMA, Momentum logic here ...
    df['SP500_Return'] = data[price_col].pct_change()
    df['SP500_Vol'] = df['SP500_Return'].rolling(20).std()
    df['VIX'] = data['^VIX'] / 100
    df['Bond_Yield'] = data['^TNX'] / 100
    df['Oil_Change'] = data['CL=F'].pct_change()
    df['SMA_50'] = data[price_col].rolling(window=50).mean()
    df['Above_SMA'] = (data[price_col] > df['SMA_50']).astype(int)
    df['Momentum'] = data[price_col].pct_change(periods=5)
    df['RSI'] = calculate_rsi(data[price_col])

    df.dropna(inplace=True)
    return df

def detect_regimes(df):
    # ... Paste your Scaling, PCA, and K-Means logic here ...
    # Return the dataframe with a new 'Cluster' column

    scaler = StandardScaler()
    pca = PCA(n_components=2)
    scaled_data = scaler.fit_transform(df)
    pca_result = pca.fit_transform(scaled_data)
    
    df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=df.index)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Captured Variance PCA: {explained_variance*100:.2f}%")

    #print("\n--- PCA Loadings (What makes up the new axes?) ---")
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=df.columns)
    #print(loadings)

    kmeans_pca = KMeans(n_clusters=2, random_state=42)
    df_pca['Cluster'] = kmeans_pca.fit_predict(df_pca)
    df_pca['Target'] = (df['SP500_Return'].shift(-1) > 0).astype(int)

    print("Saving models...")
    joblib.dump(scaler, 'scaler.gz')   # Save the scaling logic
    joblib.dump(pca, 'pca.gz')         # Save the compression logic
    joblib.dump(kmeans_pca, 'kmeans.gz') # Save the clustering brain
    print("Phase 1 Complete: Models Saved!")

    return df_pca

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))