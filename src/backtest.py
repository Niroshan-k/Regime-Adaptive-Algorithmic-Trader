import numpy as np

def run_backtest(df, models):
    print("--- RUNNING VECTORIZED BACKTEST ---")
    df['Signal'] = 0
    for cluster_id, model in models.items():
        idxs = df.index[df['Cluster'] == cluster_id]
        if len(idxs) == 0: continue

        X_regime = df.loc[idxs].drop(columns=['Target','Cluster','SMA_50','Signal'])

        #predicts -> 1 =up, 0 = down
        preds = model.predict(X_regime)

        acc = model.accuracy_score(df.loc[idxs, 'Target'], preds)
        strategy = "INVERT" if acc < 0.50 else "FOLLOW"
        print(f"Regime {cluster_id} ({strategy}): Accuracy {acc*100:.2f}%")

        if strategy == "FOLLOW":
            # Pred 1 -> Signal 1 (Long)
            # Pred 0 -> Signal 0 (Cash) - Safer than shorting in a bull regime
            signals = np.where(preds == 1, 1, 0)
        else:
            # INVERT (Mean Reversion)
            # Pred 1 (Model says Up) -> Signal -1 (We Short)
            # Pred 0 (Model says Down) -> Signal 1 (We Buy)
            signals = np.where(preds == 1, -1, 1)

        df.loc[idxs, 'Signal'] = signals