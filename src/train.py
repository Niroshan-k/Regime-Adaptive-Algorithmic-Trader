import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_models(df):
    models = {}
    for cluster_id in range(2):
        print(f"\nTraining Trader for Regime {cluster_id}...")
        
        #get only the days belong to this regime
        regime_data = df[df['Cluster'] == cluster_id]
        X = regime_data.drop(columns = ['Target', 'Cluster', 'SMA_50'], errors='ignore')
        y = regime_data['Target']

        # need enough data to train
        if len(X) < 50:
            print(f"Skipping Cluster {cluster_id} (not enough data)")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=False)

        #train
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,random_state=42)
        model.fit(X_train, y_train)

        #evaluate
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"  Trader {cluster_id} Accuracy: {accuracy*100:.2f}%")
        
        models[cluster_id] = model
        joblib.dump(model, f'models/xgb_regime_{cluster_id}.gz')
        print(f"  💾 Saved models/xgb_regime_{cluster_id}.gz")
    return models