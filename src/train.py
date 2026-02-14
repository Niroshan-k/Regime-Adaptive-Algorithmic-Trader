import xgboost as xgb
import joblib
# ... imports ...

def train_models(df):
    models = {}
    for cluster_id in range(2):
        # ... Paste your Training Loop here ...
        # Save the model
        joblib.dump(model, f"models/regime_{cluster_id}.gz")
        models[cluster_id] = model
    return models