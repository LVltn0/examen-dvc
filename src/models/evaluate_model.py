import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

# Load trained model
model = joblib.load("models/trained_model.pkl")

# Load scaled features and target
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

# make predictions
y_pred = model.predict(X_test)

# save predictions to csv
predictions_df = pd.DataFrame({
    'y_true': y_test.values.flatten(),
    'y_pred': y_pred.flatten()
})
predictions_path = "data/processed_data/predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

# eval model perf
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# save metrics to json
metrics = {
    'mse': mse,
    'r2': r2
}

os.makedirs("metrics", exist_ok=True)
metrics_path = "metrics/scores.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f)

print("Evaluation completed. Metrics and predictions saved.")
