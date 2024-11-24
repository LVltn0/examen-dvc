import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# load data
X_train_path = "data/processed_data/X_train.csv"
X_test_path = "data/processed_data/X_test.csv"
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
print(X_train.head())
print(X_test.head())

# remove non numeric columns
columns_to_scale = X_train.select_dtypes(include=["float64", "int64"]).columns

# normalize data
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

# save data
output_train_path = "data/processed_data/X_train_scaled.csv"
output_test_path = "data/processed_data/X_test_scaled.csv"
os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
X_train_scaled.to_csv(output_train_path, index=False)
X_test_scaled.to_csv(output_test_path, index=False)