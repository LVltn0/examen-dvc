import pandas as pd
import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error

# data paths
train_data_path = "data/processed_data/X_train_scaled.csv"
test_data_path = "data/processed_data/X_test_scaled.csv"
train_target_path = "data/processed_data/y_train.csv"
test_target_path = "data/processed_data/y_test.csv"

# load data
X_train = pd.read_csv(train_data_path)
X_test = pd.read_csv(test_data_path)
y_train = pd.read_csv(train_target_path).squeeze()
y_test = pd.read_csv(test_target_path).squeeze()

# init model
model = RandomForestRegressor(random_state=42)

# Définir la grille de recherche
param_grid = {
    "n_estimators": [50, 100, 200],
}

# Scorer (rmse min)
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# gridsearch init
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)

# gridsearch run
print("Running GridSearch...")
grid_search.fit(X_train, y_train)

# best params
print("Best parameters found: ", grid_search.best_params_)
best_params = grid_search.best_params_

# save model
output_path = "models/best_params.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"Best parameters saved to {output_path}")
