import pandas as pd
from sklearn.model_selection import train_test_split

# load data
file_path = "data/raw_data/raw.csv"
data = pd.read_csv(file_path)
print(data.head())
print(data.info())

# split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# save processed data
save_path = "data/processed_data"
X_train.to_csv(f"{save_path}/X_train.csv", index=False)
X_test.to_csv(f"{save_path}/X_test.csv", index=False)
y_train.to_csv(f"{save_path}/y_train.csv", index=False)
y_test.to_csv(f"{save_path}/y_test.csv", index=False)

print(f"Les fichiers ont été sauvegardés dans le dossier: {save_path}")