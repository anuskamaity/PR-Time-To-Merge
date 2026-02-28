import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Load data
df = pd.read_csv("data/prs_processed.csv")

# 2. Define label and features
y = df['time_to_merge_hours']
leakage_cols = ['pr_number', 'total_comments', 'time_to_merge_hours', 'id']
X = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# THE TRIPLE SPLIT

# 85% goes to temp, 15% goes to test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Split the 85% into train and valid
# We take about 18% of the temp set to end up with 15% of the total
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

print(f"Training set:   {len(X_train)} PRs")
print(f"Validation set: {len(X_val)} PRs")
print(f"Test set:       {len(X_test)} PRs")

# 3. Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Use the validation set
val_predictions = model.predict(X_val)
val_error = mean_absolute_error(y_val, val_predictions)
print(f"Validation Error: {round(val_error, 2)} hours")

baseline_guess = y_train.mean()
baseline_error = (y_val - baseline_guess).abs().mean()
print(f"Baseline Error (Guessing Average): {round(baseline_error, 2)} hours")

# Final run on test set (first commented out then uncommented once we're happy with validation performance)
test_predictions = model.predict(X_test)
test_error = mean_absolute_error(y_test, test_predictions)
print(f"Final Test Error: {round(test_error, 2)} hours")

# 6. Save
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")