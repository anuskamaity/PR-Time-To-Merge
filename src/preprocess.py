import pandas as pd
import os

# 1. Load the data
INPUT_FILE = "data/prs_merged_cleaned.csv"
OUTPUT_FILE = "data/prs_processed.csv"

if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    exit()

df = pd.read_csv(INPUT_FILE)

# 2. Target Variable (Time to Merge in Hours)
df["created_at"] = pd.to_datetime(df["created_at"])
df["merged_at"] = pd.to_datetime(df["merged_at"])
df["time_to_merge_hours"] = (df["merged_at"] - df["created_at"]).dt.total_seconds() / 3600

# 3. Outlier Removal
# Keep only PRs that merged within 30 days. 
# Long-stale PRs (months/years) act as noise
df = df[df["time_to_merge_hours"] < (24 * 30)] 
# Also remove PRs with negative time (rare API glitches)
df = df[df["time_to_merge_hours"] >= 0]

# 4. Feature Engineering
# Convert boolean 'is_draft' to 0/1
if 'is_draft' in df.columns:
    df['is_draft'] = df['is_draft'].astype(int)

# 5. Categorical Encoding (One-Hot Encoding)
# Encode 'author_assoc' and 'repo'
categorical_cols = ["author_assoc", "repo"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 6. Drop columns we can't use in a mathematical model
cols_to_drop = ["created_at", "merged_at", "id"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 7. Save the processed data
df.to_csv(OUTPUT_FILE, index=False)

print(f"Original records: {len(pd.read_csv(INPUT_FILE))}")
print(f"Cleaned records:  {len(df)}")
print(f"Features created: {list(df.columns)}")
print(f"Saved to: {OUTPUT_FILE}")