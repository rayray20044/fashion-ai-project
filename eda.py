import pandas as pd

# Load the dataset (skipping bad lines just in case of formatting quirks)
df = pd.read_csv("styles.csv", on_bad_lines='skip')

# Display the first 5 rows
print("--- FIRST 5 ROWS ---")
print(df.head())