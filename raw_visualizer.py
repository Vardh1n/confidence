import pandas as pd
import json

# File path
file_path = "dataset.jsonl"

# Initialize an empty list to store a subset of the data
subset = []
max_lines = 1000  # Number of lines to load for visualization

# Read the file line by line
with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        if i >= max_lines:
            break
        subset.append(json.loads(line))

# Convert the subset into a Pandas DataFrame
df = pd.DataFrame(subset)

# Display the first few rows of the DataFrame
print(df.head())

# Display the head of each column one by one
for column in df.columns:
    print(f"\nColumn: {column}")
    print(df[column][1])