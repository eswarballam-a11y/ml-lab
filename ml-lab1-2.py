import pandas as pd
from IPython.display import display
df = pd.read_csv("employee_salary_dataset.csv")
print("First 5 Rows:")
display(df.head())
print("Central Tendency")
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True).iloc[0])
print("Measure of Dispersion")
print("\nVariance:\n", df.var(numeric_only=True))
print("\nStandard Deviation:\n", df.std(numeric_only=True))