import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

# Load CSV file
df = pd.read_csv("lungdataset.csv")

print("Original Dataset")
print(df.head())

# a. Attribute Selection
df = df.drop(columns=['GENDER'])

print("\nAfter Attribute Selection")
print(df.head())

# b. Handling Missing Values
num_cols = df.select_dtypes(include=[np.number]).columns

imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

print("\nAfter Handling Missing Values")
print(df.head())

# c. Discretization (AGE into 3 bins)
discretizer = KBinsDiscretizer(
    n_bins=3,
    encode='ordinal',
    strategy='uniform'
)

df['AGE_GROUP'] = discretizer.fit_transform(df[['AGE']])

print("\nAfter Discretization")
print(df[['AGE', 'AGE_GROUP']].head())

# d. Elimination of Outliers (using AGE)
Q1 = df['AGE'].quantile(0.25)
Q3 = df['AGE'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['AGE'] >= Q1 - 1.5 * IQR) &
        (df['AGE'] <= Q3 + 1.5 * IQR)]

print("\nAfter Eliminating Outliers")
print(df.head())

print("\nFinal Preprocessed Dataset")
print(df)
