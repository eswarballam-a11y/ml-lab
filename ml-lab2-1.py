import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
# Create a sample data set
data = {
'ID': [1, 2, 3, 4, 5],
'Age': [22, 25, np.nan, 40, 120],   # 120 is an outlier
'Salary': [30000, 35000, 40000, np.nan, 1000000],  # outlier
'Experience': [1, 2, 3, 10, 15]
}
df = pd.DataFrame(data)
print("Original Dataset")
print(df)
#a. Attribute Selection
df = df.drop(columns=['ID'])
print("\nAfter Attribute Selection")
print(df)
#b. Handling Missing Values
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])
print("\nAfter Handling Missing Values")
print(df)
#c. Discretization
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df['Age_Group'] = discretizer.fit_transform(df[['Age']])
print("\nAfter Discretization")
print(df)
#d. Elimination of Outliers
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Salary'] >= Q1 - 1.5 * IQR) & (df['Salary'] <= Q3 + 1.5 * IQR)]
print("\nAfter Eliminating Outliers")
print(df)
print("\nFinal Preprocessed Dataset")
print(df)