#importing neccessary libaries
import pandas as pd
import numpy as np

df=pd.read_csv(r'C:\Users\Abraham\Desktop\DOINGS\Housing.csv')

print("\ndata information")
print(df.info())

print("\ndata head")
print(df.head())

# Check for missing values
print("\nmissing_values") 
print(df.isnull().sum())

# Summary statistics for numerical columns
print("\nsummary_stats")
print(df.describe())

# Checking unique values in categorical columns

# Check unique values in categorical columns
categorical_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", 
                       "airconditioning", "prefarea", "furnishingstatus"]

categorical_summary = {col: df[col].value_counts() for col in categorical_columns}
categorical_summary

for col, values in categorical_summary.items():
    print(f"\n{col}:\n{values}")
    
    
# Convert 'yes'/'no' categorical variables to binary (1/0)
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
df[binary_cols] 

print(df[binary_cols].apply(lambda x: x.map({"yes": 1, "no": 0})))
    
    
# Check for outliers

# Convert all numeric columns to proper data types
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Now check for outliers again
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("\nNumber of outliers detected in each column:")
print(outliers)

print("\nCleaned dataset (first 5 rows):")
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

#  plot style
sns.set(style="whitegrid")

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()


# Price Distribution Plot
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=30, kde=True, color="blue")
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Scatter Plots of Key Features vs. Price
key_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=df[feature], y=df["price"], alpha=0.6)
    plt.title(f"{feature} vs. Price")

plt.tight_layout()
plt.show()





