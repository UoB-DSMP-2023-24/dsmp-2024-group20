# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:56:01 2024

@author: Vishal Chavan
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset from a CSV file
# Make sure to update the file path to match the actual file location
df = pd.read_csv('test/data/data/total_lob_300.csv')

# Display the first few rows of the dataset for an initial inspection
print(df.head())

# Combine 'date' with an 8 AM time offset, and add 'time_window' to create a new 'datetime' column
df["datetime"] = pd.to_datetime(df["date"] + " 08:00:00") + pd.to_timedelta(df['time_window'], unit='s')

# Check for missing values in each column
print(df.isna().sum())

# Remove columns that are not needed for analysis
df = df.drop(["time_window", "label", "date"], axis=1)

# Visualize the distribution of the 'avg_price' column using a density plot
df['avg_price'].plot.density()
plt.title('Density Plot of Average Price')
plt.xlabel('Average Price')
plt.ylabel('Density')
plt.show()

# Display summary statistics of the numerical columns
print(df.describe())

# Visualize the data trends over time for all numerical columns
df.plot(subplots=True, layout=(-1, 4), figsize=(15, 10), sharex=True)
plt.tight_layout()
plt.show()

# Drop the 'datetime' column to facilitate further analysis
df_scaled = df.drop('datetime', axis=1)

# Calculate the Pearson correlation matrix for the numerical columns
pearson_corr = df_scaled.corr(method='pearson')

# Calculate the Spearman correlation matrix for the numerical columns
spearman_corr = df_scaled.corr(method='spearman')

# Visualize the Pearson correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Pearson Correlation Matrix')
plt.show()

# Visualize the Spearman correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Spearman Correlation Matrix')
plt.show()

# Drop any remaining missing values from the numerical dataset
df_scaled = df_scaled.dropna()
df = df.drop('datetime', axis=1)

# Check for any infinite values in the dataset
print(df[df == np.inf].count())

# Import the function to calculate Pearson correlation p-values
from scipy.stats import pearsonr

# Initialize a DataFrame to store the p-values for each correlation pair
p_values = pd.DataFrame(index=df.columns, columns=df.columns)

# Calculate p-values for each pair of features using Pearson correlation
for col in df.columns:
    for row in df.columns:
        _, p = pearsonr(df_scaled[col], df_scaled[row])
        p_values[col][row] = p

# Print p-values that are less than 0.05 (indicating statistical significance)
print(p_values.applymap(lambda x: x if float(x) < 0.05 else None))
