# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:55:30 2024

@author: Armanis
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

# Load data
file_path = r"C:\Users\Armanis\OneDrive\Desktop\Python CSV Files\Cereals.csv"
cereal = pd.read_csv(file_path)

# Convert type column to categorical
cereal['type'] = pd.Categorical(cereal['type'])

# Numerical variables subset
quant_vars = [
    'calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo',
    'sugars', 'potass', 'vitamins', 'weight', 'cups', 'shelf', 'rating'
]
quantcereal = cereal[quant_vars]

# Summary statistics
print(quantcereal.describe())

# Create histograms for numerical variables
for var in quant_vars:
    sns.set(style='darkgrid')
    plt.figure(figsize=(7, 7))
    plt.hist(quantcereal[var].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {var}')
    plt.show()

# Alternative: Matrix of histograms
quantcereal.hist(figsize=(12, 12), bins=30, color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplots
# Calories by type
plt.figure(figsize=(7, 5))
ax = cereal.boxplot(column='calories', by='type')
ax.set_ylabel('Calories')
plt.title('Distribution of Calories by Cereal Type')
plt.suptitle('')  # Suppress automatic titles
plt.show()

# Rating by shelf
plt.figure(figsize=(7, 5))
ax = cereal.boxplot(column='rating', by='shelf')
ax.set_ylabel('Rating')
plt.title('Distribution of Customer Ratings by Shelf Height')
plt.suptitle('')
plt.show()

# Correlation matrix
corr = quantcereal.corr()
print(corr)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', center=0, vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Numerical Variables')
plt.show()

#%%
# Principal Component Analysis (PCA)
# Without normalization
pcs = PCA()
pcs.fit(quantcereal.dropna())
pcs_summary = pd.DataFrame({
    'Standard deviation': np.sqrt(pcs.explained_variance_),
    'Proportion of variance': pcs.explained_variance_ratio_,
    'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_)
}).transpose()
pcs_summary.columns = [f'PC{i+1}' for i in range(len(pcs_summary.columns))]
print(pcs_summary.round(4))




#%%
# PCA Components
pcs_components = pd.DataFrame(
    pcs.components_.transpose(),
    columns=pcs_summary.columns,
    index=quantcereal.columns
)
print(pcs_components.iloc[:, :5])

# With normalization
scaled_data = preprocessing.scale(quantcereal.dropna())
pcs.fit(scaled_data)

#  Create PCA summary DataFrame and Transpose for better readability (optional step)
pcs_summary_norm = pd.DataFrame({
    'Standard deviation': np.sqrt(pcs.explained_variance_),
    'Proportion of variance': pcs.explained_variance_ratio_,
    'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_)
}).transpose()

# Rename columns for clarity
pcs_summary_norm.columns = [f'PC{i+1}' for i in range(len(pcs_summary_norm.columns))]

# Round to 4 decimal places
print(pcs_summary_norm.round(4))

# Print the full PCA summary with cumulative proportion
print(pcs_summary.loc['Cumulative proportion'])

#Determining the minimum of the cumulative Proportion is key to determining minimum number of variables
