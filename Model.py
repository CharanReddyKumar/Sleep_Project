#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
model_df = pd.read_csv('Processed_Data_processed.csv')
# %%
model_df.columns
# %%
model_df.describe()
# %%
# Compute the correlation matrix
correlation_matrix = model_df.corr(numeric_only=True)

# Display the correlation matrix
print(correlation_matrix)
# %%
# Plot the heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
# %%
# DROP IBI  as HR and IBI are almost same features
model_df.drop(columns=["IBI"], inplace=True)
print("Dropped 'IBI' column. Shape:", model_df.shape)
#%%
model_df.drop(columns=["TIMESTAMP"], inplace=True)
print("Dropped 'TIMESTAMP' column. Shape:", model_df.shape)
# %%
model_df.info()
# %%
# We'll only consider numeric columns for computing correlations.
numeric_cols = model_df.select_dtypes(include=['float64', 'int64', 'bool']).columns
X_numeric = model_df[numeric_cols]

# Compute the absolute correlation matrix
corr_matrix = X_numeric.corr().abs()
print(corr_matrix)
#%%
# Create a mask for the upper triangle
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
threshold = 0.90

# Find columns that have a correlation greater than the threshold with any other column
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("Features to drop due to high correlation (>", threshold, "):", to_drop)

# Drop these columns from the DataFrame
df_reduced = model_df.drop(columns=to_drop)
print("Shape after dropping highly correlated features:", df_reduced.shape)
# %%
df_reduced.describe()
# %%
model_df = df_reduced
# %%
df_reduced.columns
# %%
#%% 
# Count missing values for each column
nan_counts = model_df.isnull().sum()
print("Missing values per column:")
print(nan_counts)

# Count the total number of missing values in the DataFrame
total_nan = model_df.isnull().sum().sum()
print("\nTotal missing values in DataFrame:", total_nan)
#%%
# Optionally, visualize missing values with a heatmap (if the dataset size allows)
plt.figure(figsize=(12, 8))
sns.heatmap(model_df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

##############################

#%%

#%% [Model Building using Complete Data]


# %%
import pandas as pd

# Load the processed CSV file
df = pd.read_csv('Processed_Data_processed.csv')

# Count the unique SID values
unique_sid_count = df['SID'].nunique()
print("Unique SIDs:", unique_sid_count)

# %%
