#%%
import dask.dataframe as dd
import pandas as pd
# %%
Processed_data = dd.read_csv("processed_data.csv")
#%%
print("Total Rows:", Processed_data.shape[0].compute())
#%%
print("Missing Values:\n", Processed_data.isnull().sum().compute())
#%%
print("Sleep Stage Distribution:\n", Processed_data["Sleep_Stage"].value_counts().compute())
# %%
# Label encoding for the Sleep_Stage column
from sklearn.preprocessing import LabelEncoder

# Initialize and fit LabelEncoder
le = LabelEncoder()
le.fit(["W", "N1", "N2", "N3", "R"])

# Create mapping dictionary
stage_mapping = {stage: idx for idx, stage in enumerate(le.classes_)}

# Apply encoding
Processed_data["Sleep_Stage_Encoded"] = Processed_data["Sleep_Stage"].map(
    stage_mapping,
    meta=("Sleep_Stage_Encoded", "int64")
)

print(Processed_data["Sleep_Stage_Encoded"].head())
# %%
Processed_data.head()
# %% 
# Count the number of missing (NaN) values in each column
missing_per_column = Processed_data.isnull().sum().compute()
print("Missing values per column:")
print(missing_per_column)

# %%
