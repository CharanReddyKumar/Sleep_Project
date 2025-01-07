# %%
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the combined dataset with Dask
file_path = "dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-1.0.1/combined_data_with_sids.csv"
combined_data = dd.read_csv(file_path)

# %%
print("\nDataset Overview:")
print(combined_data.dtypes)
#%%
# Compute the number of rows and columns
num_rows = combined_data.shape[0].compute()  # Compute the delayed row count
num_columns = combined_data.shape[1]  
print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_columns}")

#%%
print("\nFirst Few Rows:")
print(combined_data.head())


# %%
# Filter the Dask DataFrame for rows with SID == "S203"
filtered_data = combined_data[combined_data["SID"] == "S103"]

# Check if the filtered_data is empty
if filtered_data.shape[0].compute() == 0:
    print("No data found for SID S203.")
else:
    # Compute and convert the filtered data to pandas for inspection
    filtered_data_pd = filtered_data.compute()
    print("Filtered Data for SID S203:")
    print(filtered_data_pd)


# %%
#Check for missing values
missing_values = combined_data.isnull().sum().compute()
print("\nMissing Values:")
print(missing_values)




# %%
filtered_data = combined_data[(combined_data["HR"] == 0) | (combined_data["HR"].isnull())]
filtered_data.head()

# %%
# Percentage of missing IBI values
missing_ibi = combined_data["IBI"].isnull().sum().compute()
total_rows = combined_data.shape[0].compute()
missing_percentage = (missing_ibi / total_rows) * 100

print(f"Total Missing IBI Values: {missing_ibi}")
print(f"Percentage of Missing IBI Values: {missing_percentage:.2f}%")

# %%
# Analyze IBI values
# Count consecutive missing values in IBI
combined_data["IBI_Missing"] = combined_data["IBI"].isnull()

# Group by consecutive missing streaks
streaks = combined_data.groupby(combined_data["IBI_Missing"].cumsum())

# Get streak lengths for missing values
missing_streak_lengths = streaks.size().compute()
longest_missing_streak = missing_streak_lengths.max()

print(f"Total Missing Streaks: {len(missing_streak_lengths)}")
print(f"Longest Missing Streak in IBI: {longest_missing_streak} timestamps")



# %%
# %%
