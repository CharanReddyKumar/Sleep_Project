# EDA.py
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
# Filter the Dask DataFrame for rows with SID == "S103"
filtered_data = combined_data[combined_data["SID"] == "S103"]

# Check if the filtered_data is empty
if filtered_data.shape[0].compute() == 0:
    print("No data found for SID S103.")
else:
    # Compute and convert the filtered data to pandas for inspection
    filtered_data_pd = filtered_data.compute()
    print("Filtered Data for SID S103:")
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
# Analyze IBI values
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

print(f"Total Missing: {len(missing_streak_lengths)}")
print(f"Longest Missing Streak in IBI: {longest_missing_streak} timestamps")

# %%
combined_data.head()

# %%
# Extract SIDs and Sleep Stages for missing IBI
missing_ibi_data = combined_data[combined_data["IBI"].isnull()]
missing_ibi_details = missing_ibi_data[["SID", "Sleep_Stage"]].compute()

# Analyze missing IBI by SID and Sleep Stage
missing_ibi_grouped = missing_ibi_details.groupby(["SID", "Sleep_Stage"]).size()
print("\nMissing IBI Counts by SID and Sleep Stage:")
print(missing_ibi_grouped)
# Looks like if Stage is "P" then IBI is NaN
# %%
#Count sleep stage in missing IBI data
sleep_stage_counts = missing_ibi_data.groupby("Sleep_Stage").size().compute()

# Display the counts
print("\nCounts of Sleep Stages in Missing IBI Data:")
print(sleep_stage_counts)
# %%
# Count number of "P" sleep stages in the dataset
# Count occurrences of 'P' in Sleep_Stage
p_count = combined_data[combined_data["Sleep_Stage"] == "P"].shape[0].compute()

print(f"Number of 'P' sleep stages in the combined data: {p_count}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Filter valid IBI rows in the 'P' sleep stage
valid_ibi_p_stage = combined_data[(combined_data["Sleep_Stage"] == "P") & (~combined_data["IBI"].isnull())].compute()

# Plot distribution of valid IBI in the 'P' stage
sns.histplot(valid_ibi_p_stage["IBI"], kde=True, bins=50)
plt.title("Distribution of Valid IBI in 'P' Sleep Stage")
plt.xlabel("IBI (seconds)")
plt.ylabel("Frequency")
plt.show()

# %%
# Filter rows where Sleep Stage is 'P' and IBI is missing
missing_ibi_p_stage = combined_data[(combined_data["Sleep_Stage"] == "P") & (combined_data["IBI"].isnull())].compute()

# Plot missing IBI over time (using TIMESTAMP)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=missing_ibi_p_stage["TIMESTAMP"], y=missing_ibi_p_stage["IBI_Missing"], alpha=0.5)
plt.title("Missing IBI in 'P' Sleep Stage Over Time")
plt.xlabel("Timestamp")
plt.ylabel("IBI Missing (1 = Missing)")
plt.show()
# %%
# Select 5 specific SIDs (replace these with actual SIDs from your dataset)
SelS001 = combined_data[combined_data["SID"] == "S001"].compute()


# Plot IBI vs. Time for each SID
plt.figure(figsize=(12, 8))
sns.lineplot(data=SelS001, x="TIMESTAMP", y="IBI", hue="SID", marker="o", alpha=0.7)
plt.title("IBI vs. Time for Selected SIDs")
plt.xlabel("Timestamp")
plt.ylabel("IBI (seconds)")
plt.legend(title="SID", loc="best")
plt.show()
# we can 
# %%
# Analyze NaN patterns for IBI for each SID
sids = combined_data["SID"].unique().compute()

nan_pattern_results = []

for sid in sids:
    # Filter data for the current SID
    sid_data = combined_data[combined_data["SID"] == sid].compute()
    
    # Add a flag for missing IBI
    sid_data["IBI_Missing"] = sid_data["IBI"].isnull()
    
    # Find the first non-missing IBI timestamp
    first_non_missing_row = sid_data[~sid_data["IBI_Missing"]]
    if not first_non_missing_row.empty:
        first_non_missing_time = first_non_missing_row["TIMESTAMP"].iloc[0]
    else:
        first_non_missing_time = None

    # Check if NaNs are only at the start
    if first_non_missing_time is None:
        only_start_missing = True  
    else:
        only_start_missing = (sid_data[sid_data["IBI_Missing"]]["TIMESTAMP"] < first_non_missing_time).all()

    # Calculate the longest NaN streak
    streaks = sid_data["IBI_Missing"].cumsum()
    streak_lengths = sid_data.groupby(streaks)["IBI_Missing"].sum()
    
    nan_pattern_results.append({
        "SID": sid,
        "Total_NaN": sid_data["IBI_Missing"].sum(),
        "Longest_NaN_Streak": streak_lengths.max(),
        "NaNs_Only_At_Start": only_start_missing
    })

# Convert results to a DataFrame
import pandas as pd
nan_pattern_df = pd.DataFrame(nan_pattern_results)

# Display the results
print(nan_pattern_df)

# %%
print(combined_data[combined_data["Sleep_Stage"] == "P"]["SID"].value_counts().compute())

#%%
# Check if "Missing" stages align with sensor dropouts
missing_signal = combined_data[combined_data["Sleep_Stage"] == "Missing"]
print(missing_signal[["HR", "BVP", "EDA", "TEMP"]].isnull().mean().compute())

#%%
# Check if P stage has any  other columns missing
# Filter rows where Sleep_Stage is "P"
p_stage_data = combined_data[combined_data["Sleep_Stage"] == "P"]

# Check if "P" exists
if p_stage_data.shape[0].compute() == 0:
    print("No 'P' stage data found.")
else:
    # Compute missing values for key columns
    missing_values = p_stage_data[["HR", "BVP", "EDA", "TEMP", "IBI", "ACC_X", "ACC_Y", "ACC_Z"]].isnull().sum().compute()
    missing_pct = (missing_values / p_stage_data.shape[0].compute()) * 100

    # Create summary table
    missing_summary = pd.DataFrame({
        "Missing_Count": missing_values,
        "Missing_Percent (%)": missing_pct.round(2)
    })
    print("Missing Values in 'P' Stage:")
    print(missing_summary)
#%%
sleep_stage_counts = combined_data["Sleep_Stage"].value_counts().compute()
print(sleep_stage_counts)
plt.figure(figsize=(10,6))
sns.barplot(x=sleep_stage_counts.index, y=sleep_stage_counts.values)
plt.title("Sleep Stage Distribution")
plt.show()
# %%[markdown]
# There is 120960 missing stages nad there is significatly large class imablnce in the data
#%%[markdown]
# Based on the EDA of P and infromation from the https://physionet.org/content/dreamt/1.0.1/ we can assume P is purely prepartional stage and we can drop the rows with P stage(can not assume as wake stage)
#%%
# Drop rows with "P" sleep stage
combined_data = combined_data[combined_data["Sleep_Stage"] != "P"]
#%%
# Drop unrelavent columns
columns_to_drop = ["Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events"]
combined_data = combined_data.drop(columns=columns_to_drop)
print(filtered_data.columns)

#%%
print(combined_data.isnull().sum().compute())
#%%[markdown]
# There is no missing IBI values after droping P stage
#%%
combined_data = combined_data[combined_data["Sleep_Stage"] != "Missing"]
# %%
sleep_stage_counts = combined_data["Sleep_Stage"].value_counts().compute()
print(sleep_stage_counts)
plt.figure(figsize=(10,6))
sns.barplot(x=sleep_stage_counts.index, y=sleep_stage_counts.values)
plt.title("Sleep Stage Distribution")
plt.show

# %%
processed_data = combined_data
output_file = ("Processed_Data.csv")
processed_data.to_csv(output_file, single_file=True, index=False)
print(f"Saved")
#%%

