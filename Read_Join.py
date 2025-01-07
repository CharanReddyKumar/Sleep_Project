# %%
import os
import dask.dataframe as dd

# Specify the root folder
root_folder = "dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-1.0.1"
data_folder = os.path.join(root_folder, "data")

# Set to track unique SIDs
unique_sids = set()

# Create an empty list for file paths and unique SIDs
file_paths = []
sids = []

# Walk through the directory structure
for folder_name, subfolders, filenames in os.walk(root_folder):
    if folder_name.endswith("data"):  # Filter for the "data" folder
        print(f"Processing files in folder: {folder_name}")
        for filename in filenames:
            if filename.endswith(".csv"):  # Ensure only CSV files
                file_path = os.path.join(folder_name, filename)
                print(f"Found file: {file_path}")

                # Extract SID from the filename
                base_sid = filename.split("_")[0]  # Extract the SID (e.g., S002 from S002_whole_df.csv)
                sid = base_sid

                # Ensure uniqueness by appending "Dup" if necessary
                counter = 1
                while sid in unique_sids:
                    sid = f"{base_sid}Dup{counter}"  # Append "Dup" and a counter for duplicates
                    counter += 1
                unique_sids.add(sid)

                # Save file path and SID
                file_paths.append(file_path)
                sids.append(sid)

# %%
# Process each file with Dask
data_frames = []
for file_path, sid in zip(file_paths, sids):
    # Load the file into a Dask DataFrame
    df = dd.read_csv(file_path)
    
    # Add the SID column
    df = df.assign(SID=sid)
    
    # Append the Dask DataFrame to the list
    data_frames.append(df)

# Combine all Dask DataFrames into one
combined_data = dd.concat(data_frames)

# Save the combined dataset (optional)
output_file = os.path.join(root_folder, "combined_data_with_sids.csv")
combined_data.to_csv(output_file, single_file=True, index=False)
print(f"\nCombined data saved to: {output_file}")
# %%

