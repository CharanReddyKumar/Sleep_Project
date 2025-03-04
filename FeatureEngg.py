# featureEngg.py
#%% [Imports]
import os
import pandas as pd
import numpy as np
from scipy.fft import fft
from concurrent.futures import ProcessPoolExecutor, as_completed

#%% [Feature Engineering Function]
def process_chunk(chunk):
    # Ensure the data is sorted by timestamp if needed
    chunk = chunk.sort_values("TIMESTAMP").reset_index(drop=True)
    
    # --- Compute Accelerometer Magnitude ---
    chunk['ACC_magnitude'] = np.sqrt(
        chunk['ACC_X']**2 +
        chunk['ACC_Y']**2 +
        chunk['ACC_Z']**2
    )
    
    # Rolling statistics for ACC magnitude (using a 1-second window at 32 Hz)
    chunk['ACC_mag_rolling_mean'] = chunk['ACC_magnitude'].rolling(window=32, min_periods=1).mean()
    chunk['ACC_mag_rolling_std']  = chunk['ACC_magnitude'].rolling(window=32, min_periods=1).std()
    
    # --- Compute Rolling Statistics for HR, EDA, TEMP ---
    # HR is sampled at 1 Hz: use a 60-second (1-minute) window
    chunk['HR_rolling_mean'] = chunk['HR'].rolling(window=60, min_periods=1).mean()
    chunk['HR_rolling_std']  = chunk['HR'].rolling(window=60, min_periods=1).std()
    
    # EDA and TEMP are sampled at 4 Hz: use a 10-second window (i.e. 40 samples)
    chunk['EDA_rolling_mean'] = chunk['EDA'].rolling(window=40, min_periods=1).mean()
    chunk['EDA_rolling_std']  = chunk['EDA'].rolling(window=40, min_periods=1).std()
    
    chunk['TEMP_rolling_mean'] = chunk['TEMP'].rolling(window=40, min_periods=1).mean()
    chunk['TEMP_rolling_std']  = chunk['TEMP'].rolling(window=40, min_periods=1).std()
    
    # --- Compute Spectral and Statistical Features for BVP ---
    # Initialize new columns for spectral features and statistical measures.
# Ensure BVP_spectral column exists before setting type
    if 'BVP_spectral' not in chunk.columns:
        chunk['BVP_spectral'] = np.nan  # Initialize it if necessary
 
    chunk['BVP_spectral'] = chunk['BVP_spectral'].astype(object)

    # Initialize other new columns
    chunk['BVP_lf']   = np.nan
    chunk['BVP_mean'] = np.nan
    chunk['BVP_std']  = np.nan
    chunk['BVP_min']  = np.nan
    chunk['BVP_max']  = np.nan

    
    # Define sliding window parameters (e.g., 320 samples with 50% overlap, step=160)
    window_size = 320
    step = 160
    
    # Loop over the chunk using the sliding window approach
    for start in range(0, len(chunk) - window_size + 1, step):
        window = chunk['BVP'].iloc[start:start+window_size]
        
        # Compute FFT and take absolute value for the first half (up to the Nyquist frequency)
        spec = np.abs(fft(window.values))[:window_size // 2]
        
        # Determine the center index of the window to assign features
        center_index = start + window_size // 2
        idx = chunk.index[center_index]
        
        # Save the spectral features:
        # - BVP_spectral: full spectrum (as a list)
        # - BVP_lf: mean of spectral values in a chosen band (e.g., bins 3 to 10)
        chunk.at[idx, 'BVP_spectral'] = [spec.tolist()]
        chunk.at[idx, 'BVP_lf'] = spec[3:11].mean()
        
        # Also compute additional statistical features for the same window
        chunk.at[idx, 'BVP_mean'] = window.mean()
        chunk.at[idx, 'BVP_std'] = window.std()
        chunk.at[idx, 'BVP_min'] = window.min()
        chunk.at[idx, 'BVP_max'] = window.max()
    
    return chunk

#%% [Main Processing Function]

#%%
def main():
    # Input and output file paths
    input_file = "Processed_Data.csv"  # Adjust path if necessary
    output_file = "Processed_Data_processed.csv"
    
    # Set an appropriate chunk size (adjust based on available memory)
    chunksize = 30000
    processed_chunks = []  # List to store processed DataFrames

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        # Read CSV in chunks and submit each for processing
        for chunk in pd.read_csv(input_file, chunksize=chunksize):
            futures.append(executor.submit(process_chunk, chunk))

        # Process completed chunks
        for future in as_completed(futures):
            try:
                processed_chunk = future.result()
                processed_chunks.append(processed_chunk)
                print(f"Processed a chunk with {len(processed_chunk)} rows.")
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    # Concatenate all processed chunks into one DataFrame
    if processed_chunks:
        processed_data = pd.concat(processed_chunks, ignore_index=True)
        
        # Save the final DataFrame to a single CSV file
        processed_data.to_csv(output_file, index=False)
        print(f"\nFinal processed data saved to {output_file}")
    else:
        print("No data processed.")
#%%
if __name__ == "__main__":
    main()

# %%
