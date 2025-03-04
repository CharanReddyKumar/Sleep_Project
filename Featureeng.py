#%% Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from scipy.fft import fft

#%% Read the data
Processed_data = pd.read_csv("processed_data.csv")

#%% Basic data overview
print("Total Rows:", Processed_data.shape[0])
print("Missing Values:\n", Processed_data.isnull().sum())
print("Sleep Stage Distribution:\n", Processed_data["Sleep_Stage"].value_counts())

#%% Label encoding for the Sleep_Stage column
le = LabelEncoder()
le.fit(["W", "N1", "N2", "N3", "R"])
stage_mapping = {stage: idx for idx, stage in enumerate(le.classes_)}
Processed_data["Sleep_Stage_Encoded"] = Processed_data["Sleep_Stage"].map(stage_mapping)
print("Encoded Sleep Stage head:\n", Processed_data["Sleep_Stage_Encoded"].head())

#%% View head and missing values per column
print("Data Head:\n", Processed_data.head())
missing_per_column = Processed_data.isnull().sum()
print("Missing values per column:")
print(missing_per_column)
print("Columns:", Processed_data.columns)

#%% Drop unwanted column
Processed_data = Processed_data.drop("IBI_Missing", axis=1)
print("After dropping IBI_Missing:\n", Processed_data.head())

#%% Compute class weights for loss function
y_labels = Processed_data["Sleep_Stage_Encoded"].values
unique_classes = np.unique(y_labels)
print("Present classes:", unique_classes)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2, 3, 4]),
    y=y_labels
)
weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", weight_dict)

#%% Compute movement feature: ACC_magnitude
Processed_data['ACC_magnitude'] = np.sqrt(
    Processed_data['ACC_X']**2 +
    Processed_data['ACC_Y']**2 +
    Processed_data['ACC_Z']**2
)
print("After adding ACC_magnitude:\n", Processed_data.head())

#%% Assign global IDs and set index
Processed_data['global_id'] = range(len(Processed_data))
Processed_data.set_index('global_id', inplace=True)
print("After assigning global IDs:\n", Processed_data.head())

#%% Spectral Features (BVP/EDA) using FFT
# Create new columns for spectral data and low frequency band
Processed_data['BVP_spectral'] = np.nan
Processed_data['BVP_lf'] = np.nan
# Convert BVP_spectral column to object type so it can hold lists
Processed_data['BVP_spectral'] = Processed_data['BVP_spectral'].astype(object)

window_size = 320  # 5-second window (assumes your sampling rate yields 320 samples per 5 seconds)
step_size = 160    # 50% overlap
bvp_series = Processed_data['BVP']

# Loop over windows and compute FFT
for start in range(0, len(bvp_series) - window_size + 1, step_size):
    window = bvp_series.iloc[start:start+window_size]
    spec = np.abs(fft(window.values))[:160]
    center_index = start + window_size // 2
    Processed_data.at[Processed_data.index[center_index], 'BVP_spectral'] = spec.tolist()
    Processed_data.at[Processed_data.index[center_index], 'BVP_lf'] = np.mean(spec[3:11])

print("After computing spectral features (BVP):")
print(Processed_data[['BVP', 'BVP_spectral', 'BVP_lf']].dropna().head())
