import os
import re
import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.input_data import NiftiMasker

# 1. Set up paths
data_dir = "/Volumes/EGZONA/Neuro_Climate_Data"  # Path to your dataset root folder
labels_file = "/Volumes/EGZONA/Neuro_Climate_Data/participants.csv"  # CSV file with Subject ID and Label
output_dir = "processed_data"  # Directory to save processed data
os.makedirs(output_dir, exist_ok=True)

# 2. Load labels and prepare data
print("Processing labels and filtering subjects...")
labels_df = pd.read_csv(labels_file)
labels_df['Label'] = labels_df['MRI_finding'].apply(lambda x: 0 if x == 'normal' else 1)  # Create binary labels

# Filter out folders with "fu"
labels_df = labels_df[~labels_df['participant_id'].str.contains("fu")]
subject_ids = labels_df['participant_id']
labels = labels_df.set_index('participant_id')['Label'].to_dict()  # Create a dictionary of labels

# 3. Load MRI images and extract features
print("Loading MRI images...")
images = []
filtered_labels = {}
filtered_subject_ids = []
masker = NiftiMasker(smoothing_fwhm=6, memory="nilearn_cache", memory_level=1)

for subject_id in subject_ids:
    subject_dir = os.path.join(data_dir, subject_id)  # Subject folder path
    anat_dir = os.path.join(subject_dir, "anat")  # Anat folder path
    if not os.path.exists(anat_dir):  # Check if anat folder exists
        print(f"Anat folder missing for subject {subject_id}, skipping...")
        continue

    # Find the appropriate file within the anat folder
    anat_files = [f for f in os.listdir(anat_dir) if re.match(rf"{subject_id}_.*\.nii\.gz", f)]
    if not anat_files:
        print(f"No appropriate MRI file found for subject {subject_id}, skipping...")
        continue

    # Assuming you want the T1w or T2w file (customize as needed)
    file_to_load = None
    for file in anat_files:
        if "T1w" in file:  # Prioritize T1-weighted images
            file_to_load = file
            break
        elif "T2w" in file:  # Use T2-weighted images if no T1 found
            file_to_load = file

    if file_to_load:
        img_path = os.path.join(anat_dir, file_to_load)
        img = load_img(img_path)  # Load the MRI image
        images.append(img)
        filtered_labels[subject_id] = labels[subject_id]  # Add to filtered labels
        filtered_subject_ids.append(subject_id)  # Track processed subjects
    else:
        print(f"No suitable MRI file found for subject {subject_id}, skipping...")

# Extract features
print("Extracting features from MRI images...")
X = masker.fit_transform(images)  # Extract features
y = np.array([filtered_labels[subject_id] for subject_id in filtered_subject_ids])

# Save processed data
print("Saving processed data...")
np.savez_compressed(os.path.join(output_dir, "mri_features.npz"), X=X, y=y, subject_ids=filtered_subject_ids)
print(f"Processed data saved in {output_dir}.")




# Optional enhancements
#from sklearn.feature_selection import SelectKBest, f_classif
#selector = SelectKBest(f_classif, k=1000)  # Keep top 1000 features
#X_new = selector.fit_transform(X, y)

## Display important regions (visualization)
#from nilearn.plotting import plot_stat_map
#important_voxel_map = masker.inverse_transform(feature_importances)
#plot_stat_map(important_voxel_map, title="Feature Importance Map")
