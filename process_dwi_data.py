import os
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu


# Example DWI file path
dwi_file = "/Volumes/EGZONA/Neuro_Climate_Data/sub-01/dwi/sub-01_run-01_dwi.nii.gz"

# Extract subject folder and run ID
subject_folder = os.path.dirname(dwi_file)  # Folder containing the files
run_id = os.path.basename(dwi_file).split('_')[1]  # Extract 'run-01'

# Dynamically set paths for b-values and b-vectors
bval_file = os.path.join(subject_folder, f"sub-01_{run_id}_dwi.bval")
bvec_file = os.path.join(subject_folder, f"sub-01_{run_id}_dwi.bvec")

# Output results
print("DWI File:", dwi_file)
print("BVAL File:", bval_file)
print("BVEC File:", bvec_file)

# Create an output folder within the subject folder for this run
output_dir = os.path.join(subject_folder, f"processed_{run_id}")
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load diffusion data and gradients
print("Loading data...")
dwi_img = nib.load(dwi_file)
data = dwi_img.get_fdata()
affine = dwi_img.affine

# Load b-values and b-vectors
bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
gtab = gradient_table(bvals, bvecs)

# Step 2: Generate brain mask
print("Creating brain mask...")
maskdata, mask = median_otsu(data, vol_idx=np.where(bvals < 100)[0], numpass=2)

# Step 3: Fit the tensor model
print("Fitting tensor model...")
tensor_model = TensorModel(gtab)
tensor_fit = tensor_model.fit(maskdata, mask=mask)

# Step 4: Compute metrics
print("Computing DTI metrics...")
fa = tensor_fit.fa  # Fractional Anisotropy
md = tensor_fit.md  # Mean Diffusivity
ad = tensor_fit.ad  # Axial Diffusivity
rd = tensor_fit.rd  # Radial Diffusivity

# Step 5: Save outputs
fa_file = os.path.join(output_dir, f"{run_id}_fa.nii.gz")
md_file = os.path.join(output_dir, f"{run_id}_md.nii.gz")
ad_file = os.path.join(output_dir, f"{run_id}_ad.nii.gz")
rd_file = os.path.join(output_dir, f"{run_id}_rd.nii.gz")

nib.save(nib.Nifti1Image(fa, affine), fa_file)
nib.save(nib.Nifti1Image(md, affine), md_file)
nib.save(nib.Nifti1Image(ad, affine), ad_file)
nib.save(nib.Nifti1Image(rd, affine), rd_file)

print(f"DTI metrics saved in: {output_dir}")
