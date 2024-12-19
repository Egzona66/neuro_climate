import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import csv
import os

# Paths to metric files
fa_file = "/Users/egzonamorina/opt/anaconda3/envs/cirq-env/ds005713-download/sub-01/dwi/processed_run-01/run-01_fa.nii.gz"  # Fractional Anisotropy
md_file = "/Users/egzonamorina/opt/anaconda3/envs/cirq-env/ds005713-download/sub-01/dwi/processed_run-01/run-01_md.nii.gz"  # Mean Diffusivity
ad_file = "/Users/egzonamorina/opt/anaconda3/envs/cirq-env/ds005713-download/sub-01/dwi/processed_run-01/run-01_ad.nii.gz"  # Axial Diffusivity
rd_file = "/Users/egzonamorina/opt/anaconda3/envs/cirq-env/ds005713-download/sub-01/dwi/processed_run-01/run-01_rd.nii.gz"  # Radial Diffusivity

# Optional: Path to ROI mask file
roi_mask_file = None  # Provide your ROI mask file here or set to None if not available

# Load NIfTI files
def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

# Compute whole-brain statistics
def compute_statistics(data):
    brain_data = data[data > 0]  # Exclude non-brain voxels (value = 0)
    mean_val = np.mean(brain_data)
    std_val = np.std(brain_data)
    return mean_val, std_val

# Perform ROI-based analysis
def roi_analysis(data, roi_mask_file):
    roi_mask, _ = load_nifti(roi_mask_file)
    roi_values = data[roi_mask > 0]  # Extract values where mask > 0
    mean_roi = np.mean(roi_values)
    std_roi = np.std(roi_values)
    return mean_roi, std_roi

# Save statistics to CSV
def save_statistics_to_csv(statistics, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    print(f"Statistics saved to: {output_csv}")

# Main Analysis
def analyze_and_visualize_metrics():
    # Load metrics
    fa_data = load_nifti(fa_file)
    md_data = load_nifti(md_file)
    ad_data = load_nifti(ad_file)
    rd_data = load_nifti(rd_file)

    # Combine metrics into a dictionary
    metrics = {
        "FA": fa_data,
        "MD": md_data,
        "AD": ad_data,
        "RD": rd_data
    }

    # Initialize statistics
    statistics = []

    # Whole-brain statistics
    print("Whole-Brain Statistics:")
    for metric_name, metric_data in metrics.items():
        mean_val, std_val = compute_statistics(metric_data)
        print(f"{metric_name} - Mean: {mean_val:.4f}, Std Dev: {std_val:.4f}")
        statistics.append({
            "Metric": metric_name,
            "Region": "Whole Brain",
            "Mean": mean_val,
            "StdDev": std_val
        })

    # ROI-based analysis (optional)
    if roi_mask_file and os.path.exists(roi_mask_file):
        print("\nPerforming ROI analysis...")
        for metric_name, metric_data in metrics.items():
            mean_roi, std_roi = roi_analysis(metric_data, roi_mask_file)
            print(f"{metric_name} ROI - Mean: {mean_roi:.4f}, Std Dev: {std_roi:.4f}")
            statistics.append({
                "Metric": metric_name,
                "Region": "ROI",
                "Mean": mean_roi,
                "StdDev": std_roi
            })

    # Save statistics to CSV
    output_csv = "/Users/egzonamorina/opt/anaconda3/envs/cirq-env/ds005713-download/sub-01/dwi/processed_run-01/subject-01_dti_statistics.csv"
    save_statistics_to_csv(statistics, output_csv)

    # Visualization with slider
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)  # Make room for the slider

    # Default metric and slice
    metric_selector = "FA"
    metric_data = metrics[metric_selector]
    slice_idx = metric_data.shape[2] // 2  # Default to middle slice
    im = ax.imshow(metric_data[:, :, slice_idx].T, cmap="gray", origin="lower")
    ax.set_title(f"{metric_selector} - Slice {slice_idx}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Slider for slice selection
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, metric_data.shape[2] - 1, valinit=slice_idx, valstep=1)

    # Update plot with slider
    def update(val):
        slice_idx = int(slider.val)
        im.set_data(metric_data[:, :, slice_idx].T)
        ax.set_title(f"{metric_selector} - Slice {slice_idx}")
        plt.draw()

    slider.on_changed(update)

    # Metric selection (manual for now)
    print("Available metrics: FA, MD, AD, RD")
    metric_selector = input("Enter metric to display (FA/MD/AD/RD): ").strip().upper()
    if metric_selector in metrics:
        metric_data = metrics[metric_selector]
        slider.valinit = metric_data.shape[2] // 2  # Reset slider for new metric
        im.set_data(metric_data[:, :, int(slider.val)].T)
        ax.set_title(f"{metric_selector} - Slice {int(slider.val)}")
        plt.draw()

    plt.show()

# Run the analysis
analyze_and_visualize_metrics()
