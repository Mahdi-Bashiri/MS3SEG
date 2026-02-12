import os
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Configuration
MAIN_DIR = r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_masks"
DEMOGRAPHICS_FILE = r"E:\MBashiri\Thesis\p6\patient_demographics_raw.csv"  # Update this path
PIXEL_SPACING = 0.9  # mm per pixel
PIXEL_AREA = PIXEL_SPACING ** 2  # mm² per pixel

# Mask folder names
MASK_FOLDERS = {
    'abWMH': 'abWMH_Masks',
    'nWMH': 'nWMH_Masks',
    'VENT': 'Vent_Masks'
}


def load_patient_demographics():
    """
    Load patient demographics from CSV file.

    Returns:
        dict: Dictionary mapping patient_id to demographic info
    """
    demographics = {}

    try:
        # Read the CSV file
        df = pd.read_csv(DEMOGRAPHICS_FILE)

        # Convert to dictionary
        for _, row in df.iterrows():
            patient_id = str(row['Patient_ID'])  # Ensure it's a string
            demographics[patient_id] = {
                'age': row['Age'],
                'gender': row['Sex'].upper()  # Normalize to uppercase
            }

        print(f"Loaded demographics for {len(demographics)} patients")
        return demographics

    except FileNotFoundError:
        print(f"Warning: Demographics file not found: {DEMOGRAPHICS_FILE}")
        print("All patients will be marked as 'Unknown' gender")
        return {}
    except Exception as e:
        print(f"Error loading demographics: {e}")
        return {}


def analyze_mask_3d(mask_path):
    """
    Analyze a 3D mask volume and return lesion count and total load.

    Args:
        mask_path: Path to the .nii.gz mask file

    Returns:
        tuple: (lesion_count, total_load_mm2)
    """
    try:
        # Load NIfTI file
        nii = nib.load(str(mask_path))
        mask_data = nii.get_fdata()

        # Ensure binary mask
        mask_data = (mask_data > 0).astype(np.uint8)

        total_lesion_count = 0
        total_area_pixels = 0

        # Process each slice
        for slice_idx in range(mask_data.shape[2]):
            slice_mask = mask_data[:, :, slice_idx]

            # Skip empty slices
            if not slice_mask.any():
                continue

            # Convert to uint8 for OpenCV
            slice_mask_uint8 = (slice_mask * 255).astype(np.uint8)

            # Find contours using OpenCV
            contours, _ = cv2.findContours(
                slice_mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Count lesions and calculate area for this slice
            slice_lesion_count = len(contours)
            slice_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                slice_area += area

            total_lesion_count += slice_lesion_count
            total_area_pixels += slice_area

        # Convert area from pixels to mm²
        total_area_mm2 = total_area_pixels * PIXEL_AREA

        return total_lesion_count, total_area_mm2

    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return 0, 0.0


def collect_patient_data(demographics):
    """
    Collect data for all patients across all mask types.

    Args:
        demographics: Dictionary with patient demographic information

    Returns:
        dict: Patient data with mask statistics
    """
    patient_data = defaultdict(lambda: {
        'gender': 'Unknown',
        'age': None,
        'abWMH_count': 0,
        'abWMH_load': 0.0,
        'nWMH_count': 0,
        'nWMH_load': 0.0,
        'VENT_count': 0,
        'VENT_load': 0.0
    })

    # Process each mask type
    for mask_type, folder_name in MASK_FOLDERS.items():
        mask_dir = Path(MAIN_DIR) / folder_name

        if not mask_dir.exists():
            print(f"Warning: Directory not found: {mask_dir}")
            continue

        # Get all patient folders
        patient_folders = [d for d in mask_dir.iterdir() if d.is_dir()]

        print(f"\nProcessing {mask_type} masks...")
        for patient_folder in patient_folders:
            patient_id = patient_folder.name

            # Get demographics for this patient
            if patient_id in demographics:
                patient_data[patient_id]['gender'] = demographics[patient_id]['gender']
                patient_data[patient_id]['age'] = demographics[patient_id]['age']
            else:
                print(f"  Warning: No demographics found for patient {patient_id}")

            # Find the mask file
            nii_files = list(patient_folder.glob('*.nii.gz'))

            if not nii_files:
                print(f"  Warning: No .nii.gz file found for patient {patient_id}")
                continue

            mask_file = nii_files[0]

            # Analyze the mask
            count, load = analyze_mask_3d(mask_file)

            # Store results
            patient_data[patient_id][f'{mask_type}_count'] = count
            patient_data[patient_id][f'{mask_type}_load'] = load

            print(f"  Patient {patient_id}: {count} lesions, {load:.2f} mm²")

    return patient_data


def create_summary_statistics(patient_data):
    """
    Create summary statistics for Table 4.

    Args:
        patient_data: Dictionary with patient-level data

    Returns:
        pandas.DataFrame: Summary table
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame.from_dict(patient_data, orient='index')

    results = []

    # Define cohorts
    cohorts = {
        'all': df,
        'female': df[df['gender'].str.upper().isin(['F', 'FEMALE'])],
        'male': df[df['gender'].str.upper().isin(['M', 'MALE'])]
    }

    # Calculate statistics for each cohort
    for cohort_name, cohort_df in cohorts.items():
        if len(cohort_df) == 0:
            print(f"Warning: No patients in {cohort_name} cohort")
            continue

        row_data = {'cohort': cohort_name}

        for mask_type in ['abWMH', 'nWMH', 'VENT']:
            count_col = f'{mask_type}_count'
            load_col = f'{mask_type}_load'

            # Count statistics (mean ± SD)
            mean_count = cohort_df[count_col].mean()
            std_count = cohort_df[count_col].std()
            row_data[f'{mask_type} count (mean ± SD)'] = f"{mean_count:.2f} ± {std_count:.2f}"

            # Load statistics
            mean_load = cohort_df[load_col].mean()
            std_load = cohort_df[load_col].std()
            median_load = cohort_df[load_col].median()

            row_data[f'{mask_type} load (mean ± SD)'] = f"{mean_load:.2f} ± {std_load:.2f}"
            row_data[f'{mask_type} load (median)'] = f"{median_load:.2f}"

        results.append(row_data)

    # Create final DataFrame
    summary_df = pd.DataFrame(results)

    # Reorder columns to match the requested format
    column_order = [
        'cohort',
        'abWMH count (mean ± SD)', 'abWMH load (mean ± SD)', 'abWMH load (median)',
        'nWMH count (mean ± SD)', 'nWMH load (mean ± SD)', 'nWMH load (median)',
        'VENT count (mean ± SD)', 'VENT load (mean ± SD)', 'VENT load (median)'
    ]

    summary_df = summary_df[column_order]

    return summary_df


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("MS Patient Mask Analysis - Table 4 Generation")
    print("=" * 80)

    # Check if main directory exists
    if not os.path.exists(MAIN_DIR):
        print(f"Error: Main directory not found: {MAIN_DIR}")
        return

    # Load patient demographics
    print("\nStep 1: Loading patient demographics...")
    demographics = load_patient_demographics()

    # Collect patient data
    print("\nStep 2: Collecting patient data...")
    patient_data = collect_patient_data(demographics)

    print(f"\nTotal patients processed: {len(patient_data)}")

    # Create summary statistics
    print("\nStep 3: Generating summary statistics...")
    summary_table = create_summary_statistics(patient_data)

    # Display results
    print("\n" + "=" * 80)
    print("TABLE 4: Annotation Statistics")
    print("=" * 80)
    print(summary_table.to_string(index=False))

    # Save to CSV
    output_file = "table4_annotation_statistics.csv"
    summary_table.to_csv(output_file, index=False)
    print(f"\nTable saved to: {output_file}")

    # Save detailed patient data
    detailed_output = "detailed_patient_data.csv"
    detailed_df = pd.DataFrame.from_dict(patient_data, orient='index')
    detailed_df.index.name = 'patient_id'
    detailed_df.to_csv(detailed_output)
    print(f"Detailed patient data saved to: {detailed_output}")

    # Print cohort summary
    df = pd.DataFrame.from_dict(patient_data, orient='index')
    print("\n" + "=" * 80)
    print("Cohort Summary:")
    print("=" * 80)
    print(f"Total patients: {len(df)}")
    print(f"Female patients: {len(df[df['gender'] == 'F'])}")
    print(f"Male patients: {len(df[df['gender'] == 'M'])}")
    print(f"Unknown gender: {len(df[df['gender'] == 'Unknown'])}")
    if 'age' in df.columns:
        print(f"Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
        print(f"Mean age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years")


if __name__ == "__main__":
    main()