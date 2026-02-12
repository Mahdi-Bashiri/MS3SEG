import os
import pydicom
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def extract_patient_demographics(main_dir):
    """
    Extract demographic information from DICOM files for all MS patients.

    Args:
        main_dir: Main directory containing patient folders

    Returns:
        DataFrame with patient demographics
    """
    data = []

    main_path = Path(main_dir)
    patient_folders = sorted([f for f in main_path.iterdir() if f.is_dir() and f.name.isdigit()])

    print(f"Found {len(patient_folders)} patient folders")

    for patient_folder in patient_folders:
        patient_id = patient_folder.name
        flair_path = patient_folder / "FLAIR"

        if not flair_path.exists():
            print(f"Warning: FLAIR folder not found for patient {patient_id}")
            continue

        # Get all files from FLAIR folder (DICOM files often have no extension)
        dicom_files = [f for f in flair_path.iterdir() if f.is_file() and not str(f).endswith('.png')]

        if not dicom_files:
            print(f"Warning: No files found for patient {patient_id}")
            continue

        try:
            # Read the first DICOM file (try multiple files if first fails)
            dcm = None
            for dicom_file in dicom_files[:5]:  # Try up to 5 files
                try:
                    dcm = pydicom.dcmread(str(dicom_file), force=True)
                    break
                except:
                    continue

            if dcm is None:
                print(f"Warning: Could not read DICOM files for patient {patient_id}")
                continue

            # Extract demographic information
            patient_info = {
                'Patient_ID': patient_id,
                'Age': None,
                'Sex': None,
            }

            # Patient Age
            if hasattr(dcm, 'PatientAge'):
                age_str = dcm.PatientAge
                # Age format is typically like '045Y' for 45 years
                if age_str:
                    age_num = ''.join(filter(str.isdigit, age_str))
                    if age_num:
                        patient_info['Age'] = int(age_num)

            # If PatientAge not available, calculate from birth date and study date
            if patient_info['Age'] is None:
                if hasattr(dcm, 'PatientBirthDate') and hasattr(dcm, 'StudyDate'):
                    try:
                        birth_date = datetime.strptime(dcm.PatientBirthDate, '%Y%m%d')
                        study_date = datetime.strptime(dcm.StudyDate, '%Y%m%d')
                        age = (study_date - birth_date).days // 365
                        patient_info['Age'] = age
                    except:
                        pass

            # Patient Sex
            if hasattr(dcm, 'PatientSex'):
                patient_info['Sex'] = dcm.PatientSex

            data.append(patient_info)

        except Exception as e:
            print(f"Error processing patient {patient_id}: {str(e)}")
            continue

    return pd.DataFrame(data)


def create_table1(df):
    """
    Create Table 1 matching the specified format with cohort breakdowns.

    Args:
        df: DataFrame with patient demographics (Patient_ID, Age, Sex)

    Returns:
        DataFrame formatted as Table 1
    """
    table_rows = []

    # Clean up sex data - standardize to F/M
    df['Sex'] = df['Sex'].str.upper().str.strip()
    df = df[df['Sex'].isin(['F', 'M'])]  # Keep only F and M

    # Filter out patients without age data
    df_with_age = df[df['Age'].notna()].copy()

    # Function to calculate statistics for a cohort
    def calc_stats(cohort_df, cohort_name):
        if len(cohort_df) == 0:
            return None

        # Patient count
        n_patients = len(cohort_df)

        # Female:Male ratio
        n_female = len(cohort_df[cohort_df['Sex'] == 'F'])
        n_male = len(cohort_df[cohort_df['Sex'] == 'M'])
        fm_ratio = f"{n_female}:{n_male}"

        # Age statistics
        ages = cohort_df['Age'].dropna()
        if len(ages) > 0:
            age_mean = ages.mean()
            age_sd = ages.std()
            age_min = ages.min()
            age_max = ages.max()
            age_median = ages.median()
            age_q1 = ages.quantile(0.25)
            age_q3 = ages.quantile(0.75)

            age_mean_sd = f"{age_mean:.1f} ± {age_sd:.1f}"
            age_range = f"{int(age_min)}-{int(age_max)}"
            age_median_val = f"{age_median:.1f}"
            age_iqr = f"{age_q1:.1f}-{age_q3:.1f}"
        else:
            age_mean_sd = "N/A"
            age_range = "N/A"
            age_median_val = "N/A"
            age_iqr = "N/A"

        return {
            'cohort': cohort_name,
            'patients': n_patients,
            'Female:Male': fm_ratio,
            'Age Mean ± SD': age_mean_sd,
            'Age Range': age_range,
            'Age Median': age_median_val,
            'Age IQR': age_iqr
        }

    # Calculate for all patients
    all_stats = calc_stats(df_with_age, 'all')
    if all_stats:
        table_rows.append(all_stats)

    # Calculate for female patients
    female_df = df_with_age[df_with_age['Sex'] == 'F']
    female_stats = calc_stats(female_df, 'female')
    if female_stats:
        table_rows.append(female_stats)

    # Calculate for male patients
    male_df = df_with_age[df_with_age['Sex'] == 'M']
    male_stats = calc_stats(male_df, 'male')
    if male_stats:
        table_rows.append(male_stats)

    # Create DataFrame
    table1_df = pd.DataFrame(table_rows)

    return table1_df


def main():
    # Main directory path
    main_dir = r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_full"

    print("Starting demographic data extraction...")
    print(f"Main directory: {main_dir}\n")

    # Extract demographics from DICOM files
    demographics_df = extract_patient_demographics(main_dir)

    # Save raw data to CSV
    demographics_df.to_csv('patient_demographics_raw.csv', index=False)
    print(f"\nRaw demographic data saved to 'patient_demographics_raw.csv'")
    print(f"Successfully processed {len(demographics_df)} patients\n")

    # Display basic statistics
    print("=" * 70)
    print("RAW DATA SUMMARY")
    print("=" * 70)
    print(f"Total patients: {len(demographics_df)}")
    print(f"Patients with age data: {demographics_df['Age'].notna().sum()}")
    print(f"Patients with sex data: {demographics_df['Sex'].notna().sum()}")
    if demographics_df['Sex'].notna().any():
        sex_counts = demographics_df['Sex'].value_counts()
        print(f"\nSex distribution:")
        for sex, count in sex_counts.items():
            print(f"  {sex}: {count}")
    print("=" * 70 + "\n")

    # Create Table 1
    table1 = create_table1(demographics_df)

    # Display the table
    print("\n" + "=" * 100)
    print("TABLE 1: Demographic Characteristics of MS Patient Cohort")
    print("=" * 100)
    print(table1.to_string(index=False))
    print("=" * 100)

    # Save table to CSV (tab-delimited to match your format)
    table1.to_csv('Table1_Demographics.txt', index=False, sep='\t')
    print("\nTable 1 saved to 'Table1_Demographics.txt' (tab-delimited)")

    # Also save as regular CSV
    table1.to_csv('Table1_Demographics.csv', index=False)
    print("Table 1 saved to 'Table1_Demographics.csv'")

    # Create a LaTeX version
    with open('Table1_Demographics_LaTeX.txt', 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Demographic Characteristics of the MS Patient Cohort}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Cohort & Patients & Female:Male & Age Mean $\\pm$ SD & Age Range & Age Median & Age IQR \\\\\n")
        f.write("\\hline\n")
        for _, row in table1.iterrows():
            f.write(f"{row['cohort']} & {row['patients']} & {row['Female:Male']} & "
                    f"{row['Age Mean ± SD']} & {row['Age Range']} & "
                    f"{row['Age Median']} & {row['Age IQR']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:demographics}\n")
        f.write("\\end{table}\n")

    print("LaTeX version saved to 'Table1_Demographics_LaTeX.txt'\n")


if __name__ == "__main__":
    main()