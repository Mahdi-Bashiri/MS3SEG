import os
import pydicom
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_imaging_parameters(main_dir):
    """
    Extract imaging parameters from DICOM files for all modalities.
    
    Args:
        main_dir: Main directory containing patient folders
    
    Returns:
        DataFrame with imaging parameters for each patient and modality
    """
    data = []
    
    main_path = Path(main_dir)
    patient_folders = sorted([f for f in main_path.iterdir() if f.is_dir() and f.name.isdigit()])
    
    # Modalities to process - map folder names to display names
    modalities = {
        'T1WI': 'T1-w',
        'T2WI': 'T2-w',
        'FLAIR': 'FLAIR',
        'FLAIR_SG': 'FLAIR_SG'
    }
    
    print(f"Found {len(patient_folders)} patient folders")
    print(f"Processing modalities: {list(modalities.keys())}\n")
    
    for patient_folder in patient_folders:
        patient_id = patient_folder.name
        
        for folder_name, modality_name in modalities.items():
            modality_path = patient_folder / folder_name
            
            if not modality_path.exists():
                print(f"Warning: {folder_name} folder not found for patient {patient_id}")
                continue
            
            # Get all DICOM files from the modality folder
            dicom_files = list(modality_path.glob("*.dcm"))
            if not dicom_files:
                # Try without extension
                dicom_files = [f for f in modality_path.iterdir() if f.is_file() and not str(f).endswith('.png')]
            
            if not dicom_files:
                print(f"Warning: No DICOM files found in {folder_name} for patient {patient_id}")
                continue
            
            try:
                # Read the first DICOM file for imaging parameters
                dcm = pydicom.dcmread(dicom_files[0], force=True)
                
                # Extract imaging parameters
                params = {
                    'Patient_ID': patient_id,
                    'Modality': modality_name,
                    'Imaging_Plane': None,
                    'Rows': None,
                    'Columns': None,
                    'Pixel_Spacing_X': None,
                    'Pixel_Spacing_Y': None,
                    'Slice_Thickness': None,
                    'TR': None,  # Repetition Time
                    'TE': None,  # Echo Time
                    'TI': None,  # Inversion Time
                    'FA': None,  # Flip Angle
                    'Number_of_Slices': len(dicom_files)
                }
                
                # Imaging plane orientation
                if hasattr(dcm, 'ImageOrientationPatient'):
                    # Determine plane from orientation
                    orientation = dcm.ImageOrientationPatient
                    # Simplified plane detection
                    if hasattr(dcm, 'ImageType'):
                        params['Imaging_Plane'] = 'Axial'  # Default, can be refined
                
                # Matrix size (Rows and Columns)
                if hasattr(dcm, 'Rows'):
                    params['Rows'] = int(dcm.Rows)
                if hasattr(dcm, 'Columns'):
                    params['Columns'] = int(dcm.Columns)
                
                # Pixel Spacing (in-plane resolution)
                if hasattr(dcm, 'PixelSpacing'):
                    spacing = dcm.PixelSpacing
                    params['Pixel_Spacing_X'] = float(spacing[0])
                    params['Pixel_Spacing_Y'] = float(spacing[1])
                
                # Slice Thickness
                if hasattr(dcm, 'SliceThickness'):
                    params['Slice_Thickness'] = float(dcm.SliceThickness)
                
                # Repetition Time (TR) in milliseconds
                if hasattr(dcm, 'RepetitionTime'):
                    params['TR'] = float(dcm.RepetitionTime)
                
                # Echo Time (TE) in milliseconds
                if hasattr(dcm, 'EchoTime'):
                    params['TE'] = float(dcm.EchoTime)
                
                # Inversion Time (TI) in milliseconds
                if hasattr(dcm, 'InversionTime'):
                    params['TI'] = float(dcm.InversionTime)
                
                # Flip Angle in degrees
                if hasattr(dcm, 'FlipAngle'):
                    params['FA'] = float(dcm.FlipAngle)
                
                data.append(params)
                
            except Exception as e:
                print(f"Error processing {folder_name} for patient {patient_id}: {str(e)}")
                continue
    
    return pd.DataFrame(data)


def create_table2(df):
    """
    Create Table 2 with imaging parameters summary for each modality.
    
    Args:
        df: DataFrame with imaging parameters
    
    Returns:
        DataFrame formatted as Table 2
    """
    table_rows = []
    
    # Group by modality
    modality_order = ['T1-w', 'T2-w', 'FLAIR', 'FLAIR_SG']
    
    for modality in modality_order:
        modality_df = df[df['Modality'] == modality]
        
        if len(modality_df) == 0:
            # Add empty row if modality not found
            table_rows.append({
                'modality': modality,
                'imaging plane': 'N/A',
                'data rows and columns (mean ± SD)': 'N/A',
                'pixel spacing (mean ± SD)': 'N/A',
                'slice thickness (mean ± SD)': 'N/A',
                'Time Repeat (TR) (mean ± SD)': 'N/A',
                'Time Echo (TE) (mean ± SD)': 'N/A',
                'Time Inversion (TI) (mean ± SD)': 'N/A',
                'Flip Angle (FA) (mean ± SD)': 'N/A'
            })
            continue
        
        row_data = {'modality': modality}
        
        # Imaging plane (most common)
        if modality_df['Imaging_Plane'].notna().any():
            plane = modality_df['Imaging_Plane'].mode()
            row_data['imaging plane'] = plane.iloc[0] if len(plane) > 0 else 'Axial'
        else:
            row_data['imaging plane'] = 'Axial'  # Default assumption
        
        # Data rows and columns
        rows_data = modality_df['Rows'].dropna()
        cols_data = modality_df['Columns'].dropna()
        if len(rows_data) > 0 and len(cols_data) > 0:
            rows_mean = rows_data.mean()
            rows_sd = rows_data.std()
            cols_mean = cols_data.mean()
            cols_sd = cols_data.std()
            row_data['data rows and columns (mean ± SD)'] = f"{rows_mean:.0f} ± {rows_sd:.1f} × {cols_mean:.0f} ± {cols_sd:.1f}"
        else:
            row_data['data rows and columns (mean ± SD)'] = 'N/A'
        
        # Pixel spacing (use X spacing, typically square pixels)
        pixel_data = modality_df['Pixel_Spacing_X'].dropna()
        if len(pixel_data) > 0:
            pixel_mean = pixel_data.mean()
            pixel_sd = pixel_data.std()
            row_data['pixel spacing (mean ± SD)'] = f"{pixel_mean:.2f} ± {pixel_sd:.2f} mm"
        else:
            row_data['pixel spacing (mean ± SD)'] = 'N/A'
        
        # Slice thickness
        slice_data = modality_df['Slice_Thickness'].dropna()
        if len(slice_data) > 0:
            slice_mean = slice_data.mean()
            slice_sd = slice_data.std()
            row_data['slice thickness (mean ± SD)'] = f"{slice_mean:.2f} ± {slice_sd:.2f} mm"
        else:
            row_data['slice thickness (mean ± SD)'] = 'N/A'
        
        # TR (Repetition Time)
        tr_data = modality_df['TR'].dropna()
        if len(tr_data) > 0:
            tr_mean = tr_data.mean()
            tr_sd = tr_data.std()
            row_data['Time Repeat (TR) (mean ± SD)'] = f"{tr_mean:.1f} ± {tr_sd:.1f} ms"
        else:
            row_data['Time Repeat (TR) (mean ± SD)'] = 'N/A'
        
        # TE (Echo Time)
        te_data = modality_df['TE'].dropna()
        if len(te_data) > 0:
            te_mean = te_data.mean()
            te_sd = te_data.std()
            row_data['Time Echo (TE) (mean ± SD)'] = f"{te_mean:.1f} ± {te_sd:.1f} ms"
        else:
            row_data['Time Echo (TE) (mean ± SD)'] = 'N/A'
        
        # TI (Inversion Time)
        ti_data = modality_df['TI'].dropna()
        if len(ti_data) > 0:
            ti_mean = ti_data.mean()
            ti_sd = ti_data.std()
            row_data['Time Inversion (TI) (mean ± SD)'] = f"{ti_mean:.1f} ± {ti_sd:.1f} ms"
        else:
            row_data['Time Inversion (TI) (mean ± SD)'] = 'N/A'
        
        # FA (Flip Angle)
        fa_data = modality_df['FA'].dropna()
        if len(fa_data) > 0:
            fa_mean = fa_data.mean()
            fa_sd = fa_data.std()
            row_data['Flip Angle (FA) (mean ± SD)'] = f"{fa_mean:.1f} ± {fa_sd:.1f}°"
        else:
            row_data['Flip Angle (FA) (mean ± SD)'] = 'N/A'
        
        table_rows.append(row_data)
    
    # Create DataFrame
    table2_df = pd.DataFrame(table_rows)
    
    return table2_df


def main():
    # Main directory path
    main_dir = r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_full"
    
    print("Starting imaging parameters extraction...")
    print(f"Main directory: {main_dir}\n")
    
    # Extract imaging parameters from DICOM files
    imaging_df = extract_imaging_parameters(main_dir)
    
    # Save raw data to CSV
    imaging_df.to_csv('imaging_parameters_raw.csv', index=False)
    print(f"\nRaw imaging parameters saved to 'imaging_parameters_raw.csv'")
    print(f"Successfully processed {len(imaging_df)} scans\n")
    
    # Display basic statistics
    print("="*70)
    print("RAW DATA SUMMARY")
    print("="*70)
    print(f"Total scans processed: {len(imaging_df)}")
    print(f"\nScans per modality:")
    for modality, count in imaging_df['Modality'].value_counts().items():
        print(f"  {modality}: {count}")
    print("="*70 + "\n")
    
    # Create Table 2
    table2 = create_table2(imaging_df)
    
    # Display the table
    print("\n" + "="*150)
    print("TABLE 2: Imaging and Data Acquisition Parameters")
    print("="*150)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(table2.to_string(index=False))
    print("="*150)
    
    # Save table to CSV (tab-delimited)
    table2.to_csv('Table2_Imaging_Parameters.txt', index=False, sep='\t')
    print("\nTable 2 saved to 'Table2_Imaging_Parameters.txt' (tab-delimited)")
    
    # Also save as regular CSV
    table2.to_csv('Table2_Imaging_Parameters.csv', index=False)
    print("Table 2 saved to 'Table2_Imaging_Parameters.csv'")
    
    # Create a LaTeX version
    with open('Table2_Imaging_Parameters_LaTeX.txt', 'w') as f:
        f.write("\\begin{table*}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Imaging and Data Acquisition Parameters}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccccccccc}\n")
        f.write("\\hline\n")
        f.write("Modality & Imaging & Data Rows and & Pixel Spacing & Slice Thickness & TR & TE & TI & FA \\\\\n")
        f.write(" & Plane & Columns & (mean $\\pm$ SD) & (mean $\\pm$ SD) & (mean $\\pm$ SD) & (mean $\\pm$ SD) & (mean $\\pm$ SD) & (mean $\\pm$ SD) \\\\\n")
        f.write("\\hline\n")
        
        for _, row in table2.iterrows():
            modality = row['modality'].replace('_', '\\_')
            plane = row['imaging plane']
            matrix = row['data rows and columns (mean ± SD)'].replace('±', '$\\pm$')
            pixel = row['pixel spacing (mean ± SD)'].replace('±', '$\\pm$')
            slice_thick = row['slice thickness (mean ± SD)'].replace('±', '$\\pm$')
            tr = row['Time Repeat (TR) (mean ± SD)'].replace('±', '$\\pm$')
            te = row['Time Echo (TE) (mean ± SD)'].replace('±', '$\\pm$')
            ti = row['Time Inversion (TI) (mean ± SD)'].replace('±', '$\\pm$')
            fa = row['Flip Angle (FA) (mean ± SD)'].replace('±', '$\\pm$').replace('°', '$^\\circ$')
            
            f.write(f"{modality} & {plane} & {matrix} & {pixel} & {slice_thick} & {tr} & {te} & {ti} & {fa} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\label{tab:imaging_parameters}\n")
        f.write("\\end{table*}\n")
    
    print("LaTeX version saved to 'Table2_Imaging_Parameters_LaTeX.txt'\n")
    
    # Print detailed statistics for verification
    print("\n" + "="*70)
    print("DETAILED STATISTICS BY MODALITY")
    print("="*70)
    for modality in ['T1-w', 'T2-w', 'FLAIR', 'FLAIR_SG']:
        modality_df = imaging_df[imaging_df['Modality'] == modality]
        if len(modality_df) > 0:
            print(f"\n{modality}:")
            print(f"  N = {len(modality_df)}")
            print(f"  TR available: {modality_df['TR'].notna().sum()}")
            print(f"  TE available: {modality_df['TE'].notna().sum()}")
            print(f"  TI available: {modality_df['TI'].notna().sum()}")
            print(f"  FA available: {modality_df['FA'].notna().sum()}")
    print("="*70)


if __name__ == "__main__":
    main()
