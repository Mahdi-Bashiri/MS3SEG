#!/usr/bin/env python3
"""
MRI NIFTI File Creator from DICOMs - Final Production Version
====================================================

This script ...

System Requirements:
- Python 3.9+

Features:
-

Author: Mahdi Bashiri Bawil
Date: September 2025
Version: 1.0 (Production Ready)
"""


import os
import json
import logging
from pathlib import Path
import subprocess
import sys
from datetime import datetime

# Required libraries
try:
    import pydicom
    import nibabel as nib
    import numpy as np
    from dicom2nifti import dicom_series_to_nifti
    from dicom2nifti.settings import disable_validate_slice_increment
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install pydicom nibabel dicom2nifti")
    sys.exit(1)

# Set up logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ms_dicom_to_nifti_conversion.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Set console output to UTF-8 if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


class MSDatasetDicomToNiftiConverter:
    """
    Specialized DICOM to NIfTI converter for MS dataset structure
    Handles the specific directory structure: PatientID/Protocol/DICOM_files
    Enhanced for research use with comprehensive metadata preservation
    """

    def __init__(self, input_directory, output_directory, method='dicom2nifti'):
        """
        Initialize MS dataset converter

        Args:
            input_directory (str): Root directory containing patient folders (6-digit IDs)
            output_directory (str): Output directory for NIfTI files
            method (str): Conversion method ('dicom2nifti', 'dcm2niix', 'nibabel')
        """
        self.input_dir = Path(input_directory)
        self.output_dir = Path(output_directory)
        self.method = method
        self.converted_series = 0
        self.failed_series = 0
        self.processed_patients = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Disable strict validation for research data
        disable_validate_slice_increment()

        # MS protocol mappings for standardization
        self.protocol_mapping = {
            'FLAIR': 'FLAIR',
            'FLAIR_SG': 'FLAIR_SG',
            'T1WI': 'T1WI',
            'T2WI': 'T2WI',
            't1wi': 'T1WI',  # Handle case variations
            't2wi': 'T2WI',
            'flair': 'FLAIR',
            'flair_sg': 'FLAIR_SG'
        }

    def find_patient_folders(self):
        """Find all patient folders (6-digit numeric folders)"""
        patient_folders = []

        for item in self.input_dir.iterdir():
            if item.is_dir() and item.name.isdigit() and len(item.name) == 6:
                patient_folders.append(item)

        return sorted(patient_folders)

    def find_protocol_folders(self, patient_folder):
        """Find protocol folders within a patient folder"""
        protocol_folders = []

        for item in patient_folder.iterdir():
            if item.is_dir():
                protocol_folders.append(item)

        return protocol_folders

    def get_dicom_files(self, protocol_folder):
        """Get all DICOM files from a protocol folder, excluding PNG files"""
        dicom_files = []

        for file in protocol_folder.iterdir():
            if file.is_file() and not file.suffix.lower() == '.png':
                if self.is_dicom_file(file):
                    dicom_files.append(file)

        return sorted(dicom_files)

    def is_dicom_file(self, file_path):
        """Check if file is a valid DICOM file"""
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except:
            return False

    def standardize_protocol_name(self, protocol_name):
        """Standardize protocol names using mapping"""
        return self.protocol_mapping.get(protocol_name, protocol_name.upper())

    def extract_comprehensive_metadata(self, protocol_folder, dicom_files):
        """Extract comprehensive metadata from DICOM series for MS research"""
        if not dicom_files:
            return None

        # Read first and last DICOM files for comprehensive metadata
        first_ds = pydicom.dcmread(dicom_files[0])
        last_ds = pydicom.dcmread(dicom_files[-1]) if len(dicom_files) > 1 else first_ds

        # Helper function to safely get DICOM attribute
        def safe_get(ds, attr, default=None):
            try:
                value = getattr(ds, attr, default)
                if hasattr(value, 'value'):  # Handle DataElement objects
                    return value.value
                return value
            except:
                return default

        # Convert MultiValue objects to lists
        def convert_multivalue(value):
            if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                try:
                    return list(value)
                except:
                    return str(value)
            return value

        metadata = {
            # === PATIENT INFORMATION ===
            "PatientID": safe_get(first_ds, 'PatientID', 'Unknown'),
            "PatientName": str(safe_get(first_ds, 'PatientName', 'Anonymous')),
            "PatientAge": safe_get(first_ds, 'PatientAge', 'Unknown'),
            "PatientSex": safe_get(first_ds, 'PatientSex', 'Unknown'),
            "PatientBirthDate": safe_get(first_ds, 'PatientBirthDate', 'Unknown'),
            "PatientWeight": safe_get(first_ds, 'PatientWeight', None),
            "PatientSize": safe_get(first_ds, 'PatientSize', None),
            "PatientPosition": safe_get(first_ds, 'PatientPosition', 'Unknown'),

            # === STUDY INFORMATION ===
            "StudyDate": safe_get(first_ds, 'StudyDate', 'Unknown'),
            "StudyTime": safe_get(first_ds, 'StudyTime', 'Unknown'),
            "StudyDescription": safe_get(first_ds, 'StudyDescription', 'Unknown'),
            "StudyInstanceUID": safe_get(first_ds, 'StudyInstanceUID', 'Unknown'),
            "StudyID": safe_get(first_ds, 'StudyID', 'Unknown'),
            "AccessionNumber": safe_get(first_ds, 'AccessionNumber', 'Unknown'),
            "ReferringPhysicianName": str(safe_get(first_ds, 'ReferringPhysicianName', 'Unknown')),
            "InstitutionName": safe_get(first_ds, 'InstitutionName', 'Unknown'),
            "InstitutionAddress": safe_get(first_ds, 'InstitutionAddress', 'Unknown'),
            "InstitutionalDepartmentName": safe_get(first_ds, 'InstitutionalDepartmentName', 'Unknown'),

            # === SERIES INFORMATION ===
            "SeriesDescription": safe_get(first_ds, 'SeriesDescription', 'Unknown'),
            "SeriesNumber": safe_get(first_ds, 'SeriesNumber', 'Unknown'),
            "SeriesInstanceUID": safe_get(first_ds, 'SeriesInstanceUID', 'Unknown'),
            "SeriesDate": safe_get(first_ds, 'SeriesDate', 'Unknown'),
            "SeriesTime": safe_get(first_ds, 'SeriesTime', 'Unknown'),
            "ProtocolName": safe_get(first_ds, 'ProtocolName', protocol_folder.name),
            "Modality": safe_get(first_ds, 'Modality', 'MR'),

            # === MR ACQUISITION PARAMETERS (Critical for MS research) ===
            "MagneticFieldStrength": safe_get(first_ds, 'MagneticFieldStrength', None),
            "SliceThickness": safe_get(first_ds, 'SliceThickness', None),
            "SpacingBetweenSlices": safe_get(first_ds, 'SpacingBetweenSlices', None),
            "RepetitionTime": safe_get(first_ds, 'RepetitionTime', None),
            "EchoTime": safe_get(first_ds, 'EchoTime', None),
            "InversionTime": safe_get(first_ds, 'InversionTime', None),
            "FlipAngle": safe_get(first_ds, 'FlipAngle', None),
            "EchoNumbers": safe_get(first_ds, 'EchoNumbers', None),
            "NumberOfAverages": safe_get(first_ds, 'NumberOfAverages', None),
            "ImagingFrequency": safe_get(first_ds, 'ImagingFrequency', None),
            "ImagedNucleus": safe_get(first_ds, 'ImagedNucleus', None),
            "MagneticFieldStrength": safe_get(first_ds, 'MagneticFieldStrength', None),

            # === SPATIAL INFORMATION ===
            "PixelSpacing": convert_multivalue(safe_get(first_ds, 'PixelSpacing', None)),
            "Rows": safe_get(first_ds, 'Rows', None),
            "Columns": safe_get(first_ds, 'Columns', None),
            "AcquisitionMatrix": convert_multivalue(safe_get(first_ds, 'AcquisitionMatrix', None)),
            "FOVDimensions": convert_multivalue(safe_get(first_ds, 'FOVDimensions', None)),
            "SliceLocation": safe_get(first_ds, 'SliceLocation', None),
            "ImageOrientationPatient": convert_multivalue(safe_get(first_ds, 'ImageOrientationPatient', None)),
            "ImagePositionPatient": convert_multivalue(safe_get(first_ds, 'ImagePositionPatient', None)),

            # === SEQUENCE INFORMATION (Important for MS protocols) ===
            "ImageType": convert_multivalue(safe_get(first_ds, 'ImageType', 'Unknown')),
            "ScanningSequence": convert_multivalue(safe_get(first_ds, 'ScanningSequence', 'Unknown')),
            "SequenceVariant": convert_multivalue(safe_get(first_ds, 'SequenceVariant', 'Unknown')),
            "ScanOptions": convert_multivalue(safe_get(first_ds, 'ScanOptions', 'Unknown')),
            "MRAcquisitionType": safe_get(first_ds, 'MRAcquisitionType', 'Unknown'),
            "SequenceName": safe_get(first_ds, 'SequenceName', 'Unknown'),
            "PulseSequenceName": safe_get(first_ds, 'PulseSequenceName', 'Unknown'),

            # === EQUIPMENT INFORMATION ===
            "Manufacturer": safe_get(first_ds, 'Manufacturer', 'Unknown'),
            "ManufacturerModelName": safe_get(first_ds, 'ManufacturerModelName', 'Unknown'),
            "SoftwareVersions": convert_multivalue(safe_get(first_ds, 'SoftwareVersions', 'Unknown')),
            "DeviceSerialNumber": safe_get(first_ds, 'DeviceSerialNumber', 'Unknown'),
            "StationName": safe_get(first_ds, 'StationName', 'Unknown'),

            # === CONTRAST AND IMAGE PARAMETERS ===
            "ContrastBolusAgent": safe_get(first_ds, 'ContrastBolusAgent', None),
            "ContrastBolusRoute": safe_get(first_ds, 'ContrastBolusRoute', None),
            "ContrastBolusVolume": safe_get(first_ds, 'ContrastBolusVolume', None),
            "WindowCenter": convert_multivalue(safe_get(first_ds, 'WindowCenter', None)),
            "WindowWidth": convert_multivalue(safe_get(first_ds, 'WindowWidth', None)),
            "RescaleIntercept": safe_get(first_ds, 'RescaleIntercept', None),
            "RescaleSlope": safe_get(first_ds, 'RescaleSlope', None),

            # === TIMING AND PHYSIOLOGICAL ===
            "AcquisitionTime": safe_get(first_ds, 'AcquisitionTime', None),
            "ContentTime": safe_get(first_ds, 'ContentTime', None),
            "TriggerTime": safe_get(first_ds, 'TriggerTime', None),
            "HeartRate": safe_get(first_ds, 'HeartRate', None),
            "CardiacNumberOfImages": safe_get(first_ds, 'CardiacNumberOfImages', None),

            # === BODY PART AND POSITIONING ===
            "BodyPartExamined": safe_get(first_ds, 'BodyPartExamined', 'Unknown'),
            "PatientPosition": safe_get(first_ds, 'PatientPosition', 'Unknown'),
            "ViewPosition": safe_get(first_ds, 'ViewPosition', None),

            # === SLICE INFORMATION ===
            "SliceLocation_First": safe_get(first_ds, 'SliceLocation', None),
            "SliceLocation_Last": safe_get(last_ds, 'SliceLocation', None),
            "InstanceNumber_First": safe_get(first_ds, 'InstanceNumber', None),
            "InstanceNumber_Last": safe_get(last_ds, 'InstanceNumber', None),

            # === FILE AND PROCESSING INFORMATION ===
            "NumberOfSlices": len(dicom_files),
            "DicomFiles": [f.name for f in dicom_files],
            "ProtocolFolder": protocol_folder.name,
            "OriginalPath": str(protocol_folder),
            "ConversionDate": datetime.now().isoformat(),
            "ConversionMethod": self.method,

            # === MS-SPECIFIC RESEARCH METADATA ===
            "MSProtocolType": self.standardize_protocol_name(protocol_folder.name),
            "IsFlairSequence": "FLAIR" in protocol_folder.name.upper(),
            "IsT1Weighted": "T1" in protocol_folder.name.upper(),
            "IsT2Weighted": "T2" in protocol_folder.name.upper(),
            "IsSuppressedFlair": "SG" in protocol_folder.name.upper(),

            # === ADDITIONAL TECHNICAL PARAMETERS ===
            "BitsAllocated": safe_get(first_ds, 'BitsAllocated', None),
            "BitsStored": safe_get(first_ds, 'BitsStored', None),
            "HighBit": safe_get(first_ds, 'HighBit', None),
            "PixelRepresentation": safe_get(first_ds, 'PixelRepresentation', None),
            "PhotometricInterpretation": safe_get(first_ds, 'PhotometricInterpretation', None),
            "SamplesPerPixel": safe_get(first_ds, 'SamplesPerPixel', None),

            # === RESEARCH-SPECIFIC QUALITY METRICS ===
            "SNR_Estimated": None,  # Can be calculated post-conversion
            "ImageQualityScore": None,  # For future quality assessment
            "MotionArtifacts": None,  # For future artifact detection
        }

        # Convert numpy arrays and special types to JSON-serializable formats
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)) and value is not None:
                try:
                    metadata[key] = list(value)
                except:
                    metadata[key] = str(value)
            elif value is not None and not isinstance(value, (int, float, str, bool, list, dict)):
                metadata[key] = str(value)

        return metadata

    def convert_with_dicom2nifti(self, protocol_folder, output_path):
        """Convert using dicom2nifti library (fixed - removed invalid reorient parameter)"""
        try:
            # Fixed: removed the invalid 'reorient' parameter
            dicom_series_to_nifti(str(protocol_folder), str(output_path))
            return True
        except Exception as e:
            logging.error(f"dicom2nifti conversion failed for {protocol_folder}: {e}")
            return False

    def convert_with_dcm2niix(self, protocol_folder, output_path):
        """Convert using dcm2niix"""
        try:
            # Check if dcm2niix is available
            subprocess.run(['dcm2niix', '-h'], capture_output=True, check=True)

            # Run dcm2niix with research-optimized settings
            cmd = [
                'dcm2niix',
                '-z', 'y',  # Compress output
                '-f', output_path.stem,  # Output filename
                '-o', str(output_path.parent),  # Output directory
                '-b', 'y',  # Create BIDS sidecar JSON
                '-ba', 'y',  # Anonymize BIDS
                '-v', 'y',  # Verbose output
                str(protocol_folder)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return True
            else:
                logging.error(f"dcm2niix failed for {protocol_folder}: {result.stderr}")
                return False

        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.error("dcm2niix not found. Install from: https://github.com/rordenlab/dcm2niix")
            return False

    def convert_with_nibabel(self, protocol_folder, output_path):
        """Convert using nibabel with enhanced header preservation"""
        try:
            dicom_files = self.get_dicom_files(protocol_folder)

            if not dicom_files:
                return False

            # Read DICOM files
            slices = []
            for dicom_file in dicom_files:
                ds = pydicom.dcmread(dicom_file)
                slices.append(ds)

            # Sort by slice location or instance number
            try:
                slices.sort(key=lambda x: float(getattr(x, 'SliceLocation', 0)))
            except:
                try:
                    slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
                except:
                    logging.warning(f"Could not sort slices for {protocol_folder}")

            # Create 3D array
            pixel_arrays = [s.pixel_array.astype(np.float32) for s in slices]
            volume = np.stack(pixel_arrays, axis=-1)

            # Get voxel spacing and create proper affine matrix
            ds = slices[0]
            pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(ds, 'SliceThickness', 1.0)

            # Create more accurate affine matrix
            affine = np.eye(4)
            affine[0, 0] = float(pixel_spacing[0])
            affine[1, 1] = float(pixel_spacing[1])
            affine[2, 2] = float(slice_thickness)

            # Try to get image position for better spatial alignment
            try:
                img_pos = getattr(ds, 'ImagePositionPatient', [0, 0, 0])
                affine[0, 3] = float(img_pos[0])
                affine[1, 3] = float(img_pos[1])
                affine[2, 3] = float(img_pos[2])
            except:
                pass

            # Create NIfTI image with enhanced header
            nifti_img = nib.Nifti1Image(volume, affine)

            # Set additional header information
            header = nifti_img.header
            header.set_xyzt_units('mm', 'sec')

            # Set data type appropriately
            if volume.dtype == np.float32:
                header.set_data_dtype(np.float32)
            else:
                header.set_data_dtype(volume.dtype)

            nib.save(nifti_img, output_path)
            return True

        except Exception as e:
            logging.error(f"nibabel conversion failed for {protocol_folder}: {e}")
            return False

    def process_patient(self, patient_folder):
        """Process all protocols for a single patient"""
        patient_id = patient_folder.name
        logging.info(f"Processing patient: {patient_id}")

        # Create patient output directory
        patient_output_dir = self.output_dir / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        # Find protocol folders
        protocol_folders = self.find_protocol_folders(patient_folder)

        if not protocol_folders:
            logging.warning(f"No protocol folders found for patient {patient_id}")
            return

        patient_conversions = 0
        patient_failures = 0

        for protocol_folder in protocol_folders:
            protocol_name = protocol_folder.name
            standardized_protocol = self.standardize_protocol_name(protocol_name)

            # Check if folder contains DICOM files
            dicom_files = self.get_dicom_files(protocol_folder)

            if not dicom_files:
                logging.warning(f"No DICOM files found in {protocol_folder}")
                patient_failures += 1
                continue

            # Generate output filename: {6-digit patientID}_{protocol}.nii.gz
            filename = f"{patient_id}_{standardized_protocol}.nii.gz"
            nifti_path = patient_output_dir / filename
            json_path = patient_output_dir / filename.replace('.nii.gz', '.json')

            logging.info(f"  Converting {protocol_name} -> {filename}")

            # Convert based on selected method
            success = False
            if self.method == 'dicom2nifti':
                success = self.convert_with_dicom2nifti(protocol_folder, nifti_path)
            elif self.method == 'dcm2niix':
                success = self.convert_with_dcm2niix(protocol_folder, nifti_path)
            elif self.method == 'nibabel':
                success = self.convert_with_nibabel(protocol_folder, nifti_path)

            if success:
                patient_conversions += 1
                self.converted_series += 1
                # Using regular characters instead of special Unicode characters
                logging.info(f"    [SUCCESS] Successfully converted: {nifti_path}")

                # Save comprehensive metadata as JSON
                metadata = self.extract_comprehensive_metadata(protocol_folder, dicom_files)
                if metadata:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
                    logging.info(f"    [SUCCESS] Metadata saved: {json_path}")
            else:
                patient_failures += 1
                self.failed_series += 1
                # Using regular characters instead of special Unicode characters
                logging.error(f"    [FAILED] Failed to convert: {protocol_folder}")

        logging.info(f"Patient {patient_id} completed: {patient_conversions} successful, {patient_failures} failed")
        self.processed_patients += 1

    def convert_all(self):
        """Convert all patients and protocols"""
        patient_folders = self.find_patient_folders()

        if not patient_folders:
            logging.error("No patient folders (6-digit numeric) found in input directory")
            return

        logging.info(f"Found {len(patient_folders)} patient folders to process")
        logging.info(f"Expected protocols per patient: FLAIR, FLAIR_SG, T1WI, T2WI")

        for i, patient_folder in enumerate(patient_folders, 1):
            logging.info(f"\nProcessing patient {i}/{len(patient_folders)}: {patient_folder.name}")
            self.process_patient(patient_folder)

        self.print_summary()

    def print_summary(self):
        """Print conversion summary"""
        print("\n" + "=" * 80)
        print("MS DATASET DICOM TO NIFTI CONVERSION SUMMARY")
        print("=" * 80)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Conversion method: {self.method}")
        print("-" * 80)
        print(f"Patients processed: {self.processed_patients}")
        print(f"Protocol series successfully converted: {self.converted_series}")
        print(f"Protocol series failed: {self.failed_series}")
        print(f"Total protocol series: {self.converted_series + self.failed_series}")
        if (self.converted_series + self.failed_series) > 0:
            success_rate = (self.converted_series / (self.converted_series + self.failed_series) * 100)
            print(f"Success rate: {success_rate:.1f}%")
        else:
            print("Success rate: N/A")
        print("-" * 80)
        print("OUTPUT STRUCTURE:")
        print("Output_Directory/")
        print("  |-- 001/")
        print("  |   |-- 001_FLAIR.nii.gz")
        print("  |   |-- 001_FLAIR.json")
        print("  |   |-- 001_FLAIR_SG.nii.gz")
        print("  |   |-- 001_FLAIR_SG.json")
        print("  |   |-- 001_T1WI.nii.gz")
        print("  |   |-- 001_T1WI.json")
        print("  |   |-- 001_T2WI.nii.gz")
        print("  |   |-- 001_T2WI.json")
        print("  |-- 002/")
        print("      |-- ...")
        print("-" * 80)
        print("RESEARCH-READY FEATURES:")
        print("+ MS-specific protocol standardization")
        print("+ Comprehensive DICOM header preservation")
        print("+ Research-quality NIfTI format with proper spatial information")
        print("+ Detailed JSON metadata files with >50 DICOM attributes")
        print("+ Professional naming convention: {PatientID}_{Protocol}.nii.gz")
        print("+ UTF-8 encoding support for international characters")
        print("+ Enhanced spatial alignment and orientation")
        if self.method == 'dcm2niix':
            print("+ BIDS-compatible output")
        print("=" * 80)


# Example usage
if __name__ == "__main__":
    # MS Dataset directories
    input_directory = r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_full"
    output_directory = r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_nifti"

    print("MS DATASET DICOM TO NIFTI CONVERTER - RESEARCH EDITION")
    print("=" * 60)
    print("Directory Structure Expected:")
    print("Input/")
    print("  |-- 001/        # 6-digit patient ID")
    print("  |   |-- FLAIR/     # Protocol folders")
    print("  |   |-- FLAIR_SG/")
    print("  |   |-- T1WI/")
    print("  |   |-- T2WI/")
    print("  |-- 002/")
    print("      |-- ...")
    print("=" * 60)
    print("Available conversion methods:")
    print("1. 'dicom2nifti' - Reliable Python library (recommended)")
    print("2. 'dcm2niix' - Most comprehensive (requires installation)")
    print("3. 'nibabel' - Enhanced fallback method with better header preservation")
    print("=" * 60)

    # Initialize converter with dicom2nifti (recommended for MS research)
    converter = MSDatasetDicomToNiftiConverter(
        input_directory=input_directory,
        output_directory=output_directory,
        method='dicom2nifti'  # Change to 'dcm2niix' if available
    )

    # Start conversion
    converter.convert_all()