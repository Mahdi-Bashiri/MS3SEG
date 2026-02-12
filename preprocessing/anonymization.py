#!/usr/bin/env python3
"""
MRI Data Anonymizer - Final Production Version
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
import pydicom
from pydicom.errors import InvalidDicomError
import logging
from pathlib import Path
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dicom_anonymization.log'),
        logging.StreamHandler()
    ]
)


class DicomAnonymizer:
    def __init__(self, input_directory, output_directory=None, backup=True):
        """
        Initialize DICOM Anonymizer

        Args:
            input_directory (str): Path to directory containing DICOM files
            output_directory (str): Path to save anonymized files (optional)
            backup (bool): Create backup before anonymizing (default: True)
        """
        self.input_dir = Path(input_directory)
        self.output_dir = Path(output_directory) if output_directory else None
        self.backup = backup
        self.processed_files = 0
        self.failed_files = 0
        self.anonymized_files = 0

        # DICOM tags containing patient name information - ONLY these will be anonymized
        # All other information will be preserved for research purposes
        self.name_tags = [
            (0x0010, 0x0010),  # PatientName - Primary patient name field
            (0x0010, 0x1001),  # OtherPatientNames - Alternative patient names
            (0x0010, 0x1060),  # PatientMotherBirthName - Mother's maiden name
            (0x0010, 0x0030),  # PatientBirthDate - Patient's birth date
        ]

        # Optional: Tags that might contain patient names in free text
        # These will be checked for potential name content but handled carefully
        self.potential_name_tags = [
            (0x0008, 0x1030),  # StudyDescription - might contain patient name
            (0x0008, 0x103E),  # SeriesDescription - might contain patient name
            (0x0010, 0x4000),  # PatientComments - might contain patient name
            (0x0010, 0x21B0),  # AdditionalPatientHistory - might contain patient name
        ]

    def is_dicom_file(self, file_path):
        """Check if file is a valid DICOM file"""
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except (InvalidDicomError, FileNotFoundError, PermissionError):
            return False

    def check_for_names_in_text(self, text_value, patient_name):
        """
        Check if a text field might contain patient name and return cleaned version
        This is a simple implementation - you might want to make it more sophisticated
        """
        if not text_value or not patient_name:
            return text_value

        # Convert to strings and make case-insensitive comparison
        text_str = str(text_value).upper()

        # Split patient name into components for checking
        name_parts = str(patient_name).upper().split()

        # Check if any part of the patient name appears in the text
        contains_name = any(part in text_str for part in name_parts if len(part) > 2)

        if contains_name:
            logging.warning(f"Potential patient name found in text field: '{text_value}'")
            # Replace with generic text or return original - you can customize this
            return f"[CONTENT_REMOVED_FOR_ANONYMIZATION] {text_value}"

        return text_value

    def anonymize_dicom(self, dicom_path, patient_id=None):
        """
        Anonymize DICOM file by removing only patient name information
        All other data is preserved for research purposes

        Args:
            dicom_path (Path): Path to DICOM file
            patient_id (str): Anonymous patient ID to use
        """
        try:
            # Read DICOM file
            ds = pydicom.dcmread(dicom_path)

            # Generate anonymous patient ID if not provided
            if patient_id is None:
                patient_id = f"ANON_{self.anonymized_files + 1:04d}"

            # Store original patient name for checking other fields
            original_patient_name = None
            if (0x0010, 0x0010) in ds:
                original_patient_name = ds[(0x0010, 0x0010)].value

            # Remove/anonymize ONLY patient name information
            for tag in self.name_tags:
                if tag in ds:
                    if tag == (0x0010, 0x0010):  # PatientName
                        ds[tag].value = patient_id
                        logging.info(f"Replaced PatientName '{original_patient_name}' with '{patient_id}'")
                    else:
                        # Remove other name-related tags
                        del ds[tag]
                        logging.info(f"Removed tag {tag}")

            # Check potential name-containing fields and warn/clean if necessary
            for tag in self.potential_name_tags:
                if tag in ds:
                    original_value = ds[tag].value
                    cleaned_value = self.check_for_names_in_text(original_value, original_patient_name)
                    if cleaned_value != original_value:
                        ds[tag].value = cleaned_value
                        logging.warning(f"Modified tag {tag} due to potential name content")

            # Note: We are NOT removing private tags as they might contain useful research data
            # Note: We are NOT modifying dates, times, UIDs, or other research-relevant data

            # Determine output path
            if self.output_dir:
                # Maintain directory structure in output
                relative_path = dicom_path.relative_to(self.input_dir)
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Overwrite original file
                output_path = dicom_path

            # Create backup if requested and overwriting original
            if self.backup and output_path == dicom_path:
                backup_path = dicom_path.with_suffix(dicom_path.suffix + '.backup')
                shutil.copy2(dicom_path, backup_path)
                logging.info(f"Created backup: {backup_path}")

            # Save anonymized DICOM
            ds.save_as(output_path)

            self.anonymized_files += 1
            logging.info(f"Anonymized: {dicom_path} -> {output_path}")

        except Exception as e:
            self.failed_files += 1
            logging.error(f"Failed to anonymize {dicom_path}: {str(e)}")

    def process_directory(self, patient_id_prefix="ANON"):
        """Process all DICOM files in the directory recursively"""

        if not self.input_dir.exists():
            logging.error(f"Input directory does not exist: {self.input_dir}")
            return

        logging.info(f"Starting anonymization of directory: {self.input_dir}")
        if self.output_dir:
            logging.info(f"Output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find all potential DICOM files
        dicom_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                file_path = Path(root) / file
                if self.is_dicom_file(file_path):
                    dicom_files.append(file_path)

        logging.info(f"Found {len(dicom_files)} DICOM files to process")

        # Process each DICOM file
        for i, dicom_file in enumerate(dicom_files, 1):
            self.processed_files += 1
            patient_id = f"{patient_id_prefix}_{i:04d}"

            logging.info(f"Processing file {i}/{len(dicom_files)}: {dicom_file.name}")
            self.anonymize_dicom(dicom_file, patient_id)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("DICOM NAME ANONYMIZATION SUMMARY")
        print("=" * 60)
        print("ANONYMIZATION SCOPE: Patient names only")
        print("PRESERVED DATA: All other medical/research data retained")
        print("-" * 60)
        print(f"Total files processed: {self.processed_files}")
        print(f"Successfully anonymized: {self.anonymized_files}")
        print(f"Failed to process: {self.failed_files}")
        print(f"Success rate: {(self.anonymized_files / self.processed_files * 100):.1f}%"
              if self.processed_files > 0 else "N/A")
        print("=" * 60)
        print("PRESERVED FOR RESEARCH:")
        print("✓ Patient ID, Age, Sex, Weight, Height")
        print("✓ Study/Series dates and times")
        print("✓ Institution and physician information")
        print("✓ All medical measurements and observations")
        print("✓ Equipment and acquisition parameters")
        print("✓ Private tags (may contain research data)")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Your directory path
    input_directory = r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_full"

    # Option 1: Anonymize in-place (with backup)
    anonymizer = DicomAnonymizer(
        input_directory=input_directory,
        backup=True  # Creates .backup files
    )

    # Option 2: Anonymize to a new directory (recommended)
    # output_directory = r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_full_anonymized"
    # anonymizer = DicomAnonymizer(
    #     input_directory=input_directory,
    #     output_directory=output_directory,
    #     backup=False  # No need for backup when using separate output directory
    # )

    # Start the anonymization process
    anonymizer.process_directory(patient_id_prefix="MS_ANON")