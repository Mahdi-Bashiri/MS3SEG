#!/usr/bin/env python3
"""
DICOM File Renamer - Patient Folder Organizer (No Extension Version)
==================================================================

This script analyzes DICOM files in patient folders and renames them based on
metadata from their headers using the format:
{PatientID}_{ProtocolName}_{SeriesNumber}_{InstanceNumber}

Features:
- Analyzes DICOM headers for metadata extraction
- Handles missing or invalid DICOM tags gracefully
- Prevents filename conflicts with automatic numbering
- Creates detailed reports of renaming operations
- Supports dry-run mode for testing
- Multi-threaded processing for large datasets
- Preserves original non-formatted file nature (no extensions)

Author: AI Assistant
Date: September 2025
Version: 1.1
"""

import os
import sys
import shutil
from pathlib import Path
import pydicom
from pydicom.errors import InvalidDicomError
import logging
import time
import re
from collections import defaultdict, Counter
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Setup comprehensive logging with Windows compatibility"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('logs/dicom_renaming.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce pydicom logging noise
    logging.getLogger('pydicom').setLevel(logging.WARNING)

    return logging.getLogger(__name__)


# ============================================================================
# DICOM FILE RENAMER CLASS
# ============================================================================

class DICOMFileRenamer:
    """
    DICOM File Renamer for Patient Folders

    Analyzes DICOM files and renames them based on header metadata:
    Format: {PatientID}_{ProtocolName}_{SeriesNumber}_{InstanceNumber}
    Note: No file extensions are added - files maintain their original non-formatted nature
    """

    def __init__(self, patient_folders_dir, dry_run=False, num_threads=4):
        """
        Initialize the DICOM File Renamer

        Args:
            patient_folders_dir (str): Directory containing patient folders
            dry_run (bool): If True, only analyze without renaming
            num_threads (int): Number of threads for processing
        """
        self.patient_folders_dir = Path(patient_folders_dir)
        self.dry_run = dry_run
        self.num_threads = num_threads

        # Statistics tracking
        self.total_files = 0
        self.processed_files = 0
        self.renamed_files = 0
        self.error_files = 0
        self.skipped_files = 0

        # Analysis results
        self.patient_analysis = {}
        self.protocol_stats = Counter()
        self.series_stats = Counter()
        self.error_log = []

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Validate directory
        if not self.patient_folders_dir.exists():
            raise FileNotFoundError(f"Patient folders directory not found: {self.patient_folders_dir}")

        self.logger.info(f"Initialized DICOM renamer for: {self.patient_folders_dir}")
        self.logger.info(f"Dry run mode: {self.dry_run}")
        self.logger.info("Files will be renamed WITHOUT extensions (preserving non-formatted nature)")

    def _clean_filename_component(self, text, max_length=30):
        """Clean text for use in filename"""
        if not text:
            return "Unknown"

        # Convert to string and clean
        text = str(text).strip()

        # Remove or replace problematic characters
        # Keep alphanumeric, hyphens, underscores, and spaces
        text = re.sub(r'[^\w\s\-_.]', '', text)

        # Replace spaces and multiple underscores/hyphens with single underscore
        text = re.sub(r'[\s_-]+', '_', text)

        # Remove leading/trailing underscores
        text = text.strip('_')

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length].rstrip('_')

        # Ensure not empty
        if not text:
            return "Unknown"

        return text

    def _extract_dicom_metadata(self, file_path):
        """
        Extract relevant metadata from DICOM file

        Returns:
            dict: Metadata dictionary with keys: patient_id, protocol_name,
                  series_number, instance_number, series_description
        """
        try:
            # Read DICOM file
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)

            # Extract metadata with fallbacks
            metadata = {
                'patient_id': None,
                'protocol_name': None,
                'series_number': None,
                'instance_number': None,
                'series_description': None,
                'study_description': None,
                'modality': None
            }

            # Patient ID
            if hasattr(ds, 'PatientID') and ds.PatientID:
                metadata['patient_id'] = str(ds.PatientID).strip()

            # Protocol Name (can be in different tags)
            protocol_candidates = []
            if hasattr(ds, 'ProtocolName') and ds.ProtocolName:
                protocol_candidates.append(str(ds.ProtocolName).strip())
            if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription:
                protocol_candidates.append(str(ds.SeriesDescription).strip())
            if hasattr(ds, 'StudyDescription') and ds.StudyDescription:
                protocol_candidates.append(str(ds.StudyDescription).strip())

            # Choose the most descriptive protocol name
            if protocol_candidates:
                # Prefer longer, more descriptive names
                metadata['protocol_name'] = max(protocol_candidates, key=len)

            # Series Number
            if hasattr(ds, 'SeriesNumber') and ds.SeriesNumber is not None:
                metadata['series_number'] = int(ds.SeriesNumber)

            # Instance Number (slice number)
            if hasattr(ds, 'InstanceNumber') and ds.InstanceNumber is not None:
                metadata['instance_number'] = int(ds.InstanceNumber)

            # Additional useful metadata
            if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription:
                metadata['series_description'] = str(ds.SeriesDescription).strip()
            if hasattr(ds, 'StudyDescription') and ds.StudyDescription:
                metadata['study_description'] = str(ds.StudyDescription).strip()
            if hasattr(ds, 'Modality') and ds.Modality:
                metadata['modality'] = str(ds.Modality).strip()

            return metadata

        except Exception as e:
            self.logger.debug(f"Error reading DICOM metadata from {file_path}: {e}")
            return None

    def _generate_new_filename(self, metadata, original_path):
        """
        Generate new filename based on metadata

        Format: {PatientID}_{ProtocolName}_{SeriesNumber}_{InstanceNumber}
        Note: NO extension is added to preserve non-formatted nature
        """
        # Extract and clean components
        patient_id = metadata.get('patient_id', 'UnknownID')
        if patient_id and len(patient_id) >= 6:
            # Extract 6-digit patient ID if available
            digits = ''.join(filter(str.isdigit, patient_id))
            if len(digits) >= 6:
                patient_id = digits[:6]

        protocol = self._clean_filename_component(
            metadata.get('protocol_name'), max_length=25
        )

        series_num = metadata.get('series_number')
        series_str = f"S{series_num:03d}" if series_num is not None else "S000"

        instance_num = metadata.get('instance_number')
        instance_str = f"I{instance_num:04d}" if instance_num is not None else "I0000"

        # Generate filename WITHOUT extension to preserve non-formatted nature
        new_filename = f"{patient_id}_{protocol}_{series_str}_{instance_str}"

        return new_filename

    def _process_single_file(self, file_path, patient_folder):
        """Process a single DICOM file"""
        try:
            self.processed_files += 1

            # Skip files that already appear to be in the correct format
            if self._is_already_renamed(file_path.name):
                self.skipped_files += 1
                return {
                    'original_filename': file_path.name,
                    'new_filename': file_path.name,
                    'metadata': {},
                    'renamed': False,
                    'reason': 'Already in correct format'
                }

            # Extract metadata
            metadata = self._extract_dicom_metadata(file_path)

            if not metadata:
                self.error_files += 1
                self.error_log.append({
                    'file': str(file_path),
                    'error': 'Could not read DICOM metadata',
                    'patient': patient_folder.name
                })
                return None

            # Generate new filename
            new_filename = self._generate_new_filename(metadata, file_path)
            new_path = patient_folder / new_filename

            # Handle filename conflicts
            counter = 1
            while new_path.exists() and new_path != file_path:
                new_filename = f"{new_filename}_v{counter:02d}"
                new_path = patient_folder / new_filename
                counter += 1

            # Store analysis data
            analysis_data = {
                'original_filename': file_path.name,
                'new_filename': new_filename,
                'metadata': metadata,
                'renamed': False
            }

            # Perform renaming if not dry run and filename changed
            if not self.dry_run and new_path != file_path:
                try:
                    shutil.move(str(file_path), str(new_path))
                    analysis_data['renamed'] = True
                    self.renamed_files += 1

                    self.logger.debug(f"Renamed: {file_path.name} -> {new_filename}")

                except Exception as e:
                    self.error_files += 1
                    self.error_log.append({
                        'file': str(file_path),
                        'error': f'Rename failed: {e}',
                        'patient': patient_folder.name
                    })
                    analysis_data['renamed'] = False

            elif new_path == file_path:
                self.skipped_files += 1
                analysis_data['renamed'] = False
                analysis_data['reason'] = 'Already correctly named'

            # Update statistics
            if metadata.get('protocol_name'):
                self.protocol_stats[metadata['protocol_name']] += 1
            if metadata.get('series_number') is not None:
                self.series_stats[f"Series_{metadata['series_number']}"] += 1

            return analysis_data

        except Exception as e:
            self.error_files += 1
            self.error_log.append({
                'file': str(file_path),
                'error': str(e),
                'patient': patient_folder.name
            })
            return None

    def _is_already_renamed(self, filename):
        """Check if filename already follows the expected pattern"""
        # Pattern: digits_text_Snumber_Inumber
        pattern = r'^\d+_.+_S\d{3}_I\d{4}(_v\d{2})?$'
        return bool(re.match(pattern, filename))

    def _process_patient_folder(self, patient_folder):
        """Process all DICOM files in a patient folder"""
        self.logger.info(f"Processing patient folder: {patient_folder.name}")

        # Find all files (DICOM files typically have no extension)
        all_files = []
        for file_path in patient_folder.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                all_files.append(file_path)

        if not all_files:
            self.logger.warning(f"No files found in {patient_folder.name}")
            return

        self.total_files += len(all_files)
        self.logger.info(f"Found {len(all_files)} files in {patient_folder.name}")

        # Process files with progress bar
        patient_analysis = []

        with tqdm(
                total=len(all_files),
                desc=f"Patient {patient_folder.name}",
                unit="files",
                leave=False
        ) as pbar:

            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, file_path, patient_folder): file_path
                    for file_path in all_files
                }

                for future in as_completed(future_to_file):
                    result = future.result()
                    if result:
                        patient_analysis.append(result)
                    pbar.update(1)

        # Store patient analysis
        self.patient_analysis[patient_folder.name] = {
            'total_files': len(all_files),
            'processed_files': len(patient_analysis),
            'renamed_files': sum(1 for r in patient_analysis if r.get('renamed', False)),
            'files': patient_analysis
        }

        self.logger.info(f"Completed {patient_folder.name}: "
                         f"{len(patient_analysis)} processed, "
                         f"{sum(1 for r in patient_analysis if r.get('renamed', False))} renamed")

    def process_all_patients(self):
        """Process all patient folders"""
        start_time = time.time()

        print("\n" + "=" * 70)
        print("DICOM FILE RENAMER - PRESERVING NON-FORMATTED NATURE")
        print("=" * 70)
        print(f"Source Directory: {self.patient_folders_dir}")
        print(f"Mode: {'DRY RUN (Analysis Only)' if self.dry_run else 'RENAME FILES'}")
        print(f"Threads: {self.num_threads}")
        print("Note: Files will be renamed WITHOUT extensions")
        print("=" * 70)

        # Find patient folders (directories with numeric names)
        patient_folders = []
        for folder in self.patient_folders_dir.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                patient_folders.append(folder)

        if not patient_folders:
            print("No patient folders found (looking for numeric folder names)")
            return

        patient_folders.sort(key=lambda x: x.name)
        print(f"Found {len(patient_folders)} patient folders")
        print("=" * 70)

        # Process each patient folder
        with tqdm(total=len(patient_folders), desc="Overall Progress", unit="patients") as overall_pbar:
            for patient_folder in patient_folders:
                try:
                    self._process_patient_folder(patient_folder)
                except Exception as e:
                    self.logger.error(f"Error processing {patient_folder.name}: {e}")
                    self.logger.debug(traceback.format_exc())
                finally:
                    overall_pbar.update(1)

        # Generate final report
        self._generate_final_report(time.time() - start_time)

    def _generate_final_report(self, total_time):
        """Generate comprehensive final report"""
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETED")
        print("=" * 70)
        print(f"Total Time: {total_time / 60:.1f} minutes")
        print(f"Total Files: {self.total_files:,}")
        print(f"Processed Files: {self.processed_files:,}")
        print(f"Renamed Files: {self.renamed_files:,}")
        print(f"Skipped Files: {self.skipped_files:,}")
        print(f"Error Files: {self.error_files:,}")
        print(f"Processing Rate: {self.processed_files / total_time:.1f} files/second")
        print("=" * 70)

        # Patient summary
        if self.patient_analysis:
            print(f"PATIENT SUMMARY ({len(self.patient_analysis)} patients):")
            print("-" * 50)
            for patient_id, analysis in sorted(self.patient_analysis.items()):
                print(f"  Patient {patient_id}: "
                      f"{analysis['total_files']} files, "
                      f"{analysis['renamed_files']} renamed")

        # Protocol statistics
        if self.protocol_stats:
            print("\nTOP PROTOCOLS/SEQUENCES:")
            print("-" * 30)
            for protocol, count in self.protocol_stats.most_common(10):
                print(f"  {protocol}: {count} files")

        # Series statistics
        if self.series_stats:
            print(f"\nSERIES DISTRIBUTION:")
            print("-" * 30)
            series_counts = len(self.series_stats)
            print(f"  Total unique series: {series_counts}")

        # Error summary
        if self.error_log:
            print(f"\nERRORS ({len(self.error_log)} total):")
            print("-" * 30)
            error_types = Counter(error['error'] for error in self.error_log)
            for error_type, count in error_types.most_common(5):
                print(f"  {error_type}: {count} files")

        print("=" * 70)
        print("NOTE: All renamed files maintain their original non-formatted nature")
        print("(no file extensions added)")
        print("=" * 70)

        # Save detailed report to file
        self._save_detailed_report()

    def _save_detailed_report(self):
        """Save detailed analysis report to JSON file"""
        report_data = {
            'summary': {
                'total_files': self.total_files,
                'processed_files': self.processed_files,
                'renamed_files': self.renamed_files,
                'skipped_files': self.skipped_files,
                'error_files': self.error_files,
                'dry_run': self.dry_run,
                'preserve_non_formatted': True
            },
            'patient_analysis': self.patient_analysis,
            'protocol_stats': dict(self.protocol_stats),
            'series_stats': dict(self.series_stats),
            'error_log': self.error_log
        }

        # Save to JSON file
        report_file = Path(f"dicom_renaming_report_{int(time.time())}.json")
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"Detailed report saved to: {report_file}")
        except Exception as e:
            self.logger.error(f"Could not save detailed report: {e}")


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def preview_renaming(patient_folders_dir, num_samples=5):
    """Preview renaming for a few files from each patient"""
    print("\n" + "=" * 60)
    print("RENAMING PREVIEW (NO EXTENSIONS)")
    print("=" * 60)

    renamer = DICOMFileRenamer(patient_folders_dir, dry_run=True, num_threads=2)

    patient_folders = [f for f in Path(patient_folders_dir).iterdir()
                       if f.is_dir() and f.name.isdigit()]

    for patient_folder in patient_folders[:3]:  # Preview first 3 patients
        print(f"\nPatient {patient_folder.name}:")
        print("-" * 30)

        files = [f for f in patient_folder.iterdir() if f.is_file()][:num_samples]

        for file_path in files:
            # Skip if already renamed
            if renamer._is_already_renamed(file_path.name):
                print(f"  {file_path.name} -> ALREADY RENAMED (skipped)")
                print()
                continue

            metadata = renamer._extract_dicom_metadata(file_path)
            if metadata:
                new_filename = renamer._generate_new_filename(metadata, file_path)
                print(f"  {file_path.name}")
                print(f"  -> {new_filename}")
                print(f"     Protocol: {metadata.get('protocol_name', 'N/A')}")
                print(f"     Series: {metadata.get('series_number', 'N/A')}")
                print(f"     Instance: {metadata.get('instance_number', 'N/A')}")
                print()
            else:
                print(f"  {file_path.name} -> ERROR: Cannot read DICOM")
                print()


def main():
    """Main execution function"""

    # ========================================================================
    # CONFIGURATION - MODIFY THIS PATH
    # ========================================================================

    PATIENT_FOLDERS_DIR = r"E:\MBashiri\Data\Dr.Shakeri.Gholghasht\subjects_folders_100"

    # ========================================================================
    # SETTINGS
    # ========================================================================

    NUM_THREADS = 4  # Adjust based on your system

    # Setup logging
    logger = setup_logging()

    print("DICOM FILE RENAMER - NON-FORMATTED VERSION")
    print("=" * 50)
    print("This script will rename DICOM files based on their metadata:")
    print("Format: {PatientID}_{ProtocolName}_{SeriesNumber}_{InstanceNumber}")
    print("Note: Files will maintain their original non-formatted nature (no extensions)")
    print()
    print(f"Patient Folders Directory: {PATIENT_FOLDERS_DIR}")
    print(f"Processing Threads: {NUM_THREADS}")
    print("=" * 50)

    # Menu options
    print("\nOptions:")
    print("1. Preview renaming (analyze a few files)")
    print("2. Dry run (analyze all files, no renaming)")
    print("3. Rename files (actual renaming)")
    print("4. Exit")

    try:
        choice = input("\nChoose option (1-4): ").strip()

        if choice == "1":
            preview_renaming(PATIENT_FOLDERS_DIR)

        elif choice == "2":
            print("\nStarting DRY RUN analysis...")
            renamer = DICOMFileRenamer(PATIENT_FOLDERS_DIR, dry_run=True, num_threads=NUM_THREADS)
            renamer.process_all_patients()

        elif choice == "3":
            print("\nWARNING: This will rename files permanently!")
            print("Files will be renamed WITHOUT extensions to preserve their non-formatted nature.")
            confirm = input("Are you sure you want to proceed? (type 'YES' to confirm): ")
            if confirm == "YES":
                print("\nStarting file renaming...")
                renamer = DICOMFileRenamer(PATIENT_FOLDERS_DIR, dry_run=False, num_threads=NUM_THREADS)
                renamer.process_all_patients()
            else:
                print("Operation cancelled.")

        elif choice == "4":
            print("Goodbye!")
            return

        else:
            print("Invalid choice. Please run the script again.")

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")

    except FileNotFoundError as e:
        print(f"\nDirectory Error: {e}")
        print("Please check that the patient folders directory exists and is accessible.")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nAn unexpected error occurred. Check logs/dicom_renaming.log for details.")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()