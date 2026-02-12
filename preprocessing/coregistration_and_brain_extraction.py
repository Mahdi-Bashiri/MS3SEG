#!/usr/bin/env python3
"""
Parallel MRI Registration and Brain Extraction Pipeline
======================================================

This script processes brain MRI data in parallel by:
1. Registering T1WI and T2WI images to FLAIR space using FSL FLIRT
2. Performing brain extraction on FLAIR images using FSL BET
3. Creating binary brain masks

Features:
- Safe parallel processing with isolated temporary directories
- Memory and resource monitoring
- Comprehensive error handling
- Progress tracking across parallel workers

Author: Mahdi Bashiri Bawil
Date: September 2025
"""

import os
import subprocess
import shutil
import tempfile
import psutil
import numpy as np
from pathlib import Path
from time import monotonic, sleep
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

def log_message(message, patient_id=None, start_time=None):
    """Thread-safe logging function."""
    timestamp = f"[{monotonic() - start_time:.1f}s]" if start_time else ""
    prefix = f"[{patient_id}]" if patient_id else "[MAIN]"
    full_message = f"{timestamp} {prefix} {message}"
    print(full_message)
    sys.stdout.flush()

def get_system_resources():
    """Get current system resource usage."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    return {
        'memory_used_gb': memory.used / (1024**3),
        'memory_available_gb': memory.available / (1024**3),
        'memory_percent': memory.percent,
        'cpu_percent': cpu_percent
    }

def find_patient_files(patient_folder):
    """Find required MRI files for a patient."""
    files = {}
    patient_folder = Path(patient_folder)
    
    # Look for required files
    flair_file = list(patient_folder.glob("*_FLAIR.nii.gz"))
    t1wi_file = list(patient_folder.glob("*_T1WI.nii.gz"))
    t2wi_file = list(patient_folder.glob("*_T2WI.nii.gz"))
    
    # Filter out FLAIR_SG files
    flair_file = [f for f in flair_file if "_FLAIR_SG" not in f.name]
    
    if flair_file:
        files['FLAIR'] = flair_file[0]
    if t1wi_file:
        files['T1WI'] = t1wi_file[0]
    if t2wi_file:
        files['T2WI'] = t2wi_file[0]
        
    return files

def safe_fsl_command(command, temp_dir, patient_id, description, start_time):
    """
    Execute FSL command with safety measures.
    
    Parameters:
    -----------
    command : list
        FSL command as list of strings
    temp_dir : str
        Unique temporary directory for this process
    patient_id : str
        Patient identifier for logging
    description : str
        Description of the operation for logging
    start_time : float
        Start time for logging
    """
    
    # Create safe environment with isolated temp directory
    env = os.environ.copy()
    env['TMPDIR'] = temp_dir
    env['TEMP'] = temp_dir
    env['TMP'] = temp_dir
    
    # Set FSL-specific environment variables to avoid conflicts
    env['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    
    log_message(f"Starting: {description}", patient_id, start_time)
    
    try:
        # Run command with timeout to prevent hanging
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            env=env,
            timeout=600  # 10-minute timeout per operation
        )
        
        log_message(f"✓ Completed: {description}", patient_id, start_time)
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        log_message(f"✗ Timeout: {description}", patient_id, start_time)
        return False, "Command timed out after 10 minutes"
    except subprocess.CalledProcessError as e:
        log_message(f"✗ Error in {description}: {e}", patient_id, start_time)
        return False, e.stderr
    except Exception as e:
        log_message(f"✗ Unexpected error in {description}: {str(e)}", patient_id, start_time)
        return False, str(e)

def robust_t2wi_registration(files, t2wi_output, temp_dir, patient_id, fsl_path, start_time):
    """
    Robust T2WI registration with multiple fallback strategies.
    """
    
    # Strategy 1: Original approach with mutual information
    strategies = [
        {
            'name': 'Default_Mode',
            'params': [
                f'{fsl_path}/flirt',
                '-in', str(files['T2WI']),
                '-ref', str(files['FLAIR']),
                '-out', str(t2wi_output),
                '-searchrx', '-180', '180',
                '-searchry', '-180', '180'
            ]
        },
        {
            'name': 'Mutual_Info_Wide_Search',
            'params': [
                f'{fsl_path}/flirt',
                '-in', str(files['T2WI']),
                '-ref', str(files['FLAIR']),
                '-out', str(t2wi_output),
                '-cost', 'mutualinfo',
                '-searchrx', '-180', '180',
                '-searchry', '-180', '180',
                '-searchrz', '-180', '180',
                '-dof', '12'
            ]
        },
        {
            'name': 'Normalized_Correlation',
            'params': [
                f'{fsl_path}/flirt',
                '-in', str(files['T2WI']),
                '-ref', str(files['FLAIR']),
                '-out', str(t2wi_output),
                '-cost', 'normcorr',
                '-searchrx', '-180', '180',
                '-searchry', '-180', '180',
                '-dof', '12'
            ]
        },
        {
            'name': 'Two_Stage_Registration',
            'params': [
                f'{fsl_path}/flirt',
                '-in', str(files['T2WI']),
                '-ref', str(files['FLAIR']),
                '-out', str(t2wi_output),
                '-cost', 'mutualinfo',
                '-coarsesearch', '18',
                '-finesearch', '6',
                '-searchrx', '-180', '180',
                '-searchry', '-180', '180',
                '-dof', '12'
            ]
        },
        {
            'name': 'Rigid_Body_Only',
            'params': [
                f'{fsl_path}/flirt',
                '-in', str(files['T2WI']),
                '-ref', str(files['FLAIR']),
                '-out', str(t2wi_output),
                '-cost', 'mutualinfo',
                '-dof', '6',
                '-searchrx', '-180', '180',
                '-searchry', '-180', '180'
            ]
        },
        {
            'name': 'Least_Squares_Robust',
            'params': [
                f'{fsl_path}/flirt',
                '-in', str(files['T2WI']),
                '-ref', str(files['FLAIR']),
                '-out', str(t2wi_output),
                '-cost', 'leastsq',
                '-searchrx', '-180', '180',
                '-searchry', '-180', '180',
                '-dof', '9'  # Affine without shears
            ]
        }
    ]
    
    for i, strategy in enumerate(strategies):
        log_message(f"Attempting strategy {i+1}/{len(strategies)}: {strategy['name']}", patient_id, start_time)
        
        success, msg = safe_fsl_command(
            strategy['params'], temp_dir, patient_id, 
            f"T2WI registration - {strategy['name']}", start_time
        )
        
        if success:
            # Verify the output is not empty
            if t2wi_output.exists():
                try:
                    import nibabel as nib
                    img = nib.load(str(t2wi_output))
                    data = img.get_fdata()
                    non_zero_count = np.count_nonzero(data)
                    
                    if non_zero_count > 3000:  # Reasonable threshold
                        log_message(f"✅ SUCCESS with {strategy['name']} ({non_zero_count} non-zero voxels)", patient_id, start_time)
                        return True, f"Success with {strategy['name']}"
                    else:
                        log_message(f"⚠️  {strategy['name']} produced mostly empty output ({non_zero_count} voxels)", patient_id, start_time)
                        # Try next strategy
                        continue
                except Exception as e:
                    log_message(f"⚠️  Cannot verify {strategy['name']} output: {str(e)}", patient_id, start_time)
                    continue
            else:
                log_message(f"⚠️  {strategy['name']} did not create output file", patient_id, start_time)
                continue
        else:
            log_message(f"❌ {strategy['name']} failed: {msg}", patient_id, start_time)
            continue
    
    return False, "All registration strategies failed"

def process_single_patient(patient_data):
    """
    Process a single patient's data with full safety isolation.
    
    This function is designed to be called by parallel workers.
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient folder path, output directory, fsl_path, and start_time
    """
    
    patient_folder = Path(patient_data['patient_folder'])
    output_dir = Path(patient_data['output_dir'])
    fsl_path = patient_data['fsl_path']
    start_time = patient_data['start_time']
    
    patient_id = patient_folder.name
    patient_start_time = monotonic()
    
    # Create isolated temporary directory for this worker
    with tempfile.TemporaryDirectory(prefix=f'fsl_{patient_id}_') as temp_dir:
        
        log_message("Starting patient processing", patient_id, start_time)
        
        try:
            # Create output folder for this patient
            patient_output_dir = output_dir / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find patient files
            files = find_patient_files(patient_folder)
            
            # Check if all required files exist
            required_files = ['FLAIR', 'T1WI', 'T2WI']
            missing_files = [f for f in required_files if f not in files]
            
            if missing_files:
                error_msg = f"Missing files: {missing_files}"
                log_message(f"✗ {error_msg}", patient_id, start_time)
                return False, patient_id, error_msg
            
            # Define output paths
            flair_output = patient_output_dir / f"{patient_id}_FLAIR.nii.gz"
            t1wi_output = patient_output_dir / f"{patient_id}_T1WI_reg.nii.gz"
            t2wi_output = patient_output_dir / f"{patient_id}_T2WI_reg.nii.gz"
            brain_extracted_output = patient_output_dir / f"{patient_id}_FLAIR_brain.nii.gz"
            brain_mask_output = patient_output_dir / f"{patient_id}_brain_mask.nii.gz"
            
            # Step 1: Copy FLAIR to output (it's our reference)
            log_message("Copying FLAIR image", patient_id, start_time)
            shutil.copy2(files['FLAIR'], flair_output)
            
            # Step 2: Register T1WI to FLAIR
            t1_command = [
                f'{fsl_path}/flirt',
                '-in', str(files['T1WI']),
                '-ref', str(files['FLAIR']),
                '-out', str(t1wi_output),
                '-searchrx', '-180', '180',
                '-searchry', '-180', '180'
            ]
            
            success_t1, msg_t1 = safe_fsl_command(
                t1_command, temp_dir, patient_id, "T1WI registration", start_time
            )
            
            if not success_t1:
                return False, patient_id, f"T1WI registration failed: {msg_t1}"
            
            # # Step 3: Register T2WI to FLAIR
            # t2_command = [
            #     f'{fsl_path}/flirt',
            #     '-in', str(files['T2WI']),
            #     '-ref', str(files['FLAIR']),
            #     '-out', str(t2wi_output),
            #     '-searchrx', '-180', '180',
            #     '-searchry', '-180', '180'
            # ]
            
            # success_t2, msg_t2 = safe_fsl_command(
            #     t2_command, temp_dir, patient_id, "T2WI registration", start_time
            # )
            
            # if not success_t2:
            #     return False, patient_id, f"T2WI registration failed: {msg_t2}"
            
            # Step 3: Register T2WI to FLAIR (ROBUST VERSION)
            success_t2, msg_t2 = robust_t2wi_registration(
                files, t2wi_output, temp_dir, patient_id, fsl_path, start_time
            )

            if not success_t2:
                return False, patient_id, f"T2WI registration failed: {msg_t2}"
            
            # Step 4: Brain extraction on FLAIR
            bet_command = [
                f'{fsl_path}/bet',
                str(flair_output),
                str(brain_extracted_output).replace('.nii.gz', ''),  # BET adds .nii.gz
                '-f', '0.5',
                '-m'  # Generate mask
            ]
            
            success_bet, msg_bet = safe_fsl_command(
                bet_command, temp_dir, patient_id, "Brain extraction", start_time
            )
            
            if not success_bet:
                return False, patient_id, f"Brain extraction failed: {msg_bet}"
            
            # Move the generated mask to the desired location
            auto_mask_path = str(brain_extracted_output).replace('.nii.gz', '_mask.nii.gz')
            if os.path.exists(auto_mask_path):
                shutil.move(auto_mask_path, str(brain_mask_output))
                log_message("Brain mask created", patient_id, start_time)
            
            # Calculate processing time for this patient
            patient_time = monotonic() - patient_start_time
            
            log_message(f"✓ Patient completed in {patient_time:.1f}s", patient_id, start_time)
            
            return True, patient_id, f"Success in {patient_time:.1f}s"
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            log_message(f"✗ {error_msg}", patient_id, start_time)
            return False, patient_id, error_msg

class ParallelMRIProcessor:
    """
    A class to handle parallel MRI registration and brain extraction using FSL tools.
    """
    
    def __init__(self, input_dir, output_dir, fsl_path='/usr/local/fsl/bin', max_workers=4):
        """
        Initialize the parallel MRI processor.
        
        Parameters:
        -----------
        input_dir : str
            Path to input directory containing patient folders
        output_dir : str
            Path to output directory for processed data
        fsl_path : str
            Path to FSL binaries
        max_workers : int
            Maximum number of parallel workers (default: 4)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.fsl_path = fsl_path
        self.max_workers = max_workers
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance monitoring
        self.start_time = None
        self.processed_count = 0
        self.failed_patients = []
    
    def monitor_resources(self):
        """Monitor and log system resources."""
        resources = get_system_resources()
        log_message(
            f"Resources - RAM: {resources['memory_used_gb']:.1f}GB/{resources['memory_available_gb']:.1f}GB "
            f"({resources['memory_percent']:.1f}%), CPU: {resources['cpu_percent']:.1f}%",
            start_time=self.start_time
        )
    
    def get_patient_folders(self):
        """Get all patient folders from input directory."""
        patient_folders = [d for d in self.input_dir.iterdir() if d.is_dir()]
        return sorted(patient_folders)
        
    def process_all_patients_parallel(self):
        """Process all patients using parallel workers."""
        
        log_message("=== Parallel MRI Processing Pipeline Started ===")
        log_message(f"Input directory: {self.input_dir}")
        log_message(f"Output directory: {self.output_dir}")
        log_message(f"FSL path: {self.fsl_path}")
        log_message(f"Max workers: {self.max_workers}")
        
        # Initial resource check
        log_message("Initial system resources:")
        self.monitor_resources()
        
        # Get all patient folders
        patient_folders = self.get_patient_folders()
        
        if not patient_folders:
            log_message("No patient folders found in input directory!")
            return
        
        log_message(f"Found {len(patient_folders)} patients to process")
        
        # Start timing
        self.start_time = monotonic()
        
        # Process patients in parallel
        successful_patients = 0
        failed_patients = []
        
        # Prepare patient data for parallel processing
        patient_data_list = []
        for folder in patient_folders:
            patient_data = {
                'patient_folder': str(folder),
                'output_dir': str(self.output_dir),
                'fsl_path': self.fsl_path,
                'start_time': self.start_time
            }
            patient_data_list.append(patient_data)
        
        # Use ProcessPoolExecutor for CPU-bound FSL operations
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Submit all jobs
            future_to_patient = {
                executor.submit(process_single_patient, patient_data): patient_data['patient_folder']
                for patient_data in patient_data_list
            }
            
            log_message(f"Submitted {len(future_to_patient)} jobs to {self.max_workers} workers", start_time=self.start_time)
            
            # Process completed jobs
            for future in as_completed(future_to_patient):
                patient_folder = future_to_patient[future]
                patient_name = Path(patient_folder).name
                
                try:
                    success, patient_id, message = future.result()
                    
                    if success:
                        successful_patients += 1
                        log_message(f"SUCCESS: {message}", start_time=self.start_time)
                    else:
                        failed_patients.append((patient_id, message))
                        log_message(f"FAILED: {patient_id} - {message}", start_time=self.start_time)
                    
                    # Progress update
                    completed = successful_patients + len(failed_patients)
                    progress = (completed / len(patient_folders)) * 100
                    
                    log_message(f"Progress: {completed}/{len(patient_folders)} ({progress:.1f}%)", start_time=self.start_time)
                    
                    # Resource monitoring every 5 patients
                    if completed % 5 == 0:
                        self.monitor_resources()
                        
                except Exception as e:
                    failed_patients.append((patient_name, f"Job execution error: {str(e)}"))
                    log_message(f"FAILED: {patient_name} - Job execution error: {str(e)}", start_time=self.start_time)
        
        # Final summary
        total_time = monotonic() - self.start_time
        
        log_message("=" * 60, start_time=self.start_time)
        log_message("FINAL SUMMARY", start_time=self.start_time)
        log_message("=" * 60, start_time=self.start_time)
        log_message(f"Total patients: {len(patient_folders)}", start_time=self.start_time)
        log_message(f"Successful: {successful_patients}", start_time=self.start_time)
        log_message(f"Failed: {len(failed_patients)}", start_time=self.start_time)
        log_message(f"Success rate: {(successful_patients/len(patient_folders)*100):.1f}%", start_time=self.start_time)
        log_message(f"Total time: {total_time/60:.1f} minutes", start_time=self.start_time)
        log_message(f"Average time per patient: {total_time/len(patient_folders):.1f} seconds", start_time=self.start_time)
        
        if successful_patients > 0:
            speedup = len(patient_folders) / (total_time / 60)  # patients per minute
            log_message(f"Processing speed: {speedup:.1f} patients/minute", start_time=self.start_time)
        
        if failed_patients:
            log_message("\nFailed patients:", start_time=self.start_time)
            for patient_id, reason in failed_patients:
                log_message(f"  {patient_id}: {reason}", start_time=self.start_time)
        
        # Final resource check
        log_message("\nFinal system resources:", start_time=self.start_time)
        self.monitor_resources()
        
        log_message(f"\nOutput saved to: {self.output_dir}", start_time=self.start_time)
        log_message("=== Processing Complete ===", start_time=self.start_time)


def main():
    """Main function to run the parallel processing pipeline."""
    
    # Configuration - ADJUST THESE PATHS AS NEEDED
    input_directory = '/mnt/e/MBashiri/Thesis/p6/Data/MS_100_patient_nifti'
    output_directory = '/mnt/e/MBashiri/Thesis/p6/Data/MS_100_patient_registered'
    
    # FSL path - adjust based on your installation
    fsl_binary_path = '/home/sai/fsl/bin'  # Update this path
    
    # Parallel processing settings
    # Start with 4 workers, can increase to 6-8 based on your system performance
    max_parallel_workers = 4
    
    # System resource check before starting
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    print(f"System Check:")
    print(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"  CPU cores: {cpu_count}")
    print(f"  Planned workers: {max_parallel_workers}")
    
    # Safety check - ensure enough memory
    estimated_memory_per_worker = 4  # GB per FSL process
    required_memory = max_parallel_workers * estimated_memory_per_worker
    
    if memory.available / (1024**3) < required_memory:
        print(f"\nWARNING: May not have enough RAM!")
        print(f"Required: ~{required_memory} GB, Available: {memory.available / (1024**3):.1f} GB")
        print("Consider reducing max_parallel_workers")
        
        user_input = input("\nContinue anyway? (y/N): ").lower().strip()
        if user_input != 'y':
            print("Exiting...")
            return
    
    # Create processor and run
    processor = ParallelMRIProcessor(
        input_dir=input_directory,
        output_dir=output_directory,
        fsl_path=fsl_binary_path,
        max_workers=max_parallel_workers
    )
    
    # Process all patients in parallel
    processor.process_all_patients_parallel()


if __name__ == "__main__":
    main()