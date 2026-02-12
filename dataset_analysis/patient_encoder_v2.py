import os
import json
import re
import shutil
from pathlib import Path
from collections import OrderedDict


class PatientIDAnonymizer:
    def __init__(self, main_dirs):
        """
        Initialize the anonymizer with main directories to process.

        Args:
            main_dirs: List of directory paths to process
        """
        self.main_dirs = main_dirs
        self.id_mapping = OrderedDict()
        self.reverse_mapping = {}

    # ------------------------------------------------------------------ #
    #  ID MAPPING                                                          #
    # ------------------------------------------------------------------ #

    def create_id_mapping(self):
        """
        Create mapping dictionary from 6-digit IDs to 3-digit IDs (001-100)
        based on ascending order of the 6-digit IDs found in the directories.
        """
        six_digit_ids = set()

        for main_dir in self.main_dirs:
            if not os.path.exists(main_dir):
                print(f"Warning: Directory not found: {main_dir}")
                continue

            for item in os.listdir(main_dir):
                if re.match(r'^\d{6}$', item):
                    six_digit_ids.add(item)

        sorted_ids = sorted(six_digit_ids)

        for idx, six_digit_id in enumerate(sorted_ids, start=1):
            three_digit_id = f"{idx:03d}"
            self.id_mapping[six_digit_id] = three_digit_id
            self.reverse_mapping[three_digit_id] = six_digit_id

        print(f"Created mapping for {len(self.id_mapping)} patient IDs")
        return self.id_mapping

    def save_mapping(self, output_file="id_mapping.json"):
        """Save the ID mapping to a JSON file for reference."""
        with open(output_file, 'w') as f:
            json.dump(self.id_mapping, f, indent=2)
        print(f"Mapping saved to {output_file}")

    def _extract_six_digit_id(self, value: str) -> str | None:
        """
        Return the first 6-digit sequence found in *value* that exists in
        the mapping, or None if nothing matches.
        """
        # Try leading 6 chars first (covers "101228 ch" style PatientIDs)
        prefix = value[:6] if len(value) >= 6 else ""
        if prefix and prefix in self.id_mapping:
            return prefix

        # Fall back to any 6-digit run in the string
        for match in re.finditer(r'\d{6}', value):
            candidate = match.group()
            if candidate in self.id_mapping:
                return candidate

        return None

    # ------------------------------------------------------------------ #
    #  TEXT / FILENAME REPLACEMENT                                         #
    # ------------------------------------------------------------------ #

    def replace_ids_in_text(self, text):
        """Replace all known 6-digit IDs in text with their 3-digit counterparts."""
        for six_digit, three_digit in self.id_mapping.items():
            text = text.replace(six_digit, three_digit)
        return text

    # ------------------------------------------------------------------ #
    #  JSON FILE PROCESSING  (enhanced)                                    #
    # ------------------------------------------------------------------ #

    def process_json_file(self, json_path, output_path=None):
        """
        Process a metadata JSON file:
          1. Replace the 6-digit patient ID in the "PatientID" field value
             with the corresponding 3-digit anonymised code.
          2. Empty the "PatientBirthDate" field.
          3. Replace the 6-digit patient ID segment inside "OriginalPath"
             with the 3-digit code.
          4. Replace any remaining 6-digit IDs everywhere else in the file.

        Args:
            json_path:   Path to the input JSON file.
            output_path: Destination path (default: <stem>_anonymized.json).
        """
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- 1. Anonymise PatientID field --------------------------------
        raw_patient_id = data.get("PatientID", "")
        six_digit = self._extract_six_digit_id(str(raw_patient_id))
        if six_digit and six_digit in self.id_mapping:
            three_digit = self.id_mapping[six_digit]
            # Keep any suffix after the 6-digit code (e.g. " ch") if present,
            # but replace only the 6-digit portion.
            data["PatientID"] = str(raw_patient_id).replace(six_digit, three_digit)
            print(f"  PatientID:     '{raw_patient_id}' -> '{data['PatientID']}'")
        else:
            print(f"  PatientID:     '{raw_patient_id}' — no mapping found, left unchanged")

        # --- 2. Empty PatientBirthDate -----------------------------------
        if "PatientBirthDate" in data:
            old_bd = data["PatientBirthDate"]
            data["PatientBirthDate"] = ""
            print(f"  PatientBirthDate: '{old_bd}' -> ''")

        # --- 3. Anonymise OriginalPath -----------------------------------
        if "OriginalPath" in data and data["OriginalPath"]:
            old_path = data["OriginalPath"]
            data["OriginalPath"] = self.replace_ids_in_text(old_path)
            if data["OriginalPath"] != old_path:
                print(f"  OriginalPath:  '{old_path}' -> '{data['OriginalPath']}'")

        # --- 4. Catch-all: replace IDs in any remaining string fields ----
        data = self._replace_ids_in_dict(data)

        # --- Write output ------------------------------------------------
        if output_path is None:
            json_file = Path(json_path)
            output_path = json_file.parent / f"{json_file.stem}_anonymized{json_file.suffix}"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Anonymized JSON saved to: {output_path}")

    def _replace_ids_in_dict(self, obj):
        """
        Recursively walk a JSON-decoded object and replace 6-digit IDs in
        all string values.
        """
        if isinstance(obj, dict):
            return {k: self._replace_ids_in_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_ids_in_dict(item) for item in obj]
        elif isinstance(obj, str):
            return self.replace_ids_in_text(obj)
        else:
            return obj

    def process_all_json_files(self, directory, recursive=True, dry_run=False):
        """
        Find and process every *.json file in *directory*.

        Args:
            directory:  Root directory to search.
            recursive:  Walk subdirectories when True.
            dry_run:    Print what would be done without writing files.
        """
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            return

        pattern = "**/*.json" if recursive else "*.json"
        json_files = list(Path(directory).glob(pattern))

        print(f"\nFound {len(json_files)} JSON file(s) in {directory}")

        for json_path in json_files:
            print(f"\nProcessing: {json_path}")
            if dry_run:
                print("  [DRY RUN] Would anonymize this file.")
            else:
                anon_path = json_path.parent / f"{json_path.stem}_anonymized{json_path.suffix}"
                self.process_json_file(str(json_path), str(anon_path))

    # ------------------------------------------------------------------ #
    #  IMAGING FILE HEADER ANONYMIZATION                                   #
    # ------------------------------------------------------------------ #

    def anonymize_dicom_header(self, file_path, dry_run=False):
        """
        Open a DICOM file (*.dcm or no-extension), clear PatientBirthDate
        (tag 0010,0030) in the header, and save in-place (or report for
        dry-run).

        Args:
            file_path: Path to the DICOM file.
            dry_run:   If True, only report — do not write.
        """
        try:
            import pydicom
        except ImportError:
            print("  [ERROR] pydicom is not installed. Run: pip install pydicom")
            return False

        try:
            ds = pydicom.dcmread(str(file_path), force=True)
        except Exception as e:
            print(f"  [ERROR] Could not read DICOM file {file_path}: {e}")
            return False

        changed = False

        # PatientBirthDate — (0010, 0030)
        if hasattr(ds, 'PatientBirthDate') and ds.PatientBirthDate:
            old_val = ds.PatientBirthDate
            if dry_run:
                print(f"  [DRY RUN] Would clear PatientBirthDate ('{old_val}') in {file_path}")
            else:
                ds.PatientBirthDate = ""
                changed = True

        # PatientID — (0010, 0020)
        if hasattr(ds, 'PatientID') and ds.PatientID:
            old_pid = str(ds.PatientID)
            six_digit = self._extract_six_digit_id(old_pid)
            if six_digit and six_digit in self.id_mapping:
                new_pid = old_pid.replace(six_digit, self.id_mapping[six_digit])
                if dry_run:
                    print(f"  [DRY RUN] Would replace PatientID ('{old_pid}' -> '{new_pid}') in {file_path}")
                else:
                    ds.PatientID = new_pid
                    changed = True

        # Optionally also clear PatientName / PatientID if desired — kept
        # commented out so it mirrors the existing script's scope.
        # if hasattr(ds, 'PatientName'):  ds.PatientName = "ANON"
        # if hasattr(ds, 'PatientID'):    ds.PatientID   = ""

        if changed:
            try:
                ds.save_as(str(file_path))
                print(f"  ✓ Cleared PatientBirthDate in DICOM: {file_path}")
                return True
            except Exception as e:
                print(f"  ✗ Failed to save DICOM {file_path}: {e}")
                return False

        return True  # no change needed

    def anonymize_nifti_header(self, file_path, dry_run=False):
        """
        Open a NIfTI file (*.nii or *.nii.gz) and clear the 'db_name' field
        (which some converters populate with patient info) and the
        'descrip' field if it contains a 6-digit patient ID.

        Note: The NIfTI-1 standard does not have a dedicated birth-date
        field. Birth date is a DICOM concept; if the NIfTI was converted
        from DICOM the birth date is not stored in the NIfTI header itself.
        This function clears the free-text fields that *could* carry PII.

        Args:
            file_path: Path to the NIfTI file.
            dry_run:   If True, only report — do not write.
        """
        try:
            import nibabel as nib
        except ImportError:
            print("  [ERROR] nibabel is not installed. Run: pip install nibabel")
            return False

        try:
            img = nib.load(str(file_path))
        except Exception as e:
            print(f"  [ERROR] Could not read NIfTI file {file_path}: {e}")
            return False

        header = img.header
        changed = False

        # NIfTI-1 free-text fields that may carry PII
        pii_fields = ['db_name', 'descrip', 'aux_file']

        for field in pii_fields:
            try:
                raw = header[field]
                # nibabel returns bytes arrays for these fields
                current = bytes(raw).rstrip(b'\x00').decode('utf-8', errors='replace').strip()
                if not current:
                    continue

                # Check if field contains a known 6-digit ID or any digits worth clearing
                has_id = any(sid in current for sid in self.id_mapping)
                if has_id or field == 'db_name':
                    if dry_run:
                        print(f"  [DRY RUN] Would clear NIfTI header field '{field}' "
                              f"('{current}') in {file_path}")
                    else:
                        header[field] = b''
                        changed = True
            except (KeyError, Exception):
                pass

        if changed:
            try:
                nib.save(img, str(file_path))
                print(f"  ✓ Cleared PII fields in NIfTI header: {file_path}")
            except Exception as e:
                print(f"  ✗ Failed to save NIfTI {file_path}: {e}")
                return False

        return True

    def _is_dicom_file(self, file_path: Path) -> bool:
        """
        Heuristic: a file is DICOM if it has a .dcm extension OR has no
        extension and starts with the DICOM preamble / magic bytes.
        """
        if file_path.suffix.lower() == '.dcm':
            return True
        if file_path.suffix == '':
            try:
                with open(file_path, 'rb') as f:
                    preamble = f.read(132)
                # DICOM magic bytes at offset 128
                return preamble[128:132] == b'DICM'
            except Exception:
                pass
        return False

    def _is_nifti_file(self, file_path: Path) -> bool:
        """Return True for *.nii and *.nii.gz files."""
        name = file_path.name.lower()
        return name.endswith('.nii') or name.endswith('.nii.gz')

    def anonymize_imaging_headers_in_directory(self, directory, recursive=True,
                                                dry_run=False):
        """
        Walk *directory* and anonymize headers of every imaging file found
        (.dcm, no-extension DICOM, .nii, .nii.gz).

        Args:
            directory:  Root directory to search.
            recursive:  Walk subdirectories when True.
            dry_run:    Report only — do not modify files.
        """
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            return

        root_path = Path(directory)
        walk_iter = root_path.rglob('*') if recursive else root_path.glob('*')

        dicom_count = nifti_count = skip_count = 0

        print(f"\n{'=' * 60}")
        print(f"Anonymizing imaging headers in: {directory}")
        print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL EDIT'}")
        print(f"{'=' * 60}")

        for file_path in walk_iter:
            if not file_path.is_file():
                continue

            if self._is_dicom_file(file_path):
                self.anonymize_dicom_header(file_path, dry_run=dry_run)
                dicom_count += 1
            elif self._is_nifti_file(file_path):
                self.anonymize_nifti_header(file_path, dry_run=dry_run)
                nifti_count += 1
            else:
                skip_count += 1

        print(f"\nSummary for {directory}:")
        print(f"  DICOM files processed : {dicom_count}")
        print(f"  NIfTI files processed : {nifti_count}")
        print(f"  Files skipped         : {skip_count}")

    # ------------------------------------------------------------------ #
    #  FILE / FOLDER RENAMING  (unchanged from original)                  #
    # ------------------------------------------------------------------ #

    def rename_files_in_directory(self, directory, dry_run=True):
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            return

        renamed_count = 0
        files = [f for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f))]

        for filename in files:
            new_filename = self.replace_ids_in_text(filename)
            if new_filename != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                if dry_run:
                    print(f"[DRY RUN] Would rename: {filename} -> {new_filename}")
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {filename} -> {new_filename}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"Error renaming {filename}: {e}")

        if not dry_run:
            print(f"Renamed {renamed_count} files in {directory}")

    def rename_folders_in_directory(self, directory, dry_run=True):
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            return

        renamed_count = 0
        folders = [f for f in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, f))]
        folders.sort(reverse=True)

        for foldername in folders:
            new_foldername = self.replace_ids_in_text(foldername)
            if new_foldername != foldername:
                old_path = os.path.join(directory, foldername)
                new_path = os.path.join(directory, new_foldername)
                if dry_run:
                    print(f"[DRY RUN] Would rename folder: {foldername} -> {new_foldername}")
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed folder: {foldername} -> {new_foldername}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"Error renaming folder {foldername}: {e}")

        if not dry_run:
            print(f"Renamed {renamed_count} folders in {directory}")

    def recursively_rename_all(self, directory, dry_run=True):
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            return

        print(f"\n{'=' * 60}")
        print(f"Processing directory: {directory}")
        print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL RENAME'}")
        print(f"{'=' * 60}\n")

        # Rename files (depth-first)
        for root, dirs, files in os.walk(directory, topdown=False):
            if files:
                print(f"\nProcessing files in: {root}")
                for filename in files:
                    new_filename = self.replace_ids_in_text(filename)
                    if new_filename != filename:
                        old_path = os.path.join(root, filename)
                        new_path = os.path.join(root, new_filename)
                        if dry_run:
                            print(f"  [DRY RUN] {filename} -> {new_filename}")
                        else:
                            try:
                                os.rename(old_path, new_path)
                                print(f"  ✓ {filename} -> {new_filename}")
                            except Exception as e:
                                print(f"  ✗ Error renaming {filename}: {e}")

        # Rename folders (bottom-up)
        for root, dirs, files in os.walk(directory, topdown=False):
            if dirs:
                print(f"\nProcessing folders in: {root}")
                for dirname in dirs:
                    new_dirname = self.replace_ids_in_text(dirname)
                    if new_dirname != dirname:
                        old_path = os.path.join(root, dirname)
                        new_path = os.path.join(root, new_dirname)
                        if dry_run:
                            print(f"  [DRY RUN] {dirname}/ -> {new_dirname}/")
                        else:
                            try:
                                os.rename(old_path, new_path)
                                print(f"  ✓ {dirname}/ -> {new_dirname}/")
                            except Exception as e:
                                print(f"  ✗ Error renaming folder {dirname}: {e}")

        print(f"\n{'=' * 60}")
        print(f"Completed processing: {directory}")
        print(f"{'=' * 60}\n")


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    main_directories = [
        r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_full",
        r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_nifti",
        r"E:\MBashiri\Thesis\p6\Data\MS_100_model_input",
        r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_gifs",
        r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_masks",
        r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_registered",
        r"E:\MBashiri\Thesis\p6\Data\MS_100_patient_preprocessed",
    ]

    anonymizer = PatientIDAnonymizer(main_directories)

    # Step 1: Create ID mapping
    print("Step 1: Creating ID mapping...")
    anonymizer.create_id_mapping()

    # Step 2: Save mapping for reference
    anonymizer.save_mapping("patient_id_mapping.json")

    # Step 3: Display mapping (first 10 entries)
    print("\nFirst 10 ID mappings:")
    for i, (old_id, new_id) in enumerate(list(anonymizer.id_mapping.items())[:10]):
        print(f"  {old_id} -> {new_id}")
    if len(anonymizer.id_mapping) > 10:
        print(f"  ... and {len(anonymizer.id_mapping) - 10} more")

    # ------------------------------------------------------------------ #
    #  DRY RUNS                                                            #
    # ------------------------------------------------------------------ #
    
    print("\n" + "=" * 60)
    print("PERFORMING DRY RUN — No actual changes will be made")
    print("=" * 60)
    
    # --- File/folder renaming dry run (original behaviour) ---
    for directory in main_directories:
        anonymizer.recursively_rename_all(directory, dry_run=True)
    
    # --- JSON file anonymization dry run (new) ---
    print("\n--- JSON files dry run ---")
    for directory in main_directories:
        anonymizer.process_all_json_files(directory, recursive=True, dry_run=True)
    
    # --- Imaging header anonymization dry run (new) ---
    print("\n--- Imaging header anonymization dry run ---")
    for directory in main_directories:
        anonymizer.anonymize_imaging_headers_in_directory(
            directory, recursive=True, dry_run=True
        )

    # ------------------------------------------------------------------ #
    #  ACTUAL CHANGES (confirm before running)                            #
    # ------------------------------------------------------------------ #

    # user_input = input(
    #     "\nAre you sure you want to apply ALL changes "
    #     "(rename files/folders + anonymize JSON fields + clear imaging headers)? (yes/no): "
    # )
    # if user_input.lower() == 'yes':

    #     # 1. Rename files and folders
    #     print("\n[1/3] Renaming files and folders...")
    #     for directory in main_directories:
    #         anonymizer.recursively_rename_all(directory, dry_run=False)

    #     # 2. Anonymize JSON field values
    #     print("\n[2/3] Anonymizing JSON metadata files...")
    #     for directory in main_directories:
    #         anonymizer.process_all_json_files(directory, recursive=True, dry_run=False)

    #     # 3. Clear birth date from imaging headers
    #     print("\n[3/3] Clearing PatientBirthDate from imaging headers...")
    #     for directory in main_directories:
    #         anonymizer.anonymize_imaging_headers_in_directory(
    #             directory, recursive=True, dry_run=False
    #         )

    #     print("\nAll anonymization steps completed!")

    # else:
    #     print("No changes made.")