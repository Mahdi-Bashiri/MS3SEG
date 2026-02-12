"""
Data Loading Utilities for MS3SEG Dataset
"""

import numpy as np
import os
from pathlib import Path
import nibabel as nib
from sklearn.model_selection import KFold
import json
from typing import Tuple, List, Dict


class MS3SEGDataLoader:
    """Data loader for MS3SEG dataset"""
    
    def __init__(self, data_dir: str, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize data loader
        
        Args:
            data_dir: Root directory containing the dataset
            target_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
    
    def load_patient_data(self, patient_id: str) -> Dict[str, np.ndarray]:
        """
        Load all data for a single patient
        
        Args:
            patient_id: Patient identifier (e.g., '001', '002')
        
        Returns:
            Dictionary containing:
                - 'flair': FLAIR image (H, W, slices)
                - 't1': T1-weighted image
                - 't2': T2-weighted image
                - 'mask': Ground truth segmentation mask
        """
        patient_dir = self.data_dir / f"patient_{patient_id}"
        
        if not patient_dir.exists():
            raise ValueError(f"Patient directory not found: {patient_dir}")
        
        data = {}
        
        # Load FLAIR (primary sequence for annotation)
        flair_path = patient_dir / f"{patient_id}_FLAIR_preprocessed.nii.gz"
        if flair_path.exists():
            data['flair'] = nib.load(str(flair_path)).get_fdata()
        
        # Load T1
        t1_path = patient_dir / f"{patient_id}_T1_preprocessed.nii.gz"
        if t1_path.exists():
            data['t1'] = nib.load(str(t1_path)).get_fdata()
        
        # Load T2
        t2_path = patient_dir / f"{patient_id}_T2_preprocessed.nii.gz"
        if t2_path.exists():
            data['t2'] = nib.load(str(t2_path)).get_fdata()
        
        # Load mask (tri-mask annotation)
        mask_path = patient_dir / f"{patient_id}_mask.nii.gz"
        if mask_path.exists():
            data['mask'] = nib.load(str(mask_path)).get_fdata()
        
        return data
    
    def prepare_slices(self, images: np.ndarray, masks: np.ndarray, 
                      normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare 2D slices from 3D volumes
        
        Args:
            images: 3D image volume (H, W, slices)
            masks: 3D mask volume (H, W, slices)
            normalize: Whether to normalize images to [0, 1]
        
        Returns:
            Tuple of (prepared_images, prepared_masks)
                - prepared_images: (slices, H, W, 1)
                - prepared_masks: (slices, H, W)
        """
        # Transpose to (slices, H, W)
        if images.ndim == 3:
            images = np.transpose(images, (2, 0, 1))
            masks = np.transpose(masks, (2, 0, 1))
        
        # Normalize images
        if normalize:
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        # Add channel dimension for images
        images = np.expand_dims(images, axis=-1)
        
        return images.astype(np.float32), masks.astype(np.uint8)
    
    def load_dataset(self, patient_ids: List[str], 
                    scenario: str = 'multi_class') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load complete dataset for specified patients
        
        Args:
            patient_ids: List of patient IDs to load
            scenario: Segmentation scenario ('multi_class', 'binary_abnormal_wmh', 
                     'binary_ventricles', 'three_class')
        
        Returns:
            Tuple of (images, masks)
                - images: (N, H, W, 1) where N is total number of slices
                - masks: (N, H, W) for binary or (N, H, W, num_classes) for multi-class
        """
        all_images = []
        all_masks = []
        
        for patient_id in patient_ids:
            try:
                # Load patient data
                data = self.load_patient_data(patient_id)
                
                if 'flair' not in data or 'mask' not in data:
                    print(f"Warning: Skipping patient {patient_id} - missing data")
                    continue
                
                # Prepare slices
                images, masks = self.prepare_slices(data['flair'], data['mask'])
                
                # Apply scenario-specific mask transformation
                masks = self.transform_masks_for_scenario(masks, scenario)
                
                all_images.append(images)
                all_masks.append(masks)
                
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue
        
        # Concatenate all slices
        all_images = np.concatenate(all_images, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        return all_images, all_masks
    
    def transform_masks_for_scenario(self, masks: np.ndarray, 
                                    scenario: str) -> np.ndarray:
        """
        Transform masks according to the specified scenario
        
        Args:
            masks: Original masks (N, H, W) with values 0-3
            scenario: Scenario type
        
        Returns:
            Transformed masks
        """
        if scenario == 'multi_class':
            # Keep all 4 classes (0: background, 1: ventricles, 2: normal WMH, 3: abnormal WMH)
            return masks
        
        elif scenario == 'three_class':
            # Merge normal WMH (class 2) into background (class 0)
            # Map: 0->0, 1->1, 2->0, 3->2
            transformed = np.zeros_like(masks)
            transformed[masks == 1] = 1  # Ventricles
            transformed[masks == 3] = 2  # Abnormal WMH
            return transformed
        
        elif scenario == 'binary_abnormal_wmh':
            # Binary: abnormal WMH vs everything else
            return (masks == 3).astype(np.uint8)
        
        elif scenario == 'binary_ventricles':
            # Binary: ventricles vs everything else
            return (masks == 1).astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def create_kfold_splits(self, patient_ids: List[str], n_splits: int = 5, 
                           test_size: float = 0.2, random_state: int = 42,
                           save_path: str = None) -> Dict:
        """
        Create k-fold cross-validation splits
        
        Args:
            patient_ids: List of all patient IDs
            n_splits: Number of folds
            test_size: Proportion for held-out test set
            random_state: Random seed
            save_path: Optional path to save splits as JSON
        
        Returns:
            Dictionary with fold information
        """
        np.random.seed(random_state)
        patient_ids = np.array(patient_ids)
        
        # Shuffle patient IDs
        shuffled_ids = patient_ids.copy()
        np.random.shuffle(shuffled_ids)
        
        # Split into train/val pool and test set
        n_test = int(len(shuffled_ids) * test_size)
        test_ids = shuffled_ids[:n_test]
        train_val_ids = shuffled_ids[n_test:]
        
        # Create k-fold splits from train/val pool
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        splits = {
            'test_patients': test_ids.tolist(),
            'folds': []
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_val_ids)):
            fold_data = {
                'fold': fold_idx + 1,
                'train_patients': train_val_ids[train_idx].tolist(),
                'val_patients': train_val_ids[val_idx].tolist()
            }
            splits['folds'].append(fold_data)
        
        # Save splits if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(splits, f, indent=2)
            print(f"Splits saved to: {save_path}")
        
        return splits
    
    def load_splits(self, splits_path: str) -> Dict:
        """
        Load pre-existing k-fold splits from JSON
        
        Args:
            splits_path: Path to splits JSON file
        
        Returns:
            Dictionary with fold information
        """
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        return splits


def calculate_class_weights(masks: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Calculate class weights inversely proportional to class frequency
    
    Args:
        masks: Mask array (N, H, W)
        num_classes: Number of classes
    
    Returns:
        Array of class weights (num_classes,)
    """
    flattened = masks.flatten()
    class_counts = np.bincount(flattened, minlength=num_classes)
    total_pixels = len(flattened)
    
    # Inverse frequency weighting
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Normalize relative to background
    class_weights = class_weights / class_weights[0]
    
    return class_weights


if __name__ == "__main__":
    # Example usage
    data_loader = MS3SEGDataLoader(data_dir="path/to/MS3SEG/data")
    
    # Create splits
    all_patients = [f"{i:03d}" for i in range(1, 101)]  # 001 to 100
    splits = data_loader.create_kfold_splits(
        all_patients, 
        n_splits=5, 
        test_size=0.2,
        save_path="splits/5fold_splits.json"
    )
    
    print(f"Test patients: {len(splits['test_patients'])}")
    print(f"Number of folds: {len(splits['folds'])}")
    for fold in splits['folds']:
        print(f"Fold {fold['fold']}: "
              f"Train={len(fold['train_patients'])}, "
              f"Val={len(fold['val_patients'])}")
