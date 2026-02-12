"""
Evaluation Metrics for MS3SEG Dataset
Implements Dice Score, IoU, and Hausdorff Distance
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice Similarity Coefficient
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient (float between 0 and 1)
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    return dice


def iou_score(y_true, y_pred, smooth=1e-6):
    """
    Calculate Intersection over Union (Jaccard Index)
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score (float between 0 and 1)
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def hausdorff_distance_95(y_true, y_pred, voxel_spacing=(1.0, 1.0)):
    """
    Calculate 95th percentile Hausdorff Distance
    
    Args:
        y_true: Ground truth binary mask (2D array)
        y_pred: Predicted binary mask (2D array)
        voxel_spacing: Pixel spacing in mm (height, width)
    
    Returns:
        95th percentile Hausdorff distance in mm
    """
    # Check if masks are empty
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 0.0
    elif np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return np.inf
    
    # Get surface points (boundary of masks)
    from scipy.ndimage import binary_erosion
    
    # Create boundary by subtracting eroded version
    true_border = y_true.astype(bool) ^ binary_erosion(y_true.astype(bool))
    pred_border = y_pred.astype(bool) ^ binary_erosion(y_pred.astype(bool))
    
    # Get coordinates of boundary points
    true_coords = np.argwhere(true_border) * np.array(voxel_spacing)
    pred_coords = np.argwhere(pred_border) * np.array(voxel_spacing)
    
    if len(true_coords) == 0 or len(pred_coords) == 0:
        return np.inf
    
    # Calculate distances from each point in pred to nearest point in true
    from scipy.spatial.distance import cdist
    distances_pred_to_true = cdist(pred_coords, true_coords).min(axis=1)
    distances_true_to_pred = cdist(true_coords, pred_coords).min(axis=1)
    
    # Combine distances and calculate 95th percentile
    all_distances = np.concatenate([distances_pred_to_true, distances_true_to_pred])
    hd95 = np.percentile(all_distances, 95)
    
    return hd95


def calculate_multiclass_metrics(y_true, y_pred, num_classes, class_names=None,
                                 voxel_spacing=(0.9, 0.9)):
    """
    Calculate metrics for multi-class segmentation
    
    Args:
        y_true: Ground truth masks (N, H, W) with integer class labels
        y_pred: Predicted masks (N, H, W) with integer class labels
        num_classes: Number of classes
        class_names: Optional list of class names
        voxel_spacing: Pixel spacing in mm
    
    Returns:
        Dictionary with metrics per class and overall metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    results = {
        'per_class': {},
        'overall': {}
    }
    
    dice_scores = []
    iou_scores = []
    hd95_scores = []
    
    # Calculate metrics for each class
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        
        # Create binary masks for this class
        true_class = (y_true == class_id).astype(np.uint8)
        pred_class = (y_pred == class_id).astype(np.uint8)
        
        # Calculate metrics
        dice = dice_coefficient(true_class, pred_class)
        iou = iou_score(true_class, pred_class)
        
        # Calculate HD95 (average over all samples)
        hd95_list = []
        for i in range(y_true.shape[0]):
            try:
                hd95 = hausdorff_distance_95(true_class[i], pred_class[i], voxel_spacing)
                if not np.isinf(hd95):
                    hd95_list.append(hd95)
            except:
                pass
        
        hd95_mean = np.mean(hd95_list) if hd95_list else np.inf
        
        results['per_class'][class_name] = {
            'Dice': dice,
            'IoU': iou,
            'HD95': hd95_mean
        }
        
        # Skip background for overall metrics
        if class_id > 0:
            dice_scores.append(dice)
            iou_scores.append(iou)
            if not np.isinf(hd95_mean):
                hd95_scores.append(hd95_mean)
    
    # Calculate overall metrics (excluding background)
    results['overall'] = {
        'Mean_Dice': np.mean(dice_scores),
        'Mean_IoU': np.mean(iou_scores),
        'Mean_HD95': np.mean(hd95_scores) if hd95_scores else np.inf
    }
    
    return results


def calculate_binary_metrics(y_true, y_pred, voxel_spacing=(0.9, 0.9)):
    """
    Calculate metrics for binary segmentation
    
    Args:
        y_true: Ground truth binary masks (N, H, W)
        y_pred: Predicted binary masks (N, H, W)
        voxel_spacing: Pixel spacing in mm
    
    Returns:
        Dictionary with Dice, IoU, and HD95
    """
    # Ensure binary
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    
    dice = dice_coefficient(y_true, y_pred)
    iou = iou_score(y_true, y_pred)
    
    # Calculate HD95
    hd95_list = []
    for i in range(y_true.shape[0]):
        try:
            hd95 = hausdorff_distance_95(y_true[i], y_pred[i], voxel_spacing)
            if not np.isinf(hd95):
                hd95_list.append(hd95)
        except:
            pass
    
    hd95_mean = np.mean(hd95_list) if hd95_list else np.inf
    
    return {
        'Dice': dice,
        'IoU': iou,
        'HD95': hd95_mean
    }


# Keras/TensorFlow loss functions for training

def dice_loss(smooth=1e-6):
    """Dice loss for training"""
    def loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - dice
    return loss


def multiclass_dice_loss(num_classes=4, class_weights=None, smooth=1e-6):
    """Multi-class dice loss"""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        dice_scores = []
        
        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]
            
            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)
            
            intersection = K.sum(y_true_f * y_pred_f)
            dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
            
            if class_weights is not None:
                dice = dice * class_weights[class_idx]
            
            dice_scores.append(dice)
        
        mean_dice = K.mean(K.stack(dice_scores))
        return 1 - mean_dice
    return loss


def unified_focal_loss(class_weights=None, delta=0.6, gamma=0.5, smooth=1e-6):
    """
    Unified Focal Loss combining Dice and Focal components
    
    Reference: Yeung et al., "Unified Focal loss: Generalising Dice and cross entropy-based 
    losses to handle class imbalanced medical image segmentation"
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Reshape for vectorized operations
        y_true_f = K.reshape(y_true, (-1, tf.shape(y_true)[-1]))
        y_pred_f = K.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
        
        # Calculate TP, FP, FN
        tp = K.sum(y_true_f * y_pred_f, axis=0)
        fp = K.sum((1 - y_true_f) * y_pred_f, axis=0)
        fn = K.sum(y_true_f * (1 - y_pred_f), axis=0)
        
        # Calculate precision, recall, dice
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        
        # Unified focal loss
        unified_loss = K.pow(1 - dice, gamma) * K.pow(1 - precision * recall, delta)
        
        # Apply class weights if provided
        if class_weights is not None:
            class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
            unified_loss = unified_loss * class_weights_tensor
        
        return K.mean(unified_loss)
    return loss


if __name__ == "__main__":
    # Example usage
    print("Testing metrics...")
    
    # Create dummy data
    y_true = np.random.randint(0, 4, size=(10, 256, 256))
    y_pred = np.random.randint(0, 4, size=(10, 256, 256))
    
    # Multi-class metrics
    class_names = ['Background', 'Ventricles', 'Normal WMH', 'Abnormal WMH']
    results = calculate_multiclass_metrics(y_true, y_pred, 4, class_names)
    
    print("\nPer-class metrics:")
    for class_name, metrics in results['per_class'].items():
        print(f"{class_name}: Dice={metrics['Dice']:.4f}, IoU={metrics['IoU']:.4f}, HD95={metrics['HD95']:.2f}mm")
    
    print("\nOverall metrics:")
    for metric_name, value in results['overall'].items():
        print(f"{metric_name}: {value:.4f}")
