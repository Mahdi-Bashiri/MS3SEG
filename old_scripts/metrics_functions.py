###################### Enhanced Metrics for K-Fold ######################


def dice_coefficient_multiclass(y_true, y_pred, class_id):
    """Calculate Dice coefficient for specific class"""
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)
    
    smooth = 1e-6
    intersection = np.sum(y_true_class * y_pred_class)
    return (2. * intersection + smooth) / (np.sum(y_true_class) + np.sum(y_pred_class) + smooth)

def dice_coefficient_binary(y_true, y_pred):
    """Calculate Dice coefficient for binary segmentation"""
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def iou_coefficient_multiclass(y_true, y_pred, class_id):
    """Calculate IoU coefficient for specific class"""
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)
    
    intersection = np.sum(y_true_class * y_pred_class)
    union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def iou_coefficient_binary(y_true, y_pred):
    """Calculate IoU coefficient for binary segmentation"""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def hausdorff_distance_95(y_true, y_pred, pixel_spacing=0.9):
    """Calculate 95th percentile Hausdorff Distance"""
    from scipy import ndimage
    
    # Get boundary points using morphological operations
    true_boundary = y_true - ndimage.binary_erosion(y_true.astype(bool))
    pred_boundary = y_pred - ndimage.binary_erosion(y_pred.astype(bool))
    
    # Get coordinates of boundary points
    true_coords = np.column_stack(np.where(true_boundary))
    pred_coords = np.column_stack(np.where(pred_boundary))
    
    # Handle edge cases
    if len(true_coords) == 0 or len(pred_coords) == 0:
        if len(true_coords) == 0 and len(pred_coords) == 0:
            return 0.0
        else:
            return float('inf')
    
    # Calculate directed Hausdorff distances
    distances_true_to_pred = []
    for true_point in true_coords:
        min_dist = np.min(np.linalg.norm(pred_coords - true_point, axis=1))
        distances_true_to_pred.append(min_dist)
    
    distances_pred_to_true = []
    for pred_point in pred_coords:
        min_dist = np.min(np.linalg.norm(true_coords - pred_point, axis=1))
        distances_pred_to_true.append(min_dist)
    
    # Combine all distances and calculate 95th percentile
    all_distances = distances_true_to_pred + distances_pred_to_true
    hd95_pixels = np.percentile(all_distances, 95)
    hd95_mm = hd95_pixels * pixel_spacing
    
    return hd95_mm

def calculate_comprehensive_metrics_kfold(y_true, y_pred, scenario_config, model_name, fold_idx):
    """Calculate comprehensive metrics for k-fold scenario"""
    metrics = {
        'Model': model_name,
        'Fold': fold_idx,
        'Scenario': scenario_config['description']
    }
    
    if scenario_config['num_classes'] in [3, 4]:  # Multi-class (3 or 4 classes)
        # Overall accuracy
        acc = accuracy_score(y_true.flatten(), y_pred.flatten())
        metrics['Overall_Accuracy'] = acc
        
        class_names = scenario_config['class_names']
        
        # Per-class metrics
        for class_id, class_name in enumerate(class_names):
            if class_id == 0:  # Skip background for main analysis
                continue
                
            dice = dice_coefficient_multiclass(y_true.flatten(), y_pred.flatten(), class_id)
            iou = iou_coefficient_multiclass(y_true.flatten(), y_pred.flatten(), class_id)
            
            # Calculate HD95 across all test images
            hd95_values = []
            for i in range(len(y_true)):
                y_true_class = (y_true[i] == class_id).astype(np.uint8)
                y_pred_class = (y_pred[i] == class_id).astype(np.uint8)
                hd95 = hausdorff_distance_95(y_true_class, y_pred_class)
                if not np.isinf(hd95):
                    hd95_values.append(hd95)
            
            mean_hd95 = np.mean(hd95_values) if hd95_values else float('inf')
            
            metrics[f'{class_name}_Dice'] = dice
            metrics[f'{class_name}_IoU'] = iou
            metrics[f'{class_name}_HD95'] = mean_hd95
        
        # Mean metrics across non-background classes
        metrics['Mean_Dice'] = np.mean([metrics[f'{name}_Dice'] for name in class_names[1:]])
        metrics['Mean_IoU'] = np.mean([metrics[f'{name}_IoU'] for name in class_names[1:]])
         
    else:  # Binary
        # Binary metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        acc = accuracy_score(y_true_flat, y_pred_flat)
        precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        dice = dice_coefficient_binary(y_true_flat, y_pred_flat)
        iou = iou_coefficient_binary(y_true_flat, y_pred_flat)
        
        # Calculate HD95
        hd95_values = []
        for i in range(len(y_true)):
            hd95 = hausdorff_distance_95(y_true[i], y_pred[i])
            if not np.isinf(hd95):
                hd95_values.append(hd95)
        
        mean_hd95 = np.mean(hd95_values) if hd95_values else float('inf')
        
        metrics['Accuracy'] = acc
        metrics['Precision'] = precision
        metrics['Recall'] = recall
        metrics['F1_Score'] = f1
        metrics['Dice'] = dice
        metrics['IoU'] = iou
        metrics['HD95'] = mean_hd95
    
    return metrics
