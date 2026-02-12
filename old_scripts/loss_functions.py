###################### Libraries ######################

import numpy as np

# Deep Learning
import tensorflow as tf
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


###################### Loss Functions ######################

def binary_dice_loss():
    """Binary Dice loss for binary segmentation"""
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - dice_coef
    return loss

def binary_focal_loss(alpha=0.25, gamma=2.0):
    """Binary focal loss"""
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        focal_loss = -focal_weight * K.log(pt)
        return K.mean(focal_loss)
    return loss

def multiclass_dice_loss(num_classes=4, class_weights=None):
    """Multi-class dice loss"""
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        dice_scores = []

        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]

            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)

            intersection = K.sum(y_true_f * y_pred_f)
            dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

            if class_weights is not None:
                dice_coef = dice_coef * class_weights[class_idx]

            dice_scores.append(dice_coef)

        mean_dice = K.mean(K.stack(dice_scores))
        return 1 - mean_dice
    return loss

def unified_focal_loss(class_weights, delta=0.6, gamma=0.5):
    """Unified Focal Loss for multi-class segmentation"""
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calculate metrics for all classes at once using vectorized operations
        y_true_f = K.reshape(y_true, (-1, tf.shape(y_true)[-1]))
        y_pred_f = K.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
        
        # Calculate TP, FP, FN for all classes
        tp = K.sum(y_true_f * y_pred_f, axis=0)
        fp = K.sum((1 - y_true_f) * y_pred_f, axis=0)
        fn = K.sum(y_true_f * (1 - y_pred_f), axis=0)
        
        # Calculate precision, recall, dice for all classes
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        
        # Calculate unified loss for all classes
        unified_loss = K.pow(1 - dice, gamma) * K.pow(1 - precision * recall, delta)
        
        # Apply class weights if provided
        if class_weights is not None:
            class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
            unified_loss = unified_loss * class_weights_tensor
        
        return K.mean(unified_loss)
    return loss

def calculate_class_weights(masks, num_classes):
    """Calculate class weights inversely proportional to class frequency"""
    flattened = masks.flatten()
    class_counts = np.bincount(flattened, minlength=num_classes)
    total_pixels = len(flattened)
    class_weights = total_pixels / (num_classes * class_counts)
    class_weights = class_weights / class_weights[0]
    return class_weights
