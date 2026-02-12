"""
MS3SEG Model Architectures
Collection of deep learning models for MS lesion segmentation
"""

from .unet import build_unet
from .unet_plusplus import build_unet_plusplus
from .unetr import build_unetr
from .swin_unetr import build_swin_unetr

__all__ = [
    'build_unet',
    'build_unet_plusplus',
    'build_unetr',
    'build_swin_unetr'
]

# Model registry for easy access
MODEL_REGISTRY = {
    'U-Net': build_unet,
    'UNet++': build_unet_plusplus,
    'UNETR': build_unetr,
    'SwinUNETR': build_swin_unetr,
}


def get_model(model_name, **kwargs):
    """
    Get model by name with specified parameters
    
    Args:
        model_name: Name of the model ('U-Net', 'UNet++', 'UNETR', 'SwinUNETR')
        **kwargs: Model-specific parameters
    
    Returns:
        Keras model instance
    
    Example:
        >>> model = get_model('U-Net', input_shape=(256, 256, 1), num_classes=4)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](**kwargs)
