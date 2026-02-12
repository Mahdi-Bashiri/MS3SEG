"""
Swin UNETR Architecture
Simplified implementation for MS3SEG Dataset

Reference:
Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation 
of Brain Tumors in MRI Images"
BrainLes Workshop, MICCAI 2022

Note: This is a simplified 2D adaptation. For full implementation with 
hierarchical shifted windows, refer to MONAI library or original paper.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LayerNormalization,
    Dense, Reshape, Add, Dropout, concatenate, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
import numpy as np


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows
    
    Args:
        x: (B, H, W, C)
        window_size: Window size
    
    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = tf.reshape(tf.transpose(x, [0, 1, 3, 2, 4, 5]), 
                        [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition
    
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, 
                            window_size, window_size, -1])
    x = tf.reshape(tf.transpose(x, [0, 1, 3, 2, 4, 5]), [B, H, W, -1])
    return x


def swin_transformer_block(x, num_heads=4, window_size=7, shift_size=0, 
                          mlp_ratio=4.0, dropout_rate=0.1):
    """
    Swin Transformer block with shifted window attention
    
    Args:
        x: Input feature map
        num_heads: Number of attention heads
        window_size: Window size for attention
        shift_size: Shift size for shifted window attention
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        dropout_rate: Dropout rate
    
    Returns:
        Output feature map
    """
    B, H, W, C = x.shape
    shortcut = x
    
    # Layer normalization
    x = LayerNormalization(epsilon=1e-5)(x)
    
    # Cyclic shift if shift_size > 0
    if shift_size > 0:
        shifted_x = tf.roll(x, shift=[-shift_size, -shift_size], axis=[1, 2])
    else:
        shifted_x = x
    
    # Simplified window-based multi-head self attention
    # Note: Full implementation would include masked attention for shifted windows
    # Flatten spatial dimensions for attention
    x_flat = tf.reshape(shifted_x, [B, H * W, C])
    
    # Multi-head attention (simplified)
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=C // num_heads, dropout=dropout_rate
    )(x_flat, x_flat)
    
    # Reshape back
    attention_output = tf.reshape(attention_output, [B, H, W, C])
    
    # Reverse cyclic shift
    if shift_size > 0:
        x = tf.roll(attention_output, shift=[shift_size, shift_size], axis=[1, 2])
    else:
        x = attention_output
    
    x = Dropout(dropout_rate)(x)
    x = Add()([shortcut, x])
    
    # MLP
    shortcut2 = x
    x = LayerNormalization(epsilon=1e-5)(x)
    
    # Reshape for MLP
    x_mlp = tf.reshape(x, [B, H * W, C])
    mlp_hidden_dim = int(C * mlp_ratio)
    x_mlp = Dense(mlp_hidden_dim, activation=tf.nn.gelu)(x_mlp)
    x_mlp = Dropout(dropout_rate)(x_mlp)
    x_mlp = Dense(C)(x_mlp)
    x_mlp = Dropout(dropout_rate)(x_mlp)
    
    # Reshape back and add residual
    x = tf.reshape(x_mlp, [B, H, W, C])
    x = Add()([shortcut2, x])
    
    return x


def patch_merging(x, reduction_factor=2):
    """
    Patch merging layer to reduce spatial resolution and increase channels
    
    Args:
        x: Input tensor (B, H, W, C)
        reduction_factor: Factor to reduce spatial dimensions
    
    Returns:
        Merged tensor (B, H/2, W/2, 2*C)
    """
    B, H, W, C = x.shape
    
    # Use strided convolution for downsampling
    x = Conv2D(C * 2, kernel_size=reduction_factor, strides=reduction_factor, 
               padding='same')(x)
    x = LayerNormalization(epsilon=1e-5)(x)
    
    return x


def build_swin_unetr(input_shape=(256, 256, 1), num_classes=4, 
                     embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                     window_size=7, mlp_ratio=4.0, dropout_rate=0.1):
    """
    Build Swin UNETR model
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of segmentation classes
        embed_dim: Patch embedding dimension
        depths: Number of blocks at each stage
        num_heads: Number of attention heads at each stage
        window_size: Window size for shifted window attention
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        dropout_rate: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # Patch embedding: 4x4 patches with embed_dim channels
    x = Conv2D(embed_dim, kernel_size=4, strides=4, padding='same')(inputs)
    x = LayerNormalization(epsilon=1e-5)(x)
    
    # Encoder with hierarchical feature extraction
    skip_connections = []
    
    # Stage 1: 64x64, dim=96
    for i in range(depths[0]):
        shift_size = 0 if (i % 2 == 0) else window_size // 2
        x = swin_transformer_block(x, num_heads[0], window_size, shift_size, 
                                  mlp_ratio, dropout_rate)
    skip_connections.append(x)
    
    # Stage 2: 32x32, dim=192
    x = patch_merging(x)
    for i in range(depths[1]):
        shift_size = 0 if (i % 2 == 0) else window_size // 2
        x = swin_transformer_block(x, num_heads[1], window_size, shift_size, 
                                  mlp_ratio, dropout_rate)
    skip_connections.append(x)
    
    # Stage 3: 16x16, dim=384
    x = patch_merging(x)
    for i in range(depths[2]):
        shift_size = 0 if (i % 2 == 0) else window_size // 2
        x = swin_transformer_block(x, num_heads[2], window_size, shift_size, 
                                  mlp_ratio, dropout_rate)
    skip_connections.append(x)
    
    # Stage 4: 8x8, dim=768 (bottleneck)
    x = patch_merging(x)
    for i in range(depths[3]):
        shift_size = 0 if (i % 2 == 0) else window_size // 2
        x = swin_transformer_block(x, num_heads[3], window_size, shift_size, 
                                  mlp_ratio, dropout_rate)
    
    # Decoder with skip connections
    # Up 1: 16x16
    x = Conv2DTranspose(embed_dim * 4, kernel_size=2, strides=2, padding='same')(x)
    x = concatenate([x, skip_connections[2]])
    x = Conv2D(embed_dim * 4, 3, padding='same', activation='relu')(x)
    x = Conv2D(embed_dim * 4, 3, padding='same', activation='relu')(x)
    
    # Up 2: 32x32
    x = Conv2DTranspose(embed_dim * 2, kernel_size=2, strides=2, padding='same')(x)
    x = concatenate([x, skip_connections[1]])
    x = Conv2D(embed_dim * 2, 3, padding='same', activation='relu')(x)
    x = Conv2D(embed_dim * 2, 3, padding='same', activation='relu')(x)
    
    # Up 3: 64x64
    x = Conv2DTranspose(embed_dim, kernel_size=2, strides=2, padding='same')(x)
    x = concatenate([x, skip_connections[0]])
    x = Conv2D(embed_dim, 3, padding='same', activation='relu')(x)
    x = Conv2D(embed_dim, 3, padding='same', activation='relu')(x)
    
    # Up 4: 256x256
    x = Conv2DTranspose(embed_dim // 2, kernel_size=4, strides=4, padding='same')(x)
    x = Conv2D(embed_dim // 2, 3, padding='same', activation='relu')(x)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Swin-UNETR')
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating Swin UNETR model...")
    model = build_swin_unetr(input_shape=(256, 256, 1), num_classes=4)
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
