"""
UNETR (UNEt TRansformers) Architecture
Simplified implementation for MS3SEG Dataset

Reference:
Hatamizadeh et al., "UNETR: Transformers for 3D Medical Image Segmentation"
IEEE WACV, 2022

Note: This is a 2D adaptation. For full 3D implementation, refer to MONAI library.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LayerNormalization,
    Dense, Reshape, MultiHeadAttention, Add, Dropout,
    BatchNormalization, Activation, concatenate
)
from tensorflow.keras.models import Model
import numpy as np


def mlp_block(x, hidden_units, dropout_rate=0.1):
    """Multi-Layer Perceptron block"""
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


def transformer_block(x, num_heads=12, mlp_dim=3072, dropout_rate=0.1):
    """Transformer encoder block"""
    # Layer normalization and Multi-Head Self-Attention
    x1 = LayerNormalization(epsilon=1e-6)(x)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=x.shape[-1] // num_heads, dropout=dropout_rate
    )(x1, x1)
    x2 = Add()([attention_output, x])
    
    # Layer normalization and MLP
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp_block(x3, hidden_units=[mlp_dim, x.shape[-1]], dropout_rate=dropout_rate)
    output = Add()([x3, x2])
    
    return output


def create_patches(images, patch_size=16):
    """Convert images to sequence of patches"""
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patch_dims = patches.shape[-1]
    num_patches = patches.shape[1] * patches.shape[2]
    patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
    return patches, num_patches


def build_unetr(input_shape=(256, 256, 1), num_classes=4, patch_size=16, 
                hidden_size=768, num_transformer_layers=12, num_heads=12,
                mlp_dim=3072, dropout_rate=0.1):
    """
    Build UNETR model with Vision Transformer encoder
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of segmentation classes
        patch_size: Size of image patches for transformer
        hidden_size: Hidden dimension size for transformer
        num_transformer_layers: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_dim: MLP hidden dimension
        dropout_rate: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # Create patches and linear projection
    patches, num_patches = create_patches(inputs, patch_size)
    
    # Linear projection of flattened patches
    projected_patches = Dense(hidden_size)(patches)
    
    # Add positional embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=hidden_size
    )(positions)
    encoded_patches = projected_patches + position_embedding
    
    # Transformer Encoder
    encoder_outputs = []
    x = encoded_patches
    
    for i in range(num_transformer_layers):
        x = transformer_block(x, num_heads, mlp_dim, dropout_rate)
        # Save outputs at specific layers for skip connections
        if i in [2, 5, 8, 11]:  # Skip connections at layers 3, 6, 9, 12
            encoder_outputs.append(x)
    
    # Reshape transformer outputs for decoder
    grid_size = input_shape[0] // patch_size
    
    def reshape_transformer_output(transformer_output, target_size):
        """Reshape transformer output to 2D feature map"""
        batch_size = tf.shape(transformer_output)[0]
        reshaped = tf.reshape(transformer_output, 
                             [batch_size, grid_size, grid_size, hidden_size])
        # Upsample to target size
        upsampled = tf.image.resize(reshaped, size=(target_size, target_size))
        return upsampled
    
    # CNN Decoder with skip connections from transformer
    # Level 1: 64x64
    skip1 = reshape_transformer_output(encoder_outputs[0], 64)
    decoder1 = Conv2D(512, 3, padding='same', activation='relu')(skip1)
    decoder1 = Conv2D(512, 3, padding='same', activation='relu')(decoder1)
    
    # Level 2: 128x128
    up2 = Conv2DTranspose(256, 2, strides=2, padding='same')(decoder1)
    skip2 = reshape_transformer_output(encoder_outputs[1], 128)
    decoder2 = concatenate([up2, skip2])
    decoder2 = Conv2D(256, 3, padding='same', activation='relu')(decoder2)
    decoder2 = Conv2D(256, 3, padding='same', activation='relu')(decoder2)
    
    # Level 3: 256x256
    up3 = Conv2DTranspose(128, 2, strides=2, padding='same')(decoder2)
    skip3 = reshape_transformer_output(encoder_outputs[2], 256)
    decoder3 = concatenate([up3, skip3])
    decoder3 = Conv2D(128, 3, padding='same', activation='relu')(decoder3)
    decoder3 = Conv2D(128, 3, padding='same', activation='relu')(decoder3)
    
    # Final output
    outputs = Conv2D(num_classes, 1, activation='softmax')(decoder3)
    
    model = Model(inputs=inputs, outputs=outputs, name='UNETR')
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating UNETR model...")
    model = build_unetr(input_shape=(256, 256, 1), num_classes=4)
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
