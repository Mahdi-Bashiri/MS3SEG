"""
U-Net Architecture for Medical Image Segmentation
Implementation for MS3SEG Dataset
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    concatenate, BatchNormalization, Activation, Dropout
)
from tensorflow.keras.models import Model


def conv_block(inputs, filters, kernel_size=3, activation='relu', 
               batch_norm=True, dropout_rate=0.0):
    """
    Convolutional block with optional batch normalization and dropout
    
    Args:
        inputs: Input tensor
        filters: Number of filters
        kernel_size: Size of convolutional kernel
        activation: Activation function
        batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate (0 = no dropout)
    
    Returns:
        Output tensor after convolutions
    """
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x


def build_unet(input_shape=(256, 256, 1), num_classes=4, filters=64, 
               batch_norm=True, dropout_rate=0.0):
    """
    Build U-Net model architecture
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of segmentation classes
        filters: Base number of filters (doubles at each level)
        batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate in encoder blocks
    
    Returns:
        Compiled Keras model
    """
    inputs = Input(input_shape)
    
    # Encoder (Contracting Path)
    conv1 = conv_block(inputs, filters, batch_norm=batch_norm, dropout_rate=dropout_rate)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, filters*2, batch_norm=batch_norm, dropout_rate=dropout_rate)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, filters*4, batch_norm=batch_norm, dropout_rate=dropout_rate)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, filters*8, batch_norm=batch_norm, dropout_rate=dropout_rate)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = conv_block(pool4, filters*16, batch_norm=batch_norm, dropout_rate=dropout_rate)
    
    # Decoder (Expansive Path)
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv4], axis=-1)
    conv6 = conv_block(up6, filters*8, batch_norm=batch_norm)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv3], axis=-1)
    conv7 = conv_block(up7, filters*4, batch_norm=batch_norm)
    
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up8, conv2], axis=-1)
    conv8 = conv_block(up8, filters*2, batch_norm=batch_norm)
    
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up9, conv1], axis=-1)
    conv9 = conv_block(up9, filters, batch_norm=batch_norm)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs, name='U-Net')
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = build_unet(input_shape=(256, 256, 1), num_classes=4)
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
