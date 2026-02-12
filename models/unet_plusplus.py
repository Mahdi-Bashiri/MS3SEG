"""
U-Net++ Architecture for Medical Image Segmentation
Implementation for MS3SEG Dataset

Reference:
Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
Deep Learning in Medical Image Analysis, 2018
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


def build_unet_plusplus(input_shape=(256, 256, 1), num_classes=4, filters=32,
                        deep_supervision=False, batch_norm=True, dropout_rate=0.0):
    """
    Build U-Net++ model with nested skip pathways
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of segmentation classes
        filters: Base number of filters
        deep_supervision: Whether to use deep supervision (multiple outputs)
        batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate in encoder blocks
    
    Returns:
        Compiled Keras model
    """
    inputs = Input(input_shape)
    
    # Encoder path with nested dense skip connections
    # Level 0
    conv0_0 = conv_block(inputs, filters, batch_norm=batch_norm, dropout_rate=dropout_rate)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0_0)
    
    # Level 1
    conv1_0 = conv_block(pool0, filters*2, batch_norm=batch_norm, dropout_rate=dropout_rate)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_0)
    
    up0_1 = UpSampling2D(size=(2, 2))(conv1_0)
    conv0_1 = conv_block(concatenate([up0_1, conv0_0]), filters, batch_norm=batch_norm)
    
    # Level 2
    conv2_0 = conv_block(pool1, filters*4, batch_norm=batch_norm, dropout_rate=dropout_rate)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_0)
    
    up1_1 = UpSampling2D(size=(2, 2))(conv2_0)
    conv1_1 = conv_block(concatenate([up1_1, conv1_0]), filters*2, batch_norm=batch_norm)
    
    up0_2 = UpSampling2D(size=(2, 2))(conv1_1)
    conv0_2 = conv_block(concatenate([up0_2, conv0_0, conv0_1]), filters, batch_norm=batch_norm)
    
    # Level 3
    conv3_0 = conv_block(pool2, filters*8, batch_norm=batch_norm, dropout_rate=dropout_rate)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_0)
    
    up2_1 = UpSampling2D(size=(2, 2))(conv3_0)
    conv2_1 = conv_block(concatenate([up2_1, conv2_0]), filters*4, batch_norm=batch_norm)
    
    up1_2 = UpSampling2D(size=(2, 2))(conv2_1)
    conv1_2 = conv_block(concatenate([up1_2, conv1_0, conv1_1]), filters*2, batch_norm=batch_norm)
    
    up0_3 = UpSampling2D(size=(2, 2))(conv1_2)
    conv0_3 = conv_block(concatenate([up0_3, conv0_0, conv0_1, conv0_2]), filters, batch_norm=batch_norm)
    
    # Level 4 (Bottleneck)
    conv4_0 = conv_block(pool3, filters*16, batch_norm=batch_norm, dropout_rate=dropout_rate)
    
    # Decoder with nested connections
    up3_1 = UpSampling2D(size=(2, 2))(conv4_0)
    conv3_1 = conv_block(concatenate([up3_1, conv3_0]), filters*8, batch_norm=batch_norm)
    
    up2_2 = UpSampling2D(size=(2, 2))(conv3_1)
    conv2_2 = conv_block(concatenate([up2_2, conv2_0, conv2_1]), filters*4, batch_norm=batch_norm)
    
    up1_3 = UpSampling2D(size=(2, 2))(conv2_2)
    conv1_3 = conv_block(concatenate([up1_3, conv1_0, conv1_1, conv1_2]), filters*2, batch_norm=batch_norm)
    
    up0_4 = UpSampling2D(size=(2, 2))(conv1_3)
    conv0_4 = conv_block(concatenate([up0_4, conv0_0, conv0_1, conv0_2, conv0_3]), filters, batch_norm=batch_norm)
    
    # Output layer(s)
    if deep_supervision:
        # Multiple outputs for deep supervision
        output1 = Conv2D(num_classes, 1, activation='softmax', name='output1')(conv0_1)
        output2 = Conv2D(num_classes, 1, activation='softmax', name='output2')(conv0_2)
        output3 = Conv2D(num_classes, 1, activation='softmax', name='output3')(conv0_3)
        output4 = Conv2D(num_classes, 1, activation='softmax', name='output4')(conv0_4)
        
        model = Model(inputs=inputs, outputs=[output1, output2, output3, output4], name='UNet++')
    else:
        # Single output from the deepest nested path
        outputs = Conv2D(num_classes, 1, activation='softmax')(conv0_4)
        model = Model(inputs=inputs, outputs=outputs, name='UNet++')
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = build_unet_plusplus(input_shape=(256, 256, 1), num_classes=4)
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
