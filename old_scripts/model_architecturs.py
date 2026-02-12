
###################### Libraries ######################

# Deep Learning
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras import backend as K
from tensorflow.keras import layers, optimizers, callbacks
from keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


###################### Model Architectures (Same as before) ######################

def build_unetr(input_shape=(256, 256, 1), num_classes=4):
    """
    UNETR (UNet Transformers) implementation for medical image segmentation
    
    Architecture:
    - Vision Transformer (ViT) encoder for global context
    - CNN decoder with skip connections from transformer layers
    - Multi-scale feature extraction from different transformer blocks
    
    Based on: Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image Segmentation" (2022)
    Adapted for 2D brain MRI segmentation
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        keras.Model: UNETR model
    """
    
    # Fixed hyperparameters for 256x256 input
    patch_size = 16  # 16x16 patches
    embed_dim = 768  # Embedding dimension
    num_heads = 12   # Multi-head attention heads
    num_layers = 12  # Transformer layers
    mlp_ratio = 4    # MLP expansion ratio
    
    # Calculate patch dimensions (fixed for 256x256 input)
    img_height, img_width = input_shape[0], input_shape[1]
    patch_height = img_height // patch_size  # 256 // 16 = 16
    patch_width = img_width // patch_size    # 256 // 16 = 16
    num_patches = patch_height * patch_width # 16 * 16 = 256
    
    inputs = Input(input_shape, name='input_layer')
    
    # ============ PATCH EMBEDDING ============
    # Extract patches using Conv2D with stride = patch_size
    patches = Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        kernel_initializer='he_normal',
        name='patch_embedding'
    )(inputs)
    
    # Reshape to sequence format: (batch, num_patches, embed_dim)
    # patches shape after Conv2D: (batch, 16, 16, 768)
    patches_flat = layers.Reshape((num_patches, embed_dim))(patches)  # (batch, 256, 768)
    
    # ============ POSITIONAL ENCODING ============
    # Create positional embedding layer
    pos_embed = layers.Embedding(
        input_dim=num_patches,
        output_dim=embed_dim,
        name='positional_embedding'
    )
    
    # Create position indices
    positions = tf.range(num_patches)
    positions = tf.expand_dims(positions, 0)
    positions = tf.tile(positions, [tf.shape(patches_flat)[0], 1])
    
    # Add positional encoding
    pos_encoding = pos_embed(positions)
    encoded_patches = layers.Add()([patches_flat, pos_encoding])
    
    # ============ TRANSFORMER BLOCKS ============
    def transformer_block(x, num_heads, embed_dim, mlp_ratio, dropout_rate=0.1, name_prefix=""):
        """Transformer encoder block with multi-head self-attention"""
        
        # Multi-Head Self-Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name=f'{name_prefix}_mha'
        )(x, x)
        
        # Add & Norm 1
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x1 = layers.Add()([x, attn_output])
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln1')(x1)
        
        # MLP Block
        mlp_dim = int(embed_dim * mlp_ratio)
        mlp_output = layers.Dense(mlp_dim, activation='gelu', name=f'{name_prefix}_mlp1')(x1)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        mlp_output = layers.Dense(embed_dim, name=f'{name_prefix}_mlp2')(mlp_output)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        
        # Add & Norm 2
        x2 = layers.Add()([x1, mlp_output])
        x2 = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln2')(x2)
        
        return x2
    
    # ============ CNN DECODER BLOCKS ============
    def decoder_block(x, skip, filters, upsample=True):
        """CNN decoder block with skip connection"""
        
        if upsample:
            x = layers.Conv2DTranspose(
                filters, 2, strides=2, padding='same', 
                kernel_initializer='he_normal'
            )(x)
        
        # Process skip connection if provided
        if skip is not None:
            # Resize skip connection to match x's spatial dimensions
            target_height = tf.shape(x)[1]
            target_width = tf.shape(x)[2]
            
            # Resize skip connection using Conv2DTranspose if needed
            skip_resized = layers.Conv2DTranspose(
                filters, 2, strides=2, padding='same',
                kernel_initializer='he_normal'
            )(skip)
            
            # Additional upsampling if still not matching
            current_height = tf.shape(skip_resized)[1] 
            while_condition = lambda h, w, s: tf.less(h, target_height)
            
            def upsample_more(skip_tensor):
                return layers.Conv2DTranspose(
                    filters, 2, strides=2, padding='same',
                    kernel_initializer='he_normal'
                )(skip_tensor)
            
            # Simple approach: match channels and concatenate
            skip_processed = layers.Conv2D(
                filters, 1, padding='same',
                kernel_initializer='he_normal'
            )(skip_resized)
            skip_processed = layers.BatchNormalization()(skip_processed)
            
            # Concatenate only if dimensions match, otherwise skip the skip connection
            try:
                x = layers.Concatenate()([x, skip_processed])
            except:
                # If dimensions still don't match, proceed without skip connection
                pass
        
        # Convolution block
        x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    # ============ BUILD TRANSFORMER ENCODER ============
    
    # Apply transformer layers with skip connections
    transformer_outputs = []
    x = encoded_patches
    
    for i in range(num_layers):
        x = transformer_block(
            x, num_heads, embed_dim, mlp_ratio, 
            dropout_rate=0.1, name_prefix=f'transformer_{i}'
        )
        
        # Collect outputs at specific layers for skip connections
        if i in [2, 5, 8, 11]:  # Layers 3, 6, 9, 12 (0-indexed)
            transformer_outputs.append(x)
    
    # ============ BUILD CNN DECODER WITH PROPER SKIP HANDLING ============
    
    # Reshape transformer outputs back to spatial format (all will be 16x16)
    skip_connections = []
    for i, trans_out in enumerate(transformer_outputs):
        # Reshape from (batch, num_patches, embed_dim) to (batch, patch_height, patch_width, embed_dim)
        reshaped = layers.Reshape((patch_height, patch_width, embed_dim))(trans_out)
        skip_connections.append(reshaped)
    
    # Start decoder from the last (deepest) transformer output
    # Level 1: 16x16 with 768 channels -> 16x16 with 512 channels
    current = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(skip_connections[-1])
    current = layers.BatchNormalization()(current)
    current = layers.Activation('relu')(current)
    
    # Level 2: 16x16 -> 32x32 (upsample + process skip from layer 9)
    current = layers.Conv2DTranspose(512, 2, strides=2, padding='same', kernel_initializer='he_normal')(current)
    
    # Process skip connection from layer 9 (also 16x16, needs upsampling to 32x32)
    skip_layer_9 = skip_connections[-2]  # Layer 9 output
    skip_layer_9_up = layers.Conv2DTranspose(256, 2, strides=2, padding='same', kernel_initializer='he_normal')(skip_layer_9)
    skip_layer_9_processed = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(skip_layer_9_up)
    skip_layer_9_processed = layers.BatchNormalization()(skip_layer_9_processed)
    
    # Reduce current channels to match
    current = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(current)
    current = layers.BatchNormalization()(current)
    
    # Concatenate
    current = layers.Concatenate()([current, skip_layer_9_processed])
    current = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(current)
    current = layers.BatchNormalization()(current)
    current = layers.Activation('relu')(current)
    
    # Level 3: 32x32 -> 64x64 (upsample + process skip from layer 6)
    current = layers.Conv2DTranspose(256, 2, strides=2, padding='same', kernel_initializer='he_normal')(current)
    
    # Process skip connection from layer 6 (16x16, needs 4x upsampling to 64x64)
    skip_layer_6 = skip_connections[-3]  # Layer 6 output
    skip_layer_6_up = layers.Conv2DTranspose(128, 2, strides=2, padding='same', kernel_initializer='he_normal')(skip_layer_6)  # 16->32
    skip_layer_6_up = layers.Conv2DTranspose(128, 2, strides=2, padding='same', kernel_initializer='he_normal')(skip_layer_6_up)  # 32->64
    skip_layer_6_processed = layers.Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(skip_layer_6_up)
    skip_layer_6_processed = layers.BatchNormalization()(skip_layer_6_processed)
    
    # Reduce current channels to match
    current = layers.Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(current)
    current = layers.BatchNormalization()(current)
    
    # Concatenate
    current = layers.Concatenate()([current, skip_layer_6_processed])
    current = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(current)
    current = layers.BatchNormalization()(current)
    current = layers.Activation('relu')(current)
    
    # Level 4: 64x64 -> 128x128 (upsample + process skip from layer 3)
    current = layers.Conv2DTranspose(128, 2, strides=2, padding='same', kernel_initializer='he_normal')(current)
    
    # Process skip connection from layer 3 (16x16, needs 8x upsampling to 128x128)
    skip_layer_3 = skip_connections[-4]  # Layer 3 output
    skip_layer_3_up = layers.Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer='he_normal')(skip_layer_3)  # 16->32
    skip_layer_3_up = layers.Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer='he_normal')(skip_layer_3_up)  # 32->64
    skip_layer_3_up = layers.Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer='he_normal')(skip_layer_3_up)  # 64->128
    skip_layer_3_processed = layers.Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(skip_layer_3_up)
    skip_layer_3_processed = layers.BatchNormalization()(skip_layer_3_processed)
    
    # Reduce current channels to match
    current = layers.Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(current)
    current = layers.BatchNormalization()(current)
    
    # Concatenate
    current = layers.Concatenate()([current, skip_layer_3_processed])
    current = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(current)
    current = layers.BatchNormalization()(current)
    current = layers.Activation('relu')(current)
    
    # Level 5: 128x128 -> 256x256 (final upsampling)
    current = layers.Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer='he_normal')(current)
    
    # ============ FINAL PROCESSING ============
    
    # Additional refinement layers
    final_features = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(current)
    final_features = layers.BatchNormalization()(final_features)
    final_features = layers.Activation('relu')(final_features)
    
    # Output layer
    if num_classes == 2:
        outputs = layers.Conv2D(
            1, 1, activation='sigmoid', 
            kernel_initializer='he_normal', 
            name='output_layer'
        )(final_features)
    else:
        outputs = layers.Conv2D(
            num_classes, 1, activation='softmax',
            kernel_initializer='he_normal', 
            name='output_layer'
        )(final_features)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='UNETR')
    
    return model

def build_unet(input_shape=(256, 256, 1), num_classes=4):
    """
    Enhanced U-Net architecture optimized for medical image segmentation
    
    Key improvements over basic U-Net:
    - Proper normalization strategy (BatchNorm after Conv, before activation)
    - Residual connections in deeper layers
    - Improved skip connections with attention gates
    - Optimized dropout scheduling
    - Better feature map progression
    - Spatial dropout for better regularization
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes (2 for binary, >2 for multi-class)
        
    Returns:
        keras.Model: Compiled U-Net model
    """
    inputs = Input(input_shape, name='input_layer')
    
    # ============ ENCODER PATH ============
    # Block 1: 256x256 -> 128x128
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = keras.layers.SpatialDropout2D(0.1)(pool1)
    
    # Block 2: 128x128 -> 64x64
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = keras.layers.SpatialDropout2D(0.1)(pool2)
    
    # Block 3: 64x64 -> 32x32
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = keras.layers.SpatialDropout2D(0.2)(pool3)
    
    # Block 4: 32x32 -> 16x16
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.Activation('relu')(conv4)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4_activated = keras.layers.Activation('relu')(conv4)
    
    # Add residual connection for deeper layers
    conv4_residual = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(pool3)
    conv4_residual = BatchNormalization()(conv4_residual)
    conv4_combined = keras.layers.Add()([conv4_activated, conv4_residual])
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_combined)
    pool4 = keras.layers.SpatialDropout2D(0.2)(pool4)
    
    # ============ BOTTLENECK ============
    # Block 5: 16x16 (bottleneck)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.Activation('relu')(conv5)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5_activated = keras.layers.Activation('relu')(conv5)
    
    # Bottleneck residual connection
    conv5_residual = Conv2D(1024, 1, padding='same', kernel_initializer='he_normal')(pool4)
    conv5_residual = BatchNormalization()(conv5_residual)
    conv5_combined = keras.layers.Add()([conv5_activated, conv5_residual])
    conv5_final = keras.layers.SpatialDropout2D(0.3)(conv5_combined)
    
    # ============ DECODER PATH ============
    # Block 6: 16x16 -> 32x32
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5_final)
    
    # Enhanced skip connection with feature matching
    skip4 = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(conv4_combined)
    skip4 = BatchNormalization()(skip4)
    merge6 = concatenate([up6, skip4], axis=3)
    
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.Activation('relu')(conv6)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.Activation('relu')(conv6)
    conv6 = keras.layers.SpatialDropout2D(0.2)(conv6)
    
    # Block 7: 32x32 -> 64x64
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    
    skip3 = Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(conv3)
    skip3 = BatchNormalization()(skip3)
    merge7 = concatenate([up7, skip3], axis=3)
    
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.Activation('relu')(conv7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.Activation('relu')(conv7)
    conv7 = keras.layers.SpatialDropout2D(0.2)(conv7)
    
    # Block 8: 64x64 -> 128x128
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    
    skip2 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(conv2)
    skip2 = BatchNormalization()(skip2)
    merge8 = concatenate([up8, skip2], axis=3)
    
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.Activation('relu')(conv8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.Activation('relu')(conv8)
    conv8 = keras.layers.SpatialDropout2D(0.1)(conv8)
    
    # Block 9: 128x128 -> 256x256
    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    
    skip1 = Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(conv1)
    skip1 = BatchNormalization()(skip1)
    merge9 = concatenate([up9, skip1], axis=3)
    
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = keras.layers.Activation('relu')(conv9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = keras.layers.Activation('relu')(conv9)
    conv9 = keras.layers.SpatialDropout2D(0.1)(conv9)
    
    # ============ OUTPUT LAYER ============
    # Final convolution layer for classification
    if num_classes == 2:
        # Binary segmentation - single output channel with sigmoid
        outputs = Conv2D(1, 1, activation='sigmoid', kernel_initializer='he_normal', name='output_layer')(conv9)
    else:
        # Multi-class segmentation - multiple channels with softmax
        outputs = Conv2D(num_classes, 1, activation='softmax', kernel_initializer='he_normal', name='output_layer')(conv9)
    
    # Create and return model
    model = Model(inputs=inputs, outputs=outputs, name='U-Net')
    
    return model

def build_swinunetr(input_shape=(256, 256, 1), num_classes=4):
    """
    SwinUNETR implementation for medical image segmentation
    
    Architecture:
    - Swin Transformer encoder with hierarchical feature extraction
    - Shifted window attention for efficient computation
    - CNN decoder with multi-scale skip connections
    - Progressive feature resolution: 64x64 -> 32x32 -> 16x16 -> 8x8
    
    Based on: Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images" (2022)
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        keras.Model: SwinUNETR model
    """
    
    # Hyperparameters
    patch_size = 4      # 4x4 patches for initial embedding
    window_size = 8     # Window size for attention
    embed_dim = 96      # Base embedding dimension
    depths = [2, 2, 6, 2]  # Number of blocks in each stage
    num_heads = [3, 6, 12, 24]  # Attention heads in each stage
    mlp_ratio = 4       # MLP expansion ratio
    
    inputs = Input(input_shape, name='input_layer')
    
    # ============ PATCH EMBEDDING ============
    def patch_embed(x, embed_dim, patch_size):
        """Initial patch embedding layer"""
        x = layers.Conv2D(
            embed_dim, 
            kernel_size=patch_size, 
            strides=patch_size,
            padding='valid',
            kernel_initializer='he_normal',
            name='patch_embed'
        )(x)
        return x
    
    # ============ FIXED PATCH MERGING ============  
    def patch_merging_fixed(x, output_dim, name_prefix):
        """Fixed patch merging with static operations"""
        # Use Conv2D with stride=2 for downsampling and channel expansion
        # This avoids dynamic padding issues
        x = layers.Conv2D(
            output_dim, 
            kernel_size=2, 
            strides=2, 
            padding='same',
            kernel_initializer='he_normal',
            name=f'{name_prefix}_merge_conv'
        )(x)
        x = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_norm')(x)
        return x
    
    # ============ SIMPLIFIED TRANSFORMER BLOCK ============
    def swin_transformer_block(x, dim, num_heads, window_size, shift_size, mlp_ratio, name_prefix):
        """Simplified Swin Transformer Block with static shapes"""
        shortcut = x
        
        # Layer Norm
        x = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_norm1')(x)
        
        # Get static shape information
        input_shape_static = x.shape
        if input_shape_static[1] is not None and input_shape_static[2] is not None:
            H, W = int(input_shape_static[1]), int(input_shape_static[2])
        else:
            # Fallback for dynamic shapes - use simpler attention
            H, W = 64, 64  # Default assumption based on patch embedding
        
        # Reshape to sequence format for attention
        x_seq = layers.Reshape((-1, dim), name=f'{name_prefix}_reshape_to_seq')(x)
        
        # Multi-head attention (simplified - no windowing to avoid dynamic shape issues)
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=0.1,
            name=f'{name_prefix}_attn'
        )(x_seq, x_seq)
        
        # Reshape back to spatial format
        attn_output = layers.Reshape((H, W, dim), name=f'{name_prefix}_reshape_to_spatial')(attn_output)
        
        # Add shortcut
        x = layers.Add(name=f'{name_prefix}_add1')([shortcut, attn_output])
        
        # MLP Block
        shortcut = x
        x = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_norm2')(x)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        x = layers.Dense(mlp_hidden_dim, activation='gelu', name=f'{name_prefix}_mlp1')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(dim, name=f'{name_prefix}_mlp2')(x)
        x = layers.Dropout(0.1)(x)
        
        # Add shortcut
        x = layers.Add(name=f'{name_prefix}_add2')([shortcut, x])
        
        return x
    
    # ============ BASIC LAYER ============
    def basic_layer(x, dim, depth, num_heads, window_size, mlp_ratio, downsample_dim, name_prefix):
        """Basic Swin Transformer layer with fixed operations"""
        
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            x = swin_transformer_block(
                x, dim, num_heads, window_size, shift_size, mlp_ratio,
                name_prefix=f'{name_prefix}_block{i}'
            )
        
        # Store skip connection before downsampling
        skip = x
        
        # Downsample if needed
        if downsample_dim is not None:
            x = patch_merging_fixed(x, downsample_dim, name_prefix=f'{name_prefix}_downsample')
            
        return x, skip
    
    # ============ CNN DECODER ============
    def decoder_block(x, skip, filters, upsample=True, name_prefix="decoder"):
        """CNN decoder block with proper naming"""
        if upsample:
            x = layers.Conv2DTranspose(
                filters, 2, strides=2, padding='same',
                kernel_initializer='he_normal',
                name=f'{name_prefix}_upsample'
            )(x)
        
        if skip is not None:
            # Match channels
            skip = layers.Conv2D(
                filters, 1, padding='same',
                kernel_initializer='he_normal',
                name=f'{name_prefix}_skip_conv'
            )(skip)
            skip = layers.BatchNormalization(name=f'{name_prefix}_skip_bn')(skip)
            
            # Concatenate
            x = layers.Concatenate(name=f'{name_prefix}_concat')([x, skip])
        
        # Convolution block
        x = layers.Conv2D(filters, 3, padding='same', 
                         kernel_initializer='he_normal', 
                         name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        
        x = layers.Conv2D(filters, 3, padding='same', 
                         kernel_initializer='he_normal',
                         name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
        
        return x
    
    # ============ BUILD SWIN TRANSFORMER ENCODER ============
    
    # Initial patch embedding (256x256 -> 64x64)
    x = patch_embed(inputs, embed_dim, patch_size)
    x = layers.LayerNormalization(epsilon=1e-5, name='embed_norm')(x)
    
    # Store skip connections
    skip_connections = []
    
    # Stage 1: 64x64, 96 dims
    x, skip1 = basic_layer(
        x, embed_dim, depths[0], num_heads[0], window_size, mlp_ratio,
        downsample_dim=embed_dim*2, name_prefix='stage1'
    )
    skip_connections.append(skip1)  # 64x64, 96 dims
    # x is now 32x32, 192 dims
    
    # Stage 2: 32x32, 192 dims  
    x, skip2 = basic_layer(
        x, embed_dim*2, depths[1], num_heads[1], window_size, mlp_ratio,
        downsample_dim=embed_dim*4, name_prefix='stage2'
    )
    skip_connections.append(skip2)  # 32x32, 192 dims
    # x is now 16x16, 384 dims
    
    # Stage 3: 16x16, 384 dims
    x, skip3 = basic_layer(
        x, embed_dim*4, depths[2], num_heads[2], window_size, mlp_ratio,
        downsample_dim=embed_dim*8, name_prefix='stage3'
    )
    skip_connections.append(skip3)  # 16x16, 384 dims
    # x is now 8x8, 768 dims
    
    # Stage 4: 8x8, 768 dims (no downsampling)
    x, _ = basic_layer(
        x, embed_dim*8, depths[3], num_heads[3], window_size, mlp_ratio,
        downsample_dim=None, name_prefix='stage4'
    )
    # x is still 8x8, 768 dims
    
    # ============ BUILD CNN DECODER ============
    
    # Initial processing of bottleneck
    x = layers.Conv2D(512, 3, padding='same', 
                     kernel_initializer='he_normal', 
                     name='bottleneck_conv')(x)
    x = layers.BatchNormalization(name='bottleneck_bn')(x)
    x = layers.Activation('relu', name='bottleneck_relu')(x)
    
    # Decoder Level 1: 8x8 -> 16x16
    x = decoder_block(x, skip_connections[2], 256, upsample=True, name_prefix="decoder1")
    
    # Decoder Level 2: 16x16 -> 32x32
    x = decoder_block(x, skip_connections[1], 128, upsample=True, name_prefix="decoder2")
    
    # Decoder Level 3: 32x32 -> 64x64
    x = decoder_block(x, skip_connections[0], 64, upsample=True, name_prefix="decoder3")
    
    # Decoder Level 4: 64x64 -> 128x128
    x = decoder_block(x, None, 32, upsample=True, name_prefix="decoder4")
    
    # Decoder Level 5: 128x128 -> 256x256
    x = decoder_block(x, None, 16, upsample=True, name_prefix="decoder5")
    
    # ============ FINAL PROCESSING ============
    
    # Final refinement
    x = layers.Conv2D(32, 3, padding='same', 
                     kernel_initializer='he_normal', 
                     name='final_conv1')(x)
    x = layers.BatchNormalization(name='final_bn1')(x)
    x = layers.Activation('relu', name='final_relu1')(x)
    
    # Output layer
    if num_classes == 2:
        outputs = layers.Conv2D(
            1, 1, activation='sigmoid',
            kernel_initializer='he_normal',
            name='output_layer'
        )(x)
    else:
        outputs = layers.Conv2D(
            num_classes, 1, activation='softmax',
            kernel_initializer='he_normal', 
            name='output_layer'
        )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='SwinUNETR')
    
    return model

def build_swinunetr_light(input_shape=(256, 256, 1), num_classes=4):
    """
    Lightweight SwinUNETR with reduced parameters and fixed shapes
    
    Useful for:
    - Faster prototyping
    - Resource-constrained environments
    - Ablation studies
    """
    
    # Reduced hyperparameters
    patch_size = 4
    embed_dim = 48      # Reduced from 96
    depths = [1, 1, 2, 1]  # Reduced depth
    num_heads = [2, 4, 8, 16]  # Reduced heads
    
    inputs = Input(input_shape, name='input_layer')
    
    # Initial patch embedding (256x256 -> 64x64)
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, 
                     padding='valid', kernel_initializer='he_normal',
                     name='patch_embed')(inputs)
    
    skip_connections = []
    
    # Stage 1: 64x64, 48 dims
    for i in range(depths[0]):
        # Simplified transformer block
        shortcut = x
        x_seq = layers.Reshape((-1, embed_dim), name=f'stage1_reshape_seq_{i}')(x)
        attn = layers.MultiHeadAttention(
            num_heads[0], embed_dim//num_heads[0],
            name=f'stage1_attn_{i}'
        )(x_seq, x_seq)
        x = layers.Reshape((64, 64, embed_dim), name=f'stage1_reshape_spatial_{i}')(attn)
        x = layers.Add(name=f'stage1_add_{i}')([shortcut, x])
        x = layers.LayerNormalization(name=f'stage1_norm_{i}')(x)
    
    skip_connections.append(x)  # 64x64, 48 dims
    x = layers.Conv2D(embed_dim*2, 2, strides=2, padding='same',
                     name='stage1_downsample')(x)  # 32x32, 96 dims
    
    # Stage 2: 32x32, 96 dims
    for i in range(depths[1]):
        shortcut = x
        x_seq = layers.Reshape((-1, embed_dim*2), name=f'stage2_reshape_seq_{i}')(x)
        attn = layers.MultiHeadAttention(
            num_heads[1], embed_dim*2//num_heads[1],
            name=f'stage2_attn_{i}'
        )(x_seq, x_seq)
        x = layers.Reshape((32, 32, embed_dim*2), name=f'stage2_reshape_spatial_{i}')(attn)
        x = layers.Add(name=f'stage2_add_{i}')([shortcut, x])
        x = layers.LayerNormalization(name=f'stage2_norm_{i}')(x)
        
    skip_connections.append(x)  # 32x32, 96 dims
    x = layers.Conv2D(embed_dim*4, 2, strides=2, padding='same',
                     name='stage2_downsample')(x)  # 16x16, 192 dims
    
    # Stage 3: 16x16, 192 dims
    for i in range(depths[2]):
        shortcut = x
        x_seq = layers.Reshape((-1, embed_dim*4), name=f'stage3_reshape_seq_{i}')(x)
        attn = layers.MultiHeadAttention(
            num_heads[2], embed_dim*4//num_heads[2],
            name=f'stage3_attn_{i}'
        )(x_seq, x_seq)
        x = layers.Reshape((16, 16, embed_dim*4), name=f'stage3_reshape_spatial_{i}')(attn)
        x = layers.Add(name=f'stage3_add_{i}')([shortcut, x])
        x = layers.LayerNormalization(name=f'stage3_norm_{i}')(x)
        
    skip_connections.append(x)  # 16x16, 192 dims
    x = layers.Conv2D(embed_dim*8, 2, strides=2, padding='same',
                     name='stage3_downsample')(x)  # 8x8, 384 dims
    
    # Stage 4: 8x8, 384 dims (bottleneck)
    for i in range(depths[3]):
        shortcut = x
        x_seq = layers.Reshape((-1, embed_dim*8), name=f'stage4_reshape_seq_{i}')(x)
        attn = layers.MultiHeadAttention(
            num_heads[3], embed_dim*8//num_heads[3],
            name=f'stage4_attn_{i}'
        )(x_seq, x_seq)
        x = layers.Reshape((8, 8, embed_dim*8), name=f'stage4_reshape_spatial_{i}')(attn)
        x = layers.Add(name=f'stage4_add_{i}')([shortcut, x])
        x = layers.LayerNormalization(name=f'stage4_norm_{i}')(x)
    
    # Simple decoder with proper naming
    # 8x8 -> 16x16
    x = layers.Conv2DTranspose(192, 2, strides=2, padding='same', 
                              name='decoder1_upsample')(x)
    skip_processed = layers.Conv2D(192, 1, padding='same', 
                                  name='decoder1_skip_conv')(skip_connections[2])
    x = layers.Concatenate(name='decoder1_concat')([x, skip_processed])
    x = layers.Conv2D(192, 3, padding='same', activation='relu', 
                     name='decoder1_conv')(x)
    
    # 16x16 -> 32x32
    x = layers.Conv2DTranspose(96, 2, strides=2, padding='same',
                              name='decoder2_upsample')(x)
    skip_processed = layers.Conv2D(96, 1, padding='same',
                                  name='decoder2_skip_conv')(skip_connections[1])
    x = layers.Concatenate(name='decoder2_concat')([x, skip_processed])
    x = layers.Conv2D(96, 3, padding='same', activation='relu',
                     name='decoder2_conv')(x)
    
    # 32x32 -> 64x64
    x = layers.Conv2DTranspose(48, 2, strides=2, padding='same',
                              name='decoder3_upsample')(x)
    skip_processed = layers.Conv2D(48, 1, padding='same',
                                  name='decoder3_skip_conv')(skip_connections[0])
    x = layers.Concatenate(name='decoder3_concat')([x, skip_processed])
    x = layers.Conv2D(48, 3, padding='same', activation='relu',
                     name='decoder3_conv')(x)
    
    # 64x64 -> 128x128
    x = layers.Conv2DTranspose(24, 2, strides=2, padding='same',
                              name='decoder4_upsample')(x)
    x = layers.Conv2D(24, 3, padding='same', activation='relu',
                     name='decoder4_conv')(x)
    
    # 128x128 -> 256x256
    x = layers.Conv2DTranspose(12, 2, strides=2, padding='same',
                              name='decoder5_upsample')(x)
    x = layers.Conv2D(12, 3, padding='same', activation='relu',
                     name='decoder5_conv')(x)
    
    # Output
    if num_classes == 2:
        outputs = layers.Conv2D(1, 1, activation='sigmoid', 
                               name='output_layer')(x)
    else:
        outputs = layers.Conv2D(num_classes, 1, activation='softmax',
                               name='output_layer')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='SwinUNETR_Light')
    return model

def build_unetplusplus(input_shape=(256, 256, 1), num_classes=4, deep_supervision=False):
    """
    UNet++ (Nested U-Net) implementation for medical image segmentation
    
    Based on: Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018)
    Paper: https://arxiv.org/abs/1807.10165
    
    Key features:
    - Nested skip pathways with dense connections
    - Redesigned skip connections to reduce semantic gap
    - Multiple decoder branches at different depths
    - Optional deep supervision for better gradient flow
    - Progressive feature aggregation
    
    Architecture:
    - 5 encoder levels (X^0,0 to X^0,4)
    - Dense nested connections between encoder and decoder
    - Skip pathways: X^i,j where i=decoder level, j=skip pathway position
    - Final outputs can be taken from multiple decoder levels
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        deep_supervision: Whether to output multiple segmentation maps for deep supervision
        
    Returns:
        keras.Model: UNet++ model
    """
    
    inputs = Input(input_shape, name='input_layer')
    
    # Filter configuration for each level
    filters = [32, 64, 128, 256, 512]
    
    # ============ STANDARD CONVOLUTION BLOCK ============
    def conv_block(x, filters, stage, block, kernel_size=3, dropout_rate=0.1):
        """Standard convolution block used throughout UNet++"""
        block_name = f'X{stage}_{block}'
        
        x = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_initializer='he_normal',
            name=f'{block_name}_conv1'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
        x = layers.Activation('relu', name=f'{block_name}_relu1')(x)
        x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout1')(x)
        
        x = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_initializer='he_normal',
            name=f'{block_name}_conv2'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
        x = layers.Activation('relu', name=f'{block_name}_relu2')(x)
        x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout2')(x)
        
        return x
    
    # ============ BUILD NESTED U-NET ARCHITECTURE ============
    
    # Initialize node storage for nested connections
    # nodes[i][j] represents node X^i,j in the paper notation
    # i = decoder level (0=deepest, 4=output level)
    # j = skip pathway position (0=encoder backbone, increasing towards output)
    nodes = {}
    
    # Level 0 (Encoder backbone): X^0,0 to X^0,4
    # These are the traditional U-Net encoder nodes
    
    # X^0,0: First encoder block (256x256)
    nodes[0] = {}
    nodes[0][0] = conv_block(inputs, filters[0], stage=0, block=0)
    
    # X^0,1: Second encoder block (128x128)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(nodes[0][0])
    nodes[0][1] = conv_block(pool1, filters[1], stage=0, block=1)
    
    # X^0,2: Third encoder block (64x64)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(nodes[0][1])
    nodes[0][2] = conv_block(pool2, filters[2], stage=0, block=2)
    
    # X^0,3: Fourth encoder block (32x32)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(nodes[0][2])
    nodes[0][3] = conv_block(pool3, filters[3], stage=0, block=3)
    
    # X^0,4: Bottleneck (16x16)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(nodes[0][3])
    nodes[0][4] = conv_block(pool4, filters[4], stage=0, block=4)
    
    # Level 1: First decoder level with nested connections
    nodes[1] = {}
    
    # X^1,0: 128x128, connected from X^0,0 and upsampled X^0,1
    up1_0 = layers.Conv2DTranspose(
        filters[0], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up1_0'
    )(nodes[0][1])
    concat1_0 = layers.Concatenate(name='concat1_0')([nodes[0][0], up1_0])
    nodes[1][0] = conv_block(concat1_0, filters[0], stage=1, block=0)
    
    # X^1,1: 64x64, connected from X^0,1 and upsampled X^0,2
    up1_1 = layers.Conv2DTranspose(
        filters[1], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up1_1'
    )(nodes[0][2])
    concat1_1 = layers.Concatenate(name='concat1_1')([nodes[0][1], up1_1])
    nodes[1][1] = conv_block(concat1_1, filters[1], stage=1, block=1)
    
    # X^1,2: 32x32, connected from X^0,2 and upsampled X^0,3
    up1_2 = layers.Conv2DTranspose(
        filters[2], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up1_2'
    )(nodes[0][3])
    concat1_2 = layers.Concatenate(name='concat1_2')([nodes[0][2], up1_2])
    nodes[1][2] = conv_block(concat1_2, filters[2], stage=1, block=2)
    
    # X^1,3: 16x16, connected from X^0,3 and upsampled X^0,4
    up1_3 = layers.Conv2DTranspose(
        filters[3], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up1_3'
    )(nodes[0][4])
    concat1_3 = layers.Concatenate(name='concat1_3')([nodes[0][3], up1_3])
    nodes[1][3] = conv_block(concat1_3, filters[3], stage=1, block=3)
    
    # Level 2: Second decoder level with denser connections
    nodes[2] = {}
    
    # X^2,0: 128x128, connected from X^0,0, X^1,0, and upsampled X^1,1
    up2_0 = layers.Conv2DTranspose(
        filters[0], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up2_0'
    )(nodes[1][1])
    concat2_0 = layers.Concatenate(name='concat2_0')([nodes[0][0], nodes[1][0], up2_0])
    nodes[2][0] = conv_block(concat2_0, filters[0], stage=2, block=0)
    
    # X^2,1: 64x64, connected from X^0,1, X^1,1, and upsampled X^1,2
    up2_1 = layers.Conv2DTranspose(
        filters[1], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up2_1'
    )(nodes[1][2])
    concat2_1 = layers.Concatenate(name='concat2_1')([nodes[0][1], nodes[1][1], up2_1])
    nodes[2][1] = conv_block(concat2_1, filters[1], stage=2, block=1)
    
    # X^2,2: 32x32, connected from X^0,2, X^1,2, and upsampled X^1,3
    up2_2 = layers.Conv2DTranspose(
        filters[2], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up2_2'
    )(nodes[1][3])
    concat2_2 = layers.Concatenate(name='concat2_2')([nodes[0][2], nodes[1][2], up2_2])
    nodes[2][2] = conv_block(concat2_2, filters[2], stage=2, block=2)
    
    # Level 3: Third decoder level
    nodes[3] = {}
    
    # X^3,0: 128x128, connected from X^0,0, X^1,0, X^2,0, and upsampled X^2,1
    up3_0 = layers.Conv2DTranspose(
        filters[0], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up3_0'
    )(nodes[2][1])
    concat3_0 = layers.Concatenate(name='concat3_0')([nodes[0][0], nodes[1][0], nodes[2][0], up3_0])
    nodes[3][0] = conv_block(concat3_0, filters[0], stage=3, block=0)
    
    # X^3,1: 64x64, connected from X^0,1, X^1,1, X^2,1, and upsampled X^2,2
    up3_1 = layers.Conv2DTranspose(
        filters[1], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up3_1'
    )(nodes[2][2])
    concat3_1 = layers.Concatenate(name='concat3_1')([nodes[0][1], nodes[1][1], nodes[2][1], up3_1])
    nodes[3][1] = conv_block(concat3_1, filters[1], stage=3, block=1)
    
    # Level 4: Final decoder level (output level)
    nodes[4] = {}
    
    # X^4,0: Final output 256x256, connected from all previous levels at position 0
    up4_0 = layers.Conv2DTranspose(
        filters[0], 2, strides=2, padding='same',
        kernel_initializer='he_normal', name='up4_0'
    )(nodes[3][1])
    concat4_0 = layers.Concatenate(name='concat4_0')([
        nodes[0][0], nodes[1][0], nodes[2][0], nodes[3][0], up4_0
    ])
    nodes[4][0] = conv_block(concat4_0, filters[0], stage=4, block=0)
    
    # ============ OUTPUT LAYERS ============
    
    if deep_supervision:
        # Multiple outputs for deep supervision
        outputs = []
        
        # Output from level 1 (X^1,0) - 128x128
        out1 = layers.Conv2D(num_classes, 1, activation='softmax' if num_classes > 2 else 'sigmoid', 
                           kernel_initializer='he_normal', name='output_1')(nodes[1][0])
        out1_upsampled = layers.UpSampling2D(size=(2, 2), name='output_1_upsampled')(out1)
        outputs.append(out1_upsampled)
        
        # Output from level 2 (X^2,0) - 128x128  
        out2 = layers.Conv2D(num_classes, 1, activation='softmax' if num_classes > 2 else 'sigmoid',
                           kernel_initializer='he_normal', name='output_2')(nodes[2][0])
        out2_upsampled = layers.UpSampling2D(size=(2, 2), name='output_2_upsampled')(out2)
        outputs.append(out2_upsampled)
        
        # Output from level 3 (X^3,0) - 128x128
        out3 = layers.Conv2D(num_classes, 1, activation='softmax' if num_classes > 2 else 'sigmoid',
                           kernel_initializer='he_normal', name='output_3')(nodes[3][0])
        out3_upsampled = layers.UpSampling2D(size=(2, 2), name='output_3_upsampled')(out3)
        outputs.append(out3_upsampled)
        
        # Main output from level 4 (X^4,0) - 256x256
        main_output = layers.Conv2D(
            num_classes, 1, 
            activation='softmax' if num_classes > 2 else 'sigmoid',
            kernel_initializer='he_normal', 
            name='main_output'
        )(nodes[4][0])
        outputs.append(main_output)
        
        model = Model(inputs=inputs, outputs=outputs, name='UNetPlusPlus_DeepSupervision')
        
    else:
        # Single main output
        main_output = layers.Conv2D(
            1 if num_classes == 2 else num_classes, 1,
            activation='sigmoid' if num_classes == 2 else 'softmax',
            kernel_initializer='he_normal',
            name='output_layer'
        )(nodes[4][0])
        
        model = Model(inputs=inputs, outputs=main_output, name='UNetPlusPlus')
    
    return model

def build_unetplusplus_light(input_shape=(256, 256, 1), num_classes=4):
    """
    Lightweight UNet++ with reduced parameters for faster training/inference
    
    Uses fewer filters and simpler connections while maintaining the nested architecture
    """
    
    inputs = Input(input_shape, name='input_layer')
    
    # Reduced filter configuration
    filters = [16, 32, 64, 128, 256]
    
    def conv_block_light(x, filters, stage, block):
        """Lighter convolution block"""
        block_name = f'X{stage}_{block}'
        
        x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
                         name=f'{block_name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
        x = layers.Activation('relu', name=f'{block_name}_relu1')(x)
        
        x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
                         name=f'{block_name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
        x = layers.Activation('relu', name=f'{block_name}_relu2')(x)
        
        return x
    
    # Simplified nested architecture (3 levels instead of 5)
    nodes = {}
    
    # Level 0: Encoder
    nodes[0] = {}
    nodes[0][0] = conv_block_light(inputs, filters[0], 0, 0)
    
    pool1 = layers.MaxPooling2D(2, name='pool1')(nodes[0][0])
    nodes[0][1] = conv_block_light(pool1, filters[1], 0, 1)
    
    pool2 = layers.MaxPooling2D(2, name='pool2')(nodes[0][1])
    nodes[0][2] = conv_block_light(pool2, filters[2], 0, 2)
    
    # Level 1: First decoder
    nodes[1] = {}
    up1_0 = layers.Conv2DTranspose(filters[0], 2, strides=2, padding='same', name='up1_0')(nodes[0][1])
    concat1_0 = layers.Concatenate(name='concat1_0')([nodes[0][0], up1_0])
    nodes[1][0] = conv_block_light(concat1_0, filters[0], 1, 0)
    
    up1_1 = layers.Conv2DTranspose(filters[1], 2, strides=2, padding='same', name='up1_1')(nodes[0][2])
    concat1_1 = layers.Concatenate(name='concat1_1')([nodes[0][1], up1_1])
    nodes[1][1] = conv_block_light(concat1_1, filters[1], 1, 1)
    
    # Level 2: Final decoder
    nodes[2] = {}
    up2_0 = layers.Conv2DTranspose(filters[0], 2, strides=2, padding='same', name='up2_0')(nodes[1][1])
    concat2_0 = layers.Concatenate(name='concat2_0')([nodes[0][0], nodes[1][0], up2_0])
    nodes[2][0] = conv_block_light(concat2_0, filters[0], 2, 0)
    
    # Output
    if num_classes == 2:
        output = layers.Conv2D(1, 1, activation='sigmoid', name='output_layer')(nodes[2][0])
    else:
        output = layers.Conv2D(num_classes, 1, activation='softmax', name='output_layer')(nodes[2][0])
    
    model = Model(inputs=inputs, outputs=output, name='UNetPlusPlus_Light')
    return model

def unetplusplus_loss_function(class_weights=None, deep_supervision=True):
    """
    Loss function for UNet++ with optional deep supervision
    
    Combines Dice loss and Cross-entropy loss with weights for auxiliary outputs
    """
    
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - dice_coef
    
    def combined_loss(y_true, y_pred):
        dice = dice_loss(y_true, y_pred)
        ce = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred))
        return dice + ce
    
    if deep_supervision:
        def deep_supervision_loss(y_true, y_pred_list):
            # Main output gets full weight, auxiliary outputs get reduced weights
            main_loss = combined_loss(y_true, y_pred_list[-1])  # Last output is main
            
            aux_weights = [0.4, 0.3, 0.2]  # Weights for auxiliary outputs
            aux_loss = 0
            
            for i, pred in enumerate(y_pred_list[:-1]):  # All except main output
                if i < len(aux_weights):
                    aux_loss += aux_weights[i] * combined_loss(y_true, pred)
            
            return main_loss + aux_loss
        
        return deep_supervision_loss
    else:
        return combined_loss

def get_model_builder(model_name):
    """Factory function to get model builder by name"""
    builders = {
        'U-Net': build_unet,
        'UNet++': build_unetplusplus,
        'UNETR': build_unetr,
        'SwinUNETR': build_swinunetr,
    }
    return builders.get(model_name)
