import numpy as np
import mne
from typing import Tuple, Optional, List, Dict, Any
from ..eeg_obj import EEG
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import pickle
import os


class ImprovedManifoldAutoencoder:
    """
    Improved Manifold Autoencoder with attention mechanisms, multi-scale features,
    and advanced loss functions for better artifact estimation accuracy.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        encoder_filters: List[int] = [64, 128, 256],
        decoder_filters: List[int] = [256, 128, 64],
        kernel_sizes: List[int] = [3, 5, 7],  # Multi-scale kernels
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-5,
        learning_rate: float = 1e-4,
        use_attention: bool = True,
        use_residuals: bool = True,
        use_spectral_loss: bool = True
    ):
        """
        Initialize the Improved Manifold Autoencoder.
        
        Args:
            input_shape: Shape of input data (channels, timepoints)
            latent_dim: Dimension of the latent space
            encoder_filters: Number of filters for each encoder layer
            decoder_filters: Number of filters for each decoder layer
            kernel_sizes: List of kernel sizes for multi-scale convolutions
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            learning_rate: Learning rate for the optimizer
            use_attention: Whether to use attention mechanisms
            use_residuals: Whether to use residual connections
            use_spectral_loss: Whether to include spectral loss component
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        self.use_residuals = use_residuals
        self.use_spectral_loss = use_spectral_loss
        
        # Normalization parameters
        self.channel_means = None
        self.channel_stds = None
        self.artifact_channel_means = None
        self.artifact_channel_stds = None
        
        # Build model components
        self.encoder_output_shape = self._calculate_encoder_output_shape()
        self.encoder = self._build_improved_encoder()
        self.decoder = self._build_improved_decoder()
        self.autoencoder = self._build_autoencoder()
        
        # Optional: Build discriminator for adversarial training
        self.discriminator = None
        self.use_adversarial = False
    
    def _multi_scale_conv_block(self, x, filters, strides=(1, 1)):
        """Multi-scale convolutional block with different kernel sizes."""
        conv_outputs = []
        
        for kernel_size in self.kernel_sizes:
            conv = layers.Conv2D(
                filters // len(self.kernel_sizes),
                (kernel_size, kernel_size),
                strides=strides,
                padding='same',
                kernel_regularizer=l2(self.l2_reg)
            )(x)
            conv_outputs.append(conv)
        
        # Concatenate multi-scale features
        if len(conv_outputs) > 1:
            x = layers.Concatenate()(conv_outputs)
        else:
            x = conv_outputs[0]
        
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x
    
    def _attention_block(self, features, name_prefix=""):
        """Self-attention mechanism for focusing on important features."""
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(features)
        max_pool = layers.GlobalMaxPooling2D()(features)
        
        # Shared MLP
        dense1 = layers.Dense(features.shape[-1] // 8, activation='relu', name=f"{name_prefix}_att_dense1")
        dense2 = layers.Dense(features.shape[-1], name=f"{name_prefix}_att_dense2")
        
        avg_out = dense2(dense1(avg_pool))
        max_out = dense2(dense1(max_pool))
        
        channel_attention = layers.Activation('sigmoid')(avg_out + max_out)
        channel_attention = layers.Reshape((1, 1, features.shape[-1]))(channel_attention)
        
        # Apply channel attention
        features = layers.Multiply()([features, channel_attention])
        
        # Spatial attention
        avg_pool_spatial = tf.reduce_mean(features, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(features, axis=-1, keepdims=True)
        spatial_concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        
        spatial_attention = layers.Conv2D(
            1, (7, 7), 
            padding='same', 
            activation='sigmoid',
            name=f"{name_prefix}_spatial_conv"
        )(spatial_concat)
        
        # Apply spatial attention
        features = layers.Multiply()([features, spatial_attention])
        
        return features
    
    def _calculate_encoder_output_shape(self) -> Tuple[int, int, int]:
        """Calculate the shape of the encoder output before flattening."""
        test_input = layers.Input(shape=self.input_shape)
        x = layers.Reshape((*self.input_shape, 1))(test_input)
        
        # Apply the same convolutions as in the encoder
        for i, filters in enumerate(self.encoder_filters):
            x = layers.Conv2D(
                filters,
                (3, 3),  # Use fixed kernel for shape calculation
                strides=(2, 2),
                padding='same'
            )(x)
        
        temp_model = Model(test_input, x)
        output_shape = temp_model.output_shape[1:]
        return output_shape
    
    def _build_improved_encoder(self) -> Model:
        """Build the improved encoder network with attention and residuals."""
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Reshape((*self.input_shape, 1))(inputs)
        
        encoder_features = []
        
        for i, filters in enumerate(self.encoder_filters):
            # Store for skip connections
            if self.use_residuals and i > 0:
                residual = x
            
            # Multi-scale convolution
            x = self._multi_scale_conv_block(x, filters, strides=(2, 2))
            
            # Apply attention if enabled
            if self.use_attention:
                x = self._attention_block(x, name_prefix=f"encoder_block{i}")
            
            # Residual connection
            if self.use_residuals and i > 0:
                # Adjust residual dimensions if needed
                if residual.shape[1:3] != x.shape[1:3]:
                    residual = layers.Conv2D(filters, (1, 1), strides=(2, 2))(residual)
                if residual.shape[-1] != x.shape[-1]:
                    residual = layers.Conv2D(filters, (1, 1))(residual)
                x = layers.Add()([x, residual])
            
            x = layers.Dropout(self.dropout_rate)(x)
            encoder_features.append(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim * 2, kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Final latent representation
        latent = layers.Dense(self.latent_dim, name='latent')(x)
        
        return Model(inputs, latent, name='improved_encoder')
    
    def _build_improved_decoder(self) -> Model:
        """Build the improved decoder network."""
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        
        # Expand from latent space
        x = layers.Dense(self.latent_dim * 2)(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Reconstruct the feature map
        x = layers.Dense(np.prod(self.encoder_output_shape))(x)
        x = layers.Reshape(self.encoder_output_shape)(x)
        
        # Decoder layers with transpose convolutions
        for i, filters in enumerate(self.decoder_filters):
            # Multi-scale transpose convolution
            x = self._multi_scale_transpose_conv_block(x, filters)
            
            # Apply attention if enabled
            if self.use_attention:
                x = self._attention_block(x, name_prefix=f"decoder_block{i}")
            
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Final layer to get to single channel
        x = layers.Conv2DTranspose(
            1,
            (3, 3),
            strides=(1, 1),
            padding='same',
            activation='linear'
        )(x)
        
        # Ensure output matches input shape
        x = self._adjust_output_shape(x, self.input_shape)
        x = layers.Reshape(self.input_shape)(x)
        
        return Model(latent_inputs, x, name='improved_decoder')
    
    def _multi_scale_transpose_conv_block(self, x, filters):
        """Multi-scale transpose convolutional block."""
        conv_outputs = []
        
        for kernel_size in self.kernel_sizes:
            conv = layers.Conv2DTranspose(
                filters // len(self.kernel_sizes),
                (kernel_size, kernel_size),
                strides=(2, 2),
                padding='same',
                kernel_regularizer=l2(self.l2_reg)
            )(x)
            conv_outputs.append(conv)
        
        if len(conv_outputs) > 1:
            x = layers.Concatenate()(conv_outputs)
        else:
            x = conv_outputs[0]
        
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x
    
    def _adjust_output_shape(self, x, target_shape):
        """Adjust output shape to match target shape using cropping or padding."""
        current_shape = x.shape[1:3]
        
        if current_shape[0] != target_shape[0] or current_shape[1] != target_shape[1]:
            height_diff = current_shape[0] - target_shape[0]
            width_diff = current_shape[1] - target_shape[1]
            
            if height_diff > 0 or width_diff > 0:
                # Crop
                crop_top = height_diff // 2
                crop_bottom = height_diff - crop_top
                crop_left = width_diff // 2
                crop_right = width_diff - crop_left
                x = layers.Cropping2D(
                    cropping=((crop_top, crop_bottom), (crop_left, crop_right))
                )(x)
            elif height_diff < 0 or width_diff < 0:
                # Pad
                pad_top = abs(height_diff) // 2
                pad_bottom = abs(height_diff) - pad_top
                pad_left = abs(width_diff) // 2
                pad_right = abs(width_diff) - pad_left
                x = layers.ZeroPadding2D(
                    padding=((pad_top, pad_bottom), (pad_left, pad_right))
                )(x)
        
        return x
    
    def _build_autoencoder(self) -> Model:
        """Build the complete autoencoder."""
        inputs = layers.Input(shape=self.input_shape)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return Model(inputs, decoded, name='improved_autoencoder')
    
    def _spectral_loss(self, y_true, y_pred):
        """Compute loss in frequency domain."""
        # Ensure we're working with 2D signals (batch, time)
        # Reshape from (batch, channels, time) to (batch * channels, time)
        batch_size = tf.shape(y_true)[0]
        n_channels = tf.shape(y_true)[1]
        n_times = tf.shape(y_true)[2]
        
        y_true_reshaped = tf.reshape(y_true, [-1, n_times])
        y_pred_reshaped = tf.reshape(y_pred, [-1, n_times])
        
        # Convert to complex for FFT
        y_true_complex = tf.cast(y_true_reshaped, tf.complex64)
        y_pred_complex = tf.cast(y_pred_reshaped, tf.complex64)
        
        # Compute FFT
        fft_true = tf.signal.fft(y_true_complex)
        fft_pred = tf.signal.fft(y_pred_complex)
        
        # Compare magnitude spectra
        mag_true = tf.abs(fft_true)
        mag_pred = tf.abs(fft_pred)
        
        # Compute spectral loss
        spectral_loss = tf.reduce_mean(tf.square(mag_true - mag_pred))
        
        return spectral_loss
    
    def _gradient_loss(self, y_true, y_pred):
        """Compute gradient loss to preserve edges."""
        # Compute gradients along time axis
        grad_true = y_true[:, :, 1:] - y_true[:, :, :-1]
        grad_pred = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        
        return tf.reduce_mean(tf.square(grad_true - grad_pred))
    
    def compile(self, loss_weights: Optional[Dict[str, float]] = None):
        """Compile the autoencoder model with improved loss function."""
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        if loss_weights is None:
            loss_weights = {
                'mse': 1.0,
                'l1': 0.1,
                'spectral': 0.2 if self.use_spectral_loss else 0.0,
                'gradient': 0.1
            }
        
        def combined_loss(y_true, y_pred):
            # Time domain losses
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
            
            # Frequency domain loss
            if self.use_spectral_loss:
                spec_loss = self._spectral_loss(y_true, y_pred)
            else:
                spec_loss = 0.0
            
            # Gradient loss
            grad_loss = self._gradient_loss(y_true, y_pred)
            
            # Weighted combination
            total_loss = (
                loss_weights['mse'] * mse_loss +
                loss_weights['l1'] * l1_loss +
                loss_weights['spectral'] * spec_loss +
                loss_weights['gradient'] * grad_loss
            )
            
            return total_loss
        
        self.autoencoder.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=['mae', 'mse']
        )
    
    def normalize_per_channel(self, data: np.ndarray) -> np.ndarray:
        """Normalize each channel independently."""
        normalized = np.zeros_like(data)
        n_epochs, n_channels, n_times = data.shape
        
        for ch in range(n_channels):
            channel_data = data[:, ch, :]
            mean = np.mean(channel_data)
            std = np.std(channel_data) + 1e-8
            normalized[:, ch, :] = (channel_data - mean) / std
        
        return normalized
    
    def augment_data(self, clean_data: np.ndarray, artifact_data: np.ndarray, 
                     augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Augment training data with various transformations."""
        augmented_clean = []
        augmented_artifacts = []
        
        for i in range(len(clean_data)):
            # Original
            augmented_clean.append(clean_data[i])
            augmented_artifacts.append(artifact_data[i])
            
            for _ in range(augmentation_factor - 1):
                # Time shift
                shift = np.random.randint(-10, 10)
                augmented_clean.append(np.roll(clean_data[i], shift, axis=1))
                augmented_artifacts.append(np.roll(artifact_data[i], shift, axis=1))
                
                # Amplitude scaling for artifacts
                scale = np.random.uniform(0.8, 1.2)
                augmented_clean.append(clean_data[i])
                augmented_artifacts.append(artifact_data[i] * scale)
                
                # Add small noise
                noise_level = 0.02
                noise = np.random.normal(0, noise_level, clean_data[i].shape)
                augmented_clean.append(clean_data[i] + noise)
                augmented_artifacts.append(artifact_data[i])
        
        return np.array(augmented_clean), np.array(augmented_artifacts)
    
    def train(
        self,
        clean_data: np.ndarray,
        noisy_data: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping_patience: int = 15,
        use_augmentation: bool = True,
        use_curriculum: bool = True
    ) -> Dict[str, Any]:
        """
        Train the improved autoencoder model.
        
        Args:
            clean_data: Clean EEG data
            noisy_data: Noisy EEG data containing artifacts
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Patience for early stopping
            use_augmentation: Whether to use data augmentation
            use_curriculum: Whether to use curriculum learning
            
        Returns:
            Training history
        """
        # Calculate artifacts
        artifact_data = noisy_data - clean_data
        
        # Print statistics
        print(f"Training data shape: {noisy_data.shape}")
        print(f"Artifact statistics:")
        print(f"  - Mean absolute artifact: {np.mean(np.abs(artifact_data)):.6f}")
        print(f"  - Artifact std: {np.std(artifact_data):.6f}")
        
        # Data augmentation
        if use_augmentation:
            print("Applying data augmentation...")
            clean_data, artifact_data = self.augment_data(clean_data, artifact_data)
            noisy_data = clean_data + artifact_data
            print(f"Augmented data shape: {noisy_data.shape}")
        
        # Channel-wise normalization
        print("Applying channel-wise normalization...")
        noisy_data_norm = self.normalize_per_channel(noisy_data)
        artifact_data_norm = self.normalize_per_channel(artifact_data)
        
        # Store normalization parameters per channel
        self.channel_means = np.mean(noisy_data, axis=(0, 2))
        self.channel_stds = np.std(noisy_data, axis=(0, 2)) + 1e-8
        self.artifact_channel_means = np.mean(artifact_data, axis=(0, 2))
        self.artifact_channel_stds = np.std(artifact_data, axis=(0, 2)) + 1e-8
        
        # Curriculum learning
        if use_curriculum:
            print("Using curriculum learning strategy...")
            # Sort by artifact magnitude
            artifact_magnitudes = np.std(artifact_data_norm, axis=(1, 2))
            sorted_indices = np.argsort(artifact_magnitudes)
            
            # Train in stages
            stages = [0.3, 0.6, 1.0]
            history_combined = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
            
            for stage_idx, stage in enumerate(stages):
                print(f"\nTraining stage {stage_idx + 1}/{len(stages)} (using {stage*100:.0f}% of data)")
                
                n_samples = int(len(sorted_indices) * stage)
                indices = sorted_indices[:n_samples]
                
                stage_noisy = noisy_data_norm[indices]
                stage_artifacts = artifact_data_norm[indices]
                
                # Callbacks for this stage
                callbacks_list = [
                    callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=early_stopping_patience // len(stages),
                        restore_best_weights=True,
                        min_delta=1e-6
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-7
                    )
                ]
                
                # Train this stage
                history = self.autoencoder.fit(
                    x=stage_noisy,
                    y=stage_artifacts,
                    batch_size=batch_size,
                    epochs=epochs // len(stages),
                    validation_split=validation_split,
                    callbacks=callbacks_list,
                    verbose=1
                )
                
                # Combine histories
                for key in history.history:
                    history_combined[key].extend(history.history[key])
            
            return history_combined
        else:
            # Standard training
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    min_delta=1e-6
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7
                ),
                callbacks.ModelCheckpoint(
                    'best_improved_artifact_model.weights.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
            ]
            
            history = self.autoencoder.fit(
                x=noisy_data_norm,
                y=artifact_data_norm,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks_list,
                verbose=1
            )
            
            return history.history
    
    def predict_artifacts(self, noisy_data: np.ndarray, use_tta: bool = False) -> np.ndarray:
        """
        Predict artifacts from noisy data.
        
        Args:
            noisy_data: Noisy EEG data
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Predicted artifacts
        """
        # Normalize per channel
        noisy_data_norm = np.zeros_like(noisy_data)
        for ch in range(noisy_data.shape[1]):
            if self.channel_means is not None and self.channel_stds is not None:
                noisy_data_norm[:, ch, :] = (
                    (noisy_data[:, ch, :] - self.channel_means[ch]) / self.channel_stds[ch]
                )
            else:
                # Fallback to channel-wise normalization
                ch_mean = np.mean(noisy_data[:, ch, :])
                ch_std = np.std(noisy_data[:, ch, :]) + 1e-8
                noisy_data_norm[:, ch, :] = (noisy_data[:, ch, :] - ch_mean) / ch_std
        
        if use_tta:
            # Test-time augmentation
            predictions = []
            
            # Original
            pred = self.autoencoder.predict(noisy_data_norm)
            predictions.append(pred)
            
            # Flipped
            pred_flip = self.autoencoder.predict(noisy_data_norm[:, :, ::-1])
            predictions.append(pred_flip[:, :, ::-1])
            
            # Small shifts
            for shift in [-5, 5]:
                shifted = np.roll(noisy_data_norm, shift, axis=2)
                pred_shift = self.autoencoder.predict(shifted)
                pred_shift = np.roll(pred_shift, -shift, axis=2)
                predictions.append(pred_shift)
            
            # Average predictions
            predicted_artifacts_norm = np.mean(predictions, axis=0)
        else:
            predicted_artifacts_norm = self.autoencoder.predict(noisy_data_norm)
        
        # Denormalize per channel
        predicted_artifacts = np.zeros_like(predicted_artifacts_norm)
        for ch in range(predicted_artifacts.shape[1]):
            if self.artifact_channel_means is not None and self.artifact_channel_stds is not None:
                predicted_artifacts[:, ch, :] = (
                    predicted_artifacts_norm[:, ch, :] * self.artifact_channel_stds[ch] + 
                    self.artifact_channel_means[ch]
                )
            else:
                predicted_artifacts[:, ch, :] = predicted_artifacts_norm[:, ch, :]
        
        return predicted_artifacts
    
    def clean_data(self, noisy_data: np.ndarray, use_tta: bool = False) -> np.ndarray:
        """Clean noisy data by subtracting predicted artifacts."""
        predicted_artifacts = self.predict_artifacts(noisy_data, use_tta=use_tta)
        return noisy_data - predicted_artifacts


class ManifoldAutoencoder:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        encoder_filters: List[int] = [64, 128, 256],
        decoder_filters: List[int] = [256, 128, 64],
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-5,
        learning_rate: float = 1e-4
    ):
        """
        Initialize the Manifold Autoencoder for artifact estimation.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data (channels, timepoints)
            latent_dim (int): Dimension of the latent space
            encoder_filters (List[int]): Number of filters for each encoder layer
            decoder_filters (List[int]): Number of filters for each decoder layer
            kernel_size (int): Size of convolutional kernels
            dropout_rate (float): Dropout rate for regularization
            l2_reg (float): L2 regularization factor
            learning_rate (float): Learning rate for the optimizer
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        # Normalization parameters (will be set during training)
        self.input_mean = None
        self.input_std = None
        self.artifact_mean = None
        self.artifact_std = None
        
        # Calculate the shape after encoder convolutions by building a test model
        self.encoder_output_shape = self._calculate_encoder_output_shape()
        print(f"Calculated encoder output shape: {self.encoder_output_shape}")
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.autoencoder = self._build_autoencoder()
    
    def _calculate_encoder_output_shape(self) -> Tuple[int, int, int]:
        """Calculate the shape of the encoder output before flattening by building a test model."""
        # Build a test encoder to determine the actual output shape
        test_input = layers.Input(shape=self.input_shape)
        x = layers.Reshape((*self.input_shape, 1))(test_input)
        
        # Apply the same convolutions as in the encoder
        for filters in self.encoder_filters:
            x = layers.Conv2D(
                filters,
                (self.kernel_size, self.kernel_size),
                strides=(2, 2),
                padding='same'
            )(x)
        
        # Create a temporary model to get the output shape
        temp_model = Model(test_input, x)
        
        # Get the output shape (excluding batch dimension)
        output_shape = temp_model.output_shape[1:]  # Remove batch dimension
        print(f"Actual encoder conv output shape: {output_shape}")
        
        return output_shape
        
    def _build_encoder(self) -> Model:
        """Build the encoder network."""
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Reshape((*self.input_shape, 1))(inputs)
        
        # Encoder layers with stride instead of maxpooling
        for filters in self.encoder_filters:
            x = layers.Conv2D(
                filters,
                (self.kernel_size, self.kernel_size),
                strides=(2, 2),
                padding='same',
                kernel_regularizer=l2(self.l2_reg)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(
            self.latent_dim,
            kernel_regularizer=l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        return Model(inputs, x, name='encoder')
    
    def _build_decoder(self) -> Model:
        """Build the decoder network."""
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        
        # Reconstruct the feature map from latent space
        x = layers.Dense(np.prod(self.encoder_output_shape))(latent_inputs)
        x = layers.Reshape(self.encoder_output_shape)(x)
        
        # Decoder layers with transpose convolutions
        for filters in self.decoder_filters:
            x = layers.Conv2DTranspose(
                filters,
                (self.kernel_size, self.kernel_size),
                strides=(2, 2),
                padding='same',
                kernel_regularizer=l2(self.l2_reg)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Final layer to get to single channel
        x = layers.Conv2DTranspose(
            1,
            (self.kernel_size, self.kernel_size),
            strides=(1, 1),  # No stride for final layer
            padding='same',
            activation='linear'
        )(x)
        
        # Crop or pad to match exact input shape if needed
        current_shape = x.shape[1:3]  # Get height and width
        target_shape = self.input_shape
        
        print(f"Decoder output shape before reshape: {current_shape}")
        print(f"Target input shape: {target_shape}")
        
        # If shapes don't match exactly, use cropping or padding
        if current_shape[0] != target_shape[0] or current_shape[1] != target_shape[1]:
            # Calculate cropping/padding needed
            height_diff = current_shape[0] - target_shape[0]
            width_diff = current_shape[1] - target_shape[1]
            
            if height_diff > 0 or width_diff > 0:
                # Need to crop
                crop_top = height_diff // 2
                crop_bottom = height_diff - crop_top
                crop_left = width_diff // 2
                crop_right = width_diff - crop_left
                
                x = layers.Cropping2D(
                    cropping=((crop_top, crop_bottom), (crop_left, crop_right))
                )(x)
            elif height_diff < 0 or width_diff < 0:
                # Need to pad
                pad_top = abs(height_diff) // 2
                pad_bottom = abs(height_diff) - pad_top
                pad_left = abs(width_diff) // 2
                pad_right = abs(width_diff) - pad_left
                
                x = layers.ZeroPadding2D(
                    padding=((pad_top, pad_bottom), (pad_left, pad_right))
                )(x)
        
        # Final reshape to match input shape
        x = layers.Reshape(self.input_shape)(x)
        
        return Model(latent_inputs, x, name='decoder')
    
    def _build_autoencoder(self) -> Model:
        """Build the complete autoencoder."""
        inputs = layers.Input(shape=self.input_shape)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return Model(inputs, decoded, name='autoencoder')
    
    def compile(self, loss_weights: Dict[str, float] = None):
        """Compile the autoencoder model."""
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        # Use a combination of losses for better artifact learning
        def combined_loss(y_true, y_pred):
            # MSE loss
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Add a small regularization term to prevent trivial solutions
            l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
            
            # Combine losses
            return mse_loss + 0.1 * l1_loss
        
        self.autoencoder.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=['mae', 'mse']
        )
    
    def train(
        self,
        clean_data: np.ndarray,
        noisy_data: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping_patience: int = 15,
        artifact_threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Train the autoencoder model.
        
        Args:
            clean_data (np.ndarray): Clean EEG data
            noisy_data (np.ndarray): Noisy EEG data containing artifacts
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            validation_split (float): Fraction of data to use for validation
            early_stopping_patience (int): Patience for early stopping
            artifact_threshold (float): Minimum artifact magnitude to consider
            
        Returns:
            Dict[str, Any]: Training history
        """
        # Ensure input shapes match
        if clean_data.shape != noisy_data.shape:
            raise ValueError(f"Clean data shape {clean_data.shape} does not match noisy data shape {noisy_data.shape}")
        
        # Calculate the difference between noisy and clean data (artifacts)
        artifact_data = noisy_data - clean_data
        
        # Print statistics about the artifacts
        artifact_magnitude = np.std(artifact_data)
        artifact_mean = np.mean(np.abs(artifact_data))
        print(f"Artifact statistics:")
        print(f"  - Mean absolute artifact: {artifact_mean:.6f}")
        print(f"  - Artifact std: {artifact_magnitude:.6f}")
        print(f"  - Max artifact: {np.max(np.abs(artifact_data)):.6f}")
        print(f"  - Min artifact: {np.min(np.abs(artifact_data)):.6f}")
        
        # Check if artifacts are meaningful
        if artifact_magnitude < artifact_threshold:
            print(f"Warning: Artifact magnitude ({artifact_magnitude:.6f}) is very small.")
            print("This might indicate that clean and noisy data are too similar.")
        
        # Print shapes for debugging
        print(f"Input data shape: {noisy_data.shape}")
        print(f"Target data shape: {artifact_data.shape}")
        
        # Normalize data to improve training stability
        noisy_data_norm = (noisy_data - np.mean(noisy_data)) / (np.std(noisy_data) + 1e-8)
        artifact_data_norm = (artifact_data - np.mean(artifact_data)) / (np.std(artifact_data) + 1e-8)
        
        print(f"Normalized artifact std: {np.std(artifact_data_norm):.6f}")
        
        # Plot the first epoch for visualization
        self._plot_first_epoch(noisy_data_norm, artifact_data_norm)
        
        # Prepare callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                min_delta=1e-6
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                min_delta=1e-6
            ),
            callbacks.ModelCheckpoint(
                'best_artifact_model.weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )
        ]
        
        # Train the model with normalized data
        history = self.autoencoder.fit(
            x=noisy_data_norm,
            y=artifact_data_norm,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1,
            shuffle=True
        )
        
        # Store normalization parameters
        self.input_mean = np.mean(noisy_data)
        self.input_std = np.std(noisy_data)
        self.artifact_mean = np.mean(artifact_data)
        self.artifact_std = np.std(artifact_data)
        
        return history.history
    
    def predict_artifacts(self, noisy_data: np.ndarray) -> np.ndarray:
        """
        Predict artifacts from noisy data.
        
        Args:
            noisy_data (np.ndarray): Noisy EEG data
            
        Returns:
            np.ndarray: Predicted artifacts
        """
        # Normalize input data using stored parameters
        if self.input_mean is not None and self.input_std is not None:
            noisy_data_norm = (noisy_data - self.input_mean) / (self.input_std + 1e-8)
        else:
            print("Warning: No normalization parameters found. Using raw data.")
            noisy_data_norm = noisy_data
        
        # Predict normalized artifacts
        predicted_artifacts_norm = self.autoencoder.predict(noisy_data_norm)
        
        # Denormalize artifacts
        if self.artifact_mean is not None and self.artifact_std is not None:
            predicted_artifacts = predicted_artifacts_norm * (self.artifact_std + 1e-8) + self.artifact_mean
        else:
            predicted_artifacts = predicted_artifacts_norm
        
        return predicted_artifacts
    
    def _plot_first_epoch(self, input_data: np.ndarray, target_data: np.ndarray):
        """
        Plot the first epoch of input and target data for visualization.
        
        Args:
            input_data (np.ndarray): Normalized input (noisy) data
            target_data (np.ndarray): Normalized target (artifact) data
        """
        if len(input_data) == 0:
            print("No data to plot")
            return
            
        # Get the first epoch
        first_input = input_data[0]  # Shape: (channels, timepoints)
        first_target = target_data[0]  # Shape: (channels, timepoints)
        
        n_channels = first_input.shape[0]
        n_timepoints = first_input.shape[1]
        
        # Create time axis (assuming arbitrary time units)
        time_axis = np.arange(n_timepoints)
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('First Epoch: Input (Noisy) vs Target (Artifacts)', fontsize=16)
        
        # Plot input data (noisy EEG)
        axes[0].set_title('Input Data (Normalized Noisy EEG)')
        for ch in range(min(n_channels, 10)):  # Plot max 10 channels for clarity
            axes[0].plot(time_axis, first_input[ch] + ch * 2, 
                        label=f'Ch {ch}', alpha=0.7, linewidth=0.8)
        axes[0].set_xlabel('Time Points')
        axes[0].set_ylabel('Amplitude (offset by channel)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot target data (artifacts)
        axes[1].set_title('Target Data (Normalized Artifacts)')
        for ch in range(min(n_channels, 10)):  # Plot max 10 channels for clarity
            axes[1].plot(time_axis, first_target[ch] + ch * 2, 
                        label=f'Ch {ch}', alpha=0.7, linewidth=0.8)
        axes[1].set_xlabel('Time Points')
        axes[1].set_ylabel('Amplitude (offset by channel)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"\nFirst epoch statistics:")
        print(f"Input data - Mean: {np.mean(first_input):.6f}, Std: {np.std(first_input):.6f}")
        print(f"Target data - Mean: {np.mean(first_target):.6f}, Std: {np.std(first_target):.6f}")
        print(f"Input range: [{np.min(first_input):.6f}, {np.max(first_input):.6f}]")
        print(f"Target range: [{np.min(first_target):.6f}, {np.max(first_target):.6f}]")
    
    def clean_data(self, noisy_data: np.ndarray) -> np.ndarray:
        """
        Clean noisy data by subtracting predicted artifacts.
        
        Args:
            noisy_data (np.ndarray): Noisy EEG data
            
        Returns:
            np.ndarray: Cleaned EEG data
        """
        predicted_artifacts = self.predict_artifacts(noisy_data)
        return noisy_data - predicted_artifacts

    def save_model(self, filename: str):
        """
        Save the trained model and its normalization parameters to a file.
        
        Args:
            filename (str): Path to the file where the model will be saved (without extension)
        """
        if self.autoencoder is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save model weights
        weights_filename = f"{filename}.weights.h5"
        self.autoencoder.save_weights(weights_filename)
        
        # Save model configuration and normalization parameters
        model_data = {
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'encoder_filters': self.encoder_filters,
            'decoder_filters': self.decoder_filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'learning_rate': self.learning_rate,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'artifact_mean': self.artifact_mean,
            'artifact_std': self.artifact_std,
            'encoder_output_shape': self.encoder_output_shape
        }
        
        config_filename = f"{filename}_config.pkl"
        with open(config_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {weights_filename} and {config_filename}")
    
    def load_model(self, filename: str):
        """
        Load the trained model and its normalization parameters from a file.
        
        Args:
            filename (str): Path to the file where the model is saved (without extension)
        """
        # Load model configuration
        config_filename = f"{filename}_config.pkl"
        if not os.path.exists(config_filename):
            raise FileNotFoundError(f"Configuration file not found: {config_filename}")
        
        with open(config_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Set model parameters
        self.input_shape = model_data['input_shape']
        self.latent_dim = model_data['latent_dim']
        self.encoder_filters = model_data['encoder_filters']
        self.decoder_filters = model_data['decoder_filters']
        self.kernel_size = model_data['kernel_size']
        self.dropout_rate = model_data['dropout_rate']
        self.l2_reg = model_data['l2_reg']
        self.learning_rate = model_data['learning_rate']
        self.input_mean = model_data['input_mean']
        self.input_std = model_data['input_std']
        self.artifact_mean = model_data['artifact_mean']
        self.artifact_std = model_data['artifact_std']
        self.encoder_output_shape = model_data['encoder_output_shape']
        
        # Rebuild the model architecture
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.autoencoder = self._build_autoencoder()
        
        # Load model weights
        weights_filename = f"{filename}.weights.h5"
        if not os.path.exists(weights_filename):
            raise FileNotFoundError(f"Weights file not found: {weights_filename}")
        
        self.autoencoder.load_weights(weights_filename)
        
        # Compile the model
        self.compile()
        
        print(f"Model loaded from {weights_filename} and {config_filename}")

    @classmethod
    def from_file(cls, filename: str) -> 'ManifoldAutoencoder':
        """
        Create a ManifoldAutoencoder instance from a saved model file.
        
        Args:
            filename (str): Path to the file where the model is saved (without extension)
            
        Returns:
            ManifoldAutoencoder: Loaded model instance
        """
        # Load configuration first to get the input shape
        config_filename = f"{filename}_config.pkl"
        if not os.path.exists(config_filename):
            raise FileNotFoundError(f"Configuration file not found: {config_filename}")
        
        with open(config_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance with loaded parameters
        instance = cls(
            input_shape=model_data['input_shape'],
            latent_dim=model_data['latent_dim'],
            encoder_filters=model_data['encoder_filters'],
            decoder_filters=model_data['decoder_filters'],
            kernel_size=model_data['kernel_size'],
            dropout_rate=model_data['dropout_rate'],
            l2_reg=model_data['l2_reg'],
            learning_rate=model_data['learning_rate']
        )
        
        # Load the rest of the model
        instance.load_model(filename)
        
        return instance


class ArtifactEstimator:
    def __init__(self, eeg: EEG):
        """
        Initialize the ArtifactEstimator with an EEG object.
        
        Args:
            eeg (EEG): EEG object containing both clean (raw) and noisy (raw_orig) data
        """
        self.eeg = eeg
        self.clean_epochs = None
        self.noisy_epochs = None
        self.epochs_info = None
        self.model = None
        
    def prepare_epochs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare epochs from both clean and noisy data around artifact positions.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing (clean_epochs, noisy_epochs)
        """
        if self.eeg.loaded_triggers is None:
            raise ValueError("No triggers found in EEG object")
            
        # Get picks excluding bad channels
        picks = mne.pick_types(self.eeg.mne_raw.info, eeg=True, exclude='bads')
            
        # Create epochs from clean data
        events = self.eeg.triggers_as_events
        clean_epochs = mne.Epochs(
            self.eeg.mne_raw,
            events,
            tmin=self.eeg.tmin,
            tmax=self.eeg.tmax,
            picks=picks,
            baseline=None,
            preload=True
        )
        
        # Create epochs from noisy data
        noisy_epochs = mne.Epochs(
            self.eeg.mne_raw_orig,
            events,
            tmin=self.eeg.tmin,
            tmax=self.eeg.tmax,
            picks=picks,
            baseline=None,
            preload=True
        )
        
        # Store epochs information
        self.epochs_info = {
            'n_epochs': len(clean_epochs),
            'n_channels': clean_epochs.info['nchan'],
            'n_times': len(clean_epochs.times),
            'sfreq': clean_epochs.info['sfreq'],
            'ch_names': clean_epochs.ch_names
        }
        
        # Get data as numpy arrays
        clean_data = clean_epochs.get_data()
        noisy_data = noisy_epochs.get_data()
        
        # Store the data
        self.clean_epochs = clean_data
        self.noisy_epochs = noisy_data
        
        return clean_data, noisy_data
    
    def get_data_shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of the prepared epochs data.
        
        Returns:
            Tuple[int, int, int]: Shape of epochs (n_epochs, n_channels, n_times)
        """
        if self.clean_epochs is None:
            raise ValueError("Epochs not prepared yet. Call prepare_epochs() first.")
        return self.clean_epochs.shape
    
    def get_epochs_info(self) -> dict:
        """
        Get information about the prepared epochs.
        
        Returns:
            dict: Dictionary containing epochs information
        """
        if self.epochs_info is None:
            raise ValueError("Epochs not prepared yet. Call prepare_epochs() first.")
        return self.epochs_info
        
    def train_model(
        self,
        latent_dim: int = 32,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the manifold autoencoder model.
        
        Args:
            latent_dim (int): Dimension of the latent space
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            Dict[str, Any]: Training history
        """
        if self.clean_epochs is None or self.noisy_epochs is None:
            raise ValueError("Epochs not prepared yet. Call prepare_epochs() first.")
            
        # Print shapes for debugging
        print(f"Clean data shape: {self.clean_epochs.shape}")
        print(f"Noisy data shape: {self.noisy_epochs.shape}")
        
        # Initialize and compile the model
        self.model = ManifoldAutoencoder(
            input_shape=(self.clean_epochs.shape[1], self.clean_epochs.shape[2]),
            latent_dim=latent_dim,
            encoder_filters=[64, 128, 256],  # Reduced number of layers
            decoder_filters=[256, 128, 64]   # Reduced number of layers
        )
        self.model.compile()
        
        # Train the model
        history = self.model.train(
            clean_data=self.clean_epochs,
            noisy_data=self.noisy_epochs,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split
        )
        
        return history
    
    def predict_artifacts(self, noisy_data: np.ndarray) -> np.ndarray:
        """
        Predict artifacts from noisy data.
        
        Args:
            noisy_data (np.ndarray): Noisy EEG data
            
        Returns:
            np.ndarray: Predicted artifacts
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        return self.model.predict_artifacts(noisy_data)
    
    def clean_data(self, noisy_data: np.ndarray) -> np.ndarray:
        """
        Clean noisy data by subtracting predicted artifacts.
        
        Args:
            noisy_data (np.ndarray): Noisy EEG data
            
        Returns:
            np.ndarray: Cleaned EEG data
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        return self.model.clean_data(noisy_data)
    
    def clean_continuous_data(self, use_noisy_original: bool = True) -> np.ndarray:
        """
        Clean the continuous EEG data by applying the trained model to epochs and 
        reconstructing the full continuous data with original time indices.
        
        This method:
        1. Creates epochs from the continuous data at trigger positions
        2. Applies the trained artifact removal model to each epoch
        3. Reconstructs the continuous data by placing cleaned epochs back at their original positions
        4. Handles overlapping regions by averaging corrections
        5. Preserves the original data structure and timing
        
        Args:
            use_noisy_original (bool): If True, use mne_raw_orig as the source data.
                                     If False, use mne_raw as the source data.
            
        Returns:
            np.ndarray: Cleaned continuous EEG data with shape (n_channels, n_times)
            
        Example:
            >>> estimator = ArtifactEstimator(eeg)
            >>> estimator.prepare_epochs()
            >>> estimator.train_model()
            >>> cleaned_data = estimator.clean_continuous_data()
            >>> # Apply to Raw object directly
            >>> estimator.apply_cleaning_to_raw(inplace=True)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        if self.eeg.loaded_triggers is None:
            raise ValueError("No triggers found in EEG object")
        
        # Choose source data
        source_raw = self.eeg.mne_raw_orig if use_noisy_original else self.eeg.mne_raw
        
        # Get picks excluding bad channels (same as used in prepare_epochs)
        picks = mne.pick_types(source_raw.info, eeg=True, exclude='bads')
        
        # Get the continuous data
        continuous_data = source_raw.get_data(picks=picks).copy()
        n_channels, n_times = continuous_data.shape
        
        print(f"Processing continuous data with shape: {continuous_data.shape}")
        
        # Create epochs from the source data
        events = self.eeg.triggers_as_events
        epochs = mne.Epochs(
            source_raw,
            events,
            tmin=self.eeg.tmin,
            tmax=self.eeg.tmax,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False
        )
        
        # Get epoch data
        epoch_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times_epoch)
        n_epochs, n_channels_epoch, n_times_epoch = epoch_data.shape
        
        print(f"Created {n_epochs} epochs with shape: {epoch_data.shape}")
        
        # Clean the epochs using the trained model
        cleaned_epochs = self.clean_data(epoch_data)
        
        # Calculate sample indices for each epoch
        sfreq = source_raw.info['sfreq']
        tmin_samples = int(self.eeg.tmin * sfreq)
        tmax_samples = int(self.eeg.tmax * sfreq)
        epoch_length_samples = tmax_samples - tmin_samples + 1
        
        # Ensure epoch length matches
        if epoch_length_samples != n_times_epoch:
            print(f"Warning: Calculated epoch length ({epoch_length_samples}) doesn't match actual ({n_times_epoch})")
            epoch_length_samples = n_times_epoch
        
        # Create a copy of the continuous data for cleaning
        cleaned_continuous = continuous_data.copy()
        
        # Track which samples have been modified (for handling overlaps)
        modification_count = np.zeros(n_times, dtype=int)
        accumulated_corrections = np.zeros_like(continuous_data)
        
        # Place cleaned epochs back into continuous data
        for epoch_idx, trigger_sample in enumerate(self.eeg.loaded_triggers):
            # Calculate the start and end indices for this epoch in the continuous data
            start_idx = trigger_sample + tmin_samples
            end_idx = start_idx + epoch_length_samples
            
            # Ensure we don't go beyond the data boundaries
            start_idx = max(0, start_idx)
            end_idx = min(n_times, end_idx)
            
            # Calculate corresponding indices in the epoch data
            epoch_start = max(0, -start_idx + trigger_sample + tmin_samples)
            epoch_end = epoch_start + (end_idx - start_idx)
            
            if start_idx < end_idx and epoch_start < epoch_end:
                # Calculate the correction (difference between cleaned and original)
                original_segment = continuous_data[:, start_idx:end_idx]
                cleaned_segment = cleaned_epochs[epoch_idx, :, epoch_start:epoch_end]
                correction = cleaned_segment - original_segment
                
                # Accumulate corrections for overlapping regions
                accumulated_corrections[:, start_idx:end_idx] += correction
                modification_count[start_idx:end_idx] += 1
        
        # Apply averaged corrections where there were overlaps
        for i in range(n_times):
            if modification_count[i] > 0:
                cleaned_continuous[:, i] = continuous_data[:, i] + accumulated_corrections[:, i] / modification_count[i]
        
        print(f"Applied corrections to {np.sum(modification_count > 0)} samples")
        print(f"Maximum overlap count: {np.max(modification_count)}")
        
        return cleaned_continuous
    
    def apply_cleaning_to_raw(self, use_noisy_original: bool = True, inplace: bool = True) -> Optional[mne.io.Raw]:
        """
        Apply the trained model to clean the continuous EEG data and update the MNE Raw object.
        
        Args:
            use_noisy_original (bool): If True, use mne_raw_orig as the source data.
                                     If False, use mne_raw as the source data.
            inplace (bool): If True, modify the existing Raw object. If False, return a new Raw object.
            
        Returns:
            Optional[mne.io.Raw]: If inplace=False, returns the cleaned Raw object. Otherwise returns None.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get cleaned continuous data
        cleaned_data = self.clean_continuous_data(use_noisy_original=use_noisy_original)
        
        # Choose target raw object
        target_raw = self.eeg.mne_raw_orig if use_noisy_original else self.eeg.mne_raw
        
        # Get picks excluding bad channels
        picks = mne.pick_types(target_raw.info, eeg=True, exclude='bads')
        
        if inplace:
            # Modify the existing Raw object
            target_raw._data[picks] = cleaned_data
            print(f"Applied cleaning to {target_raw} in place")
            return None
        else:
            # Create a new Raw object
            new_raw = target_raw.copy()
            new_raw._data[picks] = cleaned_data
            print(f"Created new cleaned Raw object")
            return new_raw
    
    def save_model(self, filename: str):
        """
        Save the trained model and epochs information to files.
        
        Args:
            filename (str): Path to the file where the model will be saved (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save the underlying ManifoldAutoencoder model
        self.model.save_model(filename)
        
        # Save epochs information
        if self.epochs_info is not None:
            epochs_info_filename = f"{filename}_epochs_info.pkl"
            with open(epochs_info_filename, 'wb') as f:
                pickle.dump(self.epochs_info, f)
            print(f"Epochs info saved to {epochs_info_filename}")
    
    def load_model(self, filename: str):
        """
        Load the trained model and epochs information from files.
        
        Args:
            filename (str): Path to the file where the model is saved (without extension)
        """
        # Load the ManifoldAutoencoder model
        self.model = ManifoldAutoencoder.from_file(filename)
        
        # Load epochs information if available
        epochs_info_filename = f"{filename}_epochs_info.pkl"
        if os.path.exists(epochs_info_filename):
            with open(epochs_info_filename, 'rb') as f:
                self.epochs_info = pickle.load(f)
            print(f"Epochs info loaded from {epochs_info_filename}")
        else:
            print(f"Warning: Epochs info file not found: {epochs_info_filename}")
    
    @classmethod
    def from_file(cls, eeg: EEG, filename: str) -> 'ArtifactEstimator':
        """
        Create an ArtifactEstimator instance with a loaded model.
        
        Args:
            eeg (EEG): EEG object containing the data
            filename (str): Path to the file where the model is saved (without extension)
            
        Returns:
            ArtifactEstimator: Instance with loaded model
        """
        estimator = cls(eeg)
        estimator.load_model(filename)
        return estimator


class ImprovedArtifactEstimator(ArtifactEstimator):
    """
    Improved Artifact Estimator with support for the enhanced model architecture
    and additional features like ensemble learning and advanced post-processing.
    """
    
    def __init__(self, eeg: EEG, use_improved_model: bool = True):
        """
        Initialize the Improved ArtifactEstimator.
        
        Args:
            eeg: EEG object containing both clean and noisy data
            use_improved_model: Whether to use the improved model architecture
        """
        super().__init__(eeg)
        self.use_improved_model = use_improved_model
        self.ensemble_models = []
        self.post_processing_enabled = True
        
    def train_model(
        self,
        latent_dim: int = 32,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        use_augmentation: bool = True,
        use_curriculum: bool = True,
        use_attention: bool = True,
        use_spectral_loss: bool = True,
        ensemble_size: int = 1
    ) -> Dict[str, Any]:
        """
        Train the improved manifold autoencoder model.
        
        Args:
            latent_dim: Dimension of the latent space
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            use_augmentation: Whether to use data augmentation
            use_curriculum: Whether to use curriculum learning
            use_attention: Whether to use attention mechanisms
            use_spectral_loss: Whether to include spectral loss
            ensemble_size: Number of models to train for ensemble (1 = no ensemble)
            
        Returns:
            Training history (or list of histories for ensemble)
        """
        if self.clean_epochs is None or self.noisy_epochs is None:
            raise ValueError("Epochs not prepared yet. Call prepare_epochs() first.")
        
        print(f"Training {'improved' if self.use_improved_model else 'standard'} model")
        print(f"Clean data shape: {self.clean_epochs.shape}")
        print(f"Noisy data shape: {self.noisy_epochs.shape}")
        
        if ensemble_size > 1:
            print(f"Training ensemble of {ensemble_size} models...")
            histories = []
            
            for i in range(ensemble_size):
                print(f"\nTraining model {i+1}/{ensemble_size}")
                
                if self.use_improved_model:
                    model = ImprovedManifoldAutoencoder(
                        input_shape=(self.clean_epochs.shape[1], self.clean_epochs.shape[2]),
                        latent_dim=latent_dim + i * 8,  # Vary latent dimension
                        encoder_filters=[64 + i*16, 128 + i*16, 256],
                        decoder_filters=[256, 128 + i*16, 64 + i*16],
                        dropout_rate=0.2 + i * 0.05,  # Vary dropout
                        use_attention=use_attention,
                        use_spectral_loss=use_spectral_loss
                    )
                else:
                    model = ManifoldAutoencoder(
                        input_shape=(self.clean_epochs.shape[1], self.clean_epochs.shape[2]),
                        latent_dim=latent_dim + i * 8,
                        encoder_filters=[64 + i*16, 128 + i*16, 256],
                        decoder_filters=[256, 128 + i*16, 64 + i*16],
                        dropout_rate=0.2 + i * 0.05
                    )
                
                model.compile()
                
                # Train with different random seeds
                np.random.seed(42 + i)
                tf.random.set_seed(42 + i)
                
                if self.use_improved_model:
                    history = model.train(
                        clean_data=self.clean_epochs,
                        noisy_data=self.noisy_epochs,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        use_augmentation=use_augmentation,
                        use_curriculum=use_curriculum
                    )
                else:
                    history = model.train(
                        clean_data=self.clean_epochs,
                        noisy_data=self.noisy_epochs,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split
                    )
                
                self.ensemble_models.append(model)
                histories.append(history)
            
            # Set the first model as the primary model
            self.model = self.ensemble_models[0]
            return histories
        else:
            # Single model training
            if self.use_improved_model:
                self.model = ImprovedManifoldAutoencoder(
                    input_shape=(self.clean_epochs.shape[1], self.clean_epochs.shape[2]),
                    latent_dim=latent_dim,
                    use_attention=use_attention,
                    use_spectral_loss=use_spectral_loss
                )
            else:
                self.model = ManifoldAutoencoder(
                    input_shape=(self.clean_epochs.shape[1], self.clean_epochs.shape[2]),
                    latent_dim=latent_dim
                )
            
            self.model.compile()
            
            if self.use_improved_model:
                history = self.model.train(
                    clean_data=self.clean_epochs,
                    noisy_data=self.noisy_epochs,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split,
                    use_augmentation=use_augmentation,
                    use_curriculum=use_curriculum
                )
            else:
                history = self.model.train(
                    clean_data=self.clean_epochs,
                    noisy_data=self.noisy_epochs,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split
                )
            
            return history
    
    def predict_artifacts(self, noisy_data: np.ndarray, use_tta: bool = False) -> np.ndarray:
        """
        Predict artifacts from noisy data using ensemble if available.
        
        Args:
            noisy_data: Noisy EEG data
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Predicted artifacts
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        if len(self.ensemble_models) > 1:
            # Ensemble prediction
            predictions = []
            for model in self.ensemble_models:
                if self.use_improved_model and hasattr(model, 'predict_artifacts'):
                    pred = model.predict_artifacts(noisy_data, use_tta=use_tta)
                else:
                    pred = model.predict_artifacts(noisy_data)
                predictions.append(pred)
            
            # Average ensemble predictions
            predicted_artifacts = np.mean(predictions, axis=0)
            
            # Optional: weighted average based on validation performance
            # This would require storing validation scores during training
        else:
            # Single model prediction
            if self.use_improved_model and hasattr(self.model, 'predict_artifacts'):
                predicted_artifacts = self.model.predict_artifacts(noisy_data, use_tta=use_tta)
            else:
                predicted_artifacts = self.model.predict_artifacts(noisy_data)
        
        # Apply post-processing if enabled
        if self.post_processing_enabled:
            predicted_artifacts = self._post_process_artifacts(predicted_artifacts, noisy_data)
        
        return predicted_artifacts
    
    def _post_process_artifacts(self, predicted_artifacts: np.ndarray, noisy_data: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to refine artifact predictions.
        
        Args:
            predicted_artifacts: Raw predicted artifacts
            noisy_data: Original noisy data
            
        Returns:
            Post-processed artifacts
        """
        from scipy.signal import savgol_filter
        from scipy.ndimage import median_filter
        
        processed_artifacts = predicted_artifacts.copy()
        
        # Calculate local SNR for adaptive filtering
        cleaned_data = noisy_data - predicted_artifacts
        signal_power = np.var(cleaned_data, axis=2, keepdims=True)
        artifact_power = np.var(predicted_artifacts, axis=2, keepdims=True)
        snr = signal_power / (artifact_power + 1e-8)
        
        for i in range(processed_artifacts.shape[0]):
            for j in range(processed_artifacts.shape[1]):
                # Apply median filter to remove spikes
                processed_artifacts[i, j] = median_filter(processed_artifacts[i, j], size=5)
                
                # Apply adaptive smoothing based on SNR
                if snr[i, j, 0] < 1.0:  # Low SNR, more smoothing
                    window_length = min(51, processed_artifacts.shape[2] // 4)
                    if window_length % 2 == 0:
                        window_length += 1
                    if window_length >= 5:  # Savgol filter requires window_length >= 5
                        processed_artifacts[i, j] = savgol_filter(
                            processed_artifacts[i, j], window_length, 3
                        )
        
        return processed_artifacts
    
    def evaluate_performance(self, test_clean: Optional[np.ndarray] = None, 
                           test_noisy: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate the model performance on test data.
        
        Args:
            test_clean: Clean test data (if None, uses validation split from training)
            test_noisy: Noisy test data (if None, uses validation split from training)
            
        Returns:
            Dictionary of performance metrics
        """
        if test_clean is None or test_noisy is None:
            # Use a portion of the training data for evaluation
            n_test = int(0.2 * len(self.clean_epochs))
            test_clean = self.clean_epochs[-n_test:]
            test_noisy = self.noisy_epochs[-n_test:]
        
        # Predict artifacts
        predicted_artifacts = self.predict_artifacts(test_noisy)
        true_artifacts = test_noisy - test_clean
        
        # Calculate metrics
        mse = np.mean((predicted_artifacts - true_artifacts) ** 2)
        mae = np.mean(np.abs(predicted_artifacts - true_artifacts))
        
        # Signal-to-artifact ratio improvement
        original_sar = np.var(test_clean) / np.var(true_artifacts)
        cleaned_data = test_noisy - predicted_artifacts
        residual_artifacts = cleaned_data - test_clean
        cleaned_sar = np.var(test_clean) / np.var(residual_artifacts)
        sar_improvement = cleaned_sar / original_sar
        
        # Correlation between predicted and true artifacts
        correlation = np.corrcoef(
            predicted_artifacts.flatten(), 
            true_artifacts.flatten()
        )[0, 1]
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(correlation),
            'sar_improvement': float(sar_improvement),
            'original_sar_db': float(10 * np.log10(original_sar)),
            'cleaned_sar_db': float(10 * np.log10(cleaned_sar))
        }
        
        print("\nPerformance Metrics:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  SAR Improvement: {metrics['sar_improvement']:.2f}x")
        print(f"  Original SAR: {metrics['original_sar_db']:.2f} dB")
        print(f"  Cleaned SAR: {metrics['cleaned_sar_db']:.2f} dB")
        
        return metrics
    
    def visualize_results(self, epoch_idx: int = 0, channel_idx: int = 0):
        """
        Visualize the artifact removal results for a specific epoch and channel.
        
        Args:
            epoch_idx: Index of the epoch to visualize
            channel_idx: Index of the channel to visualize
        """
        if self.clean_epochs is None or self.noisy_epochs is None:
            raise ValueError("No data available. Call prepare_epochs() first.")
        
        # Get data for visualization
        clean = self.clean_epochs[epoch_idx, channel_idx]
        noisy = self.noisy_epochs[epoch_idx, channel_idx]
        true_artifact = noisy - clean
        
        # Predict artifact
        noisy_epoch = self.noisy_epochs[epoch_idx:epoch_idx+1]
        predicted_artifact = self.predict_artifacts(noisy_epoch)[0, channel_idx]
        cleaned = noisy - predicted_artifact
        
        # Create time axis
        time = np.arange(len(clean)) / self.epochs_info['sfreq']
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Original signals
        axes[0].plot(time, clean, 'b-', label='Clean', alpha=0.7)
        axes[0].plot(time, noisy, 'r-', label='Noisy', alpha=0.7)
        axes[0].set_ylabel('Amplitude (μV)')
        axes[0].legend()
        axes[0].set_title(f'Original Signals - Epoch {epoch_idx}, Channel {channel_idx}')
        axes[0].grid(True, alpha=0.3)
        
        # Artifacts
        axes[1].plot(time, true_artifact, 'k-', label='True Artifact', alpha=0.7)
        axes[1].plot(time, predicted_artifact, 'g-', label='Predicted Artifact', alpha=0.7)
        axes[1].set_ylabel('Amplitude (μV)')
        axes[1].legend()
        axes[1].set_title('Artifact Comparison')
        axes[1].grid(True, alpha=0.3)
        
        # Cleaned signal
        axes[2].plot(time, clean, 'b-', label='Clean (Ground Truth)', alpha=0.7)
        axes[2].plot(time, cleaned, 'g-', label='Cleaned (Predicted)', alpha=0.7)
        axes[2].set_ylabel('Amplitude (μV)')
        axes[2].legend()
        axes[2].set_title('Cleaned Signal Comparison')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        residual = cleaned - clean
        axes[3].plot(time, residual, 'r-', label='Residual Error')
        axes[3].set_ylabel('Amplitude (μV)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend()
        axes[3].set_title('Residual Error (Cleaned - Clean)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        artifact_reduction = 1 - (np.var(residual) / np.var(true_artifact))
        print(f"\nArtifact reduction: {artifact_reduction*100:.1f}%")
        print(f"Residual RMS: {np.sqrt(np.mean(residual**2)):.4f} μV")
    
    def save_ensemble(self, base_filename: str):
        """
        Save all models in the ensemble.
        
        Args:
            base_filename: Base filename for saving models
        """
        if len(self.ensemble_models) > 1:
            print(f"Saving ensemble of {len(self.ensemble_models)} models...")
            for i, model in enumerate(self.ensemble_models):
                model.save_model(f"{base_filename}_model_{i}")
            
            # Save ensemble metadata
            ensemble_data = {
                'n_models': len(self.ensemble_models),
                'use_improved_model': self.use_improved_model,
                'post_processing_enabled': self.post_processing_enabled,
                'epochs_info': self.epochs_info
            }
            
            with open(f"{base_filename}_ensemble_meta.pkl", 'wb') as f:
                pickle.dump(ensemble_data, f)
        else:
            # Save single model
            super().save_model(base_filename)
    
    def load_ensemble(self, base_filename: str):
        """
        Load an ensemble of models.
        
        Args:
            base_filename: Base filename for loading models
        """
        # Load ensemble metadata
        with open(f"{base_filename}_ensemble_meta.pkl", 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.use_improved_model = ensemble_data['use_improved_model']
        self.post_processing_enabled = ensemble_data['post_processing_enabled']
        self.epochs_info = ensemble_data.get('epochs_info')
        
        # Load all models
        self.ensemble_models = []
        for i in range(ensemble_data['n_models']):
            if self.use_improved_model:
                model = ImprovedManifoldAutoencoder.from_file(f"{base_filename}_model_{i}")
            else:
                model = ManifoldAutoencoder.from_file(f"{base_filename}_model_{i}")
            self.ensemble_models.append(model)
        
        # Set the first model as primary
        self.model = self.ensemble_models[0]
        print(f"Loaded ensemble of {len(self.ensemble_models)} models")
