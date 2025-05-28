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
