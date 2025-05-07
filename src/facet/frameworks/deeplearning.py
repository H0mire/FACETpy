from typing import Any, Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from facet.eeg_obj import EEG
import mne
import os

class EEGDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

class DoubleConv(nn.Module):
    """Double convolution block used in UNet"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Ensure x1 and x2 have matching dimensions for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet model for EEG artifact removal"""
    def __init__(self, n_channels=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        # Contracting path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Expansive path
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        
        return output

class DeepLearning:
    def __init__(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Check for Apple Silicon (M1/M2/M3) GPU support via MPS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS (Metal) for acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for computation")
            
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.data_info = None
        
    def _prepare_data(self, eeg_obj: EEG) -> None:
        """
        Prepare the data for the deep learning model.
        
        This method prepares input-target pairs for UNet training:
        - Input: Epochs from artifact-contaminated EEG (mne_raw_orig)
        - Target: Corresponding epochs from cleaned EEG (mne_raw)
        
        Args:
            eeg_obj: The EEG object containing the data
            
        Returns:
            None: Sets up the data loaders for training, validation and testing
        """
        if eeg_obj.mne_raw is None or eeg_obj.mne_raw_orig is None:
            raise ValueError("Both cleaned (mne_raw) and original (mne_raw_orig) EEG data must be available")
            
        if eeg_obj.loaded_triggers is None or len(eeg_obj.loaded_triggers) == 0:
            raise ValueError("No artifact triggers found in the EEG object")
            
        # Create epochs around the artifact triggers
        events = np.column_stack(
            (
                eeg_obj.loaded_triggers,
                np.zeros_like(eeg_obj.loaded_triggers),
                np.ones_like(eeg_obj.loaded_triggers),
            )
        )
        
        # Get the artifact time window
        tmin = eeg_obj.tmin
        tmax = eeg_obj.tmax
        
        # Select EEG channels
        picks = mne.pick_types(
            eeg_obj.mne_raw.info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude="bads",
        )
        
        # Create epochs for the noisy data (input)
        noisy_epochs = mne.Epochs(
            eeg_obj.mne_raw_orig,
            events=events,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            baseline=None,
            preload=True,
        )
        
        # Create epochs for the cleaned data (target)
        clean_epochs = mne.Epochs(
            eeg_obj.mne_raw,
            events=events,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            baseline=None,
            preload=True,
        )
        
        # Convert epochs to numpy arrays
        X = noisy_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        y = clean_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        
        # Add a dimension for the channel dimension expected by UNet
        X = X[:, np.newaxis, :, :]  # Shape: (n_epochs, 1, n_channels, n_times)
        y = y[:, np.newaxis, :, :]  # Shape: (n_epochs, 1, n_channels, n_times)
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Split data into train, validation, and test sets
        n_samples = len(X)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_size = n_samples - train_size - val_size
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = EEGDataset(X[train_indices], y[train_indices])
        val_dataset = EEGDataset(X[val_indices], y[val_indices])
        test_dataset = EEGDataset(X[test_indices], y[test_indices])
        
        # For macOS, adjust DataLoader settings for better performance
        # Use fewer worker processes to reduce memory pressure
        num_workers = 0 if self.device.type == "mps" else 2
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=16, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type != "cpu")
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=16,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type != "cpu")
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=16,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type != "cpu")
        )
        
        # Store metadata
        self.data_info = {
            "n_channels": X.shape[2],
            "n_timepoints": X.shape[3],
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "n_test": len(test_dataset),
            "sfreq": eeg_obj.mne_raw.info['sfreq'],
            "ch_names": [eeg_obj.mne_raw.ch_names[i] for i in picks]
        }
        
        print(f"Data prepared: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")

    def train(self, eeg_obj: EEG) -> None:
        """
        Train the deep learning model.
        
        This method trains a UNet model to remove artifacts from EEG data.
        
        Args:
            eeg_obj: The EEG object containing the data
            
        Returns:
            None
        """
        # Prepare data if not already prepared
        if self.train_loader is None:
            self._prepare_data(eeg_obj)
            
        # Initialize the UNet model
        self.model = UNet(n_channels=1).to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training parameters
        num_epochs = 50
        best_val_loss = float('inf')
        patience = 5  # Early stopping patience
        patience_counter = 0
        
        # Create a directory for saving models if it doesn't exist
        model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'best_unet_model.pt')
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in self.train_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss = train_loss / len(self.train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(self.val_loader.dataset)
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
            
            # Check if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'data_info': self.data_info
                }, model_path)
                
                print(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}')
                print(f'Model saved to: {model_path}')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load the best model after training
        self.load_model(model_path)
        print(f"Training completed. Final validation loss: {best_val_loss:.6f}")
        
        # Evaluate on test set
        self._evaluate_test_set()

    def _evaluate_test_set(self) -> float:
        """
        Evaluate the model on the test set.
        
        Returns:
            float: The test loss
        """
        if self.model is None or self.test_loader is None:
            raise ValueError("Model not trained or test data not prepared")
            
        self.model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
        
        test_loss = test_loss / len(self.test_loader.dataset)
        print(f'Test Loss: {test_loss:.6f}')
        
        return test_loss

    def predict(self, eeg_obj: EEG) -> Any:
        """
        Apply the trained UNet model to clean EEG data.
        
        This method applies the trained model to remove artifacts from new EEG data.
        
        Args:
            eeg_obj: The EEG object containing the data to process
            
        Returns:
            EEG: A new EEG object with cleaned data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a pre-trained model")
            
        if eeg_obj.mne_raw_orig is None:
            raise ValueError("Original (contaminated) EEG data not found in the EEG object")
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Create a copy of the input EEG object for the result
        result_eeg = eeg_obj.copy()
        
        # Get original data
        orig_data = eeg_obj.mne_raw_orig.get_data()
        n_channels, n_times = orig_data.shape
        
        # Process data in epochs if triggers are available
        if eeg_obj.loaded_triggers is not None and len(eeg_obj.loaded_triggers) > 0:
            # Create epochs around the artifact triggers
            events = np.column_stack(
                (
                    eeg_obj.loaded_triggers,
                    np.zeros_like(eeg_obj.loaded_triggers),
                    np.ones_like(eeg_obj.loaded_triggers),
                )
            )
            
            # Get the artifact time window
            tmin = eeg_obj.tmin
            tmax = eeg_obj.tmax
            
            # Select EEG channels
            picks = mne.pick_types(
                eeg_obj.mne_raw_orig.info,
                meg=False,
                eeg=True,
                stim=False,
                eog=False,
                exclude="bads",
            )
            
            # Create epochs for the noisy data
            noisy_epochs = mne.Epochs(
                eeg_obj.mne_raw_orig,
                events=events,
                tmin=tmin,
                tmax=tmax,
                picks=picks,
                baseline=None,
                preload=True,
            )
            
            # Process each epoch with the model
            with torch.no_grad():
                # Create a new array to store the cleaned data
                cleaned_data = eeg_obj.mne_raw_orig.get_data().copy()
                
                # Process each epoch
                for i, epoch_data in enumerate(noisy_epochs.get_data()):
                    # Prepare the data for the model
                    input_data = torch.FloatTensor(epoch_data[np.newaxis, np.newaxis, :, :]).to(self.device)
                    
                    # Apply the model
                    cleaned_epoch = self.model(input_data).cpu().numpy()[0, 0]  # Remove batch and channel dimensions
                    
                    # Calculate the start and end samples for this epoch in the original data
                    event_sample = events[i, 0]
                    start_sample = event_sample + int(tmin * eeg_obj.mne_raw_orig.info['sfreq'])
                    end_sample = start_sample + cleaned_epoch.shape[1]
                    
                    # Make sure indices are within bounds
                    if start_sample >= 0 and end_sample <= n_times:
                        # Replace the corresponding segment in the cleaned data
                        cleaned_data[picks, start_sample:end_sample] = cleaned_epoch
            
            # Update the EEG object with cleaned data
            result_eeg.mne_raw._data = cleaned_data
            
        else:
            # Process the entire data at once (less efficient for large datasets)
            # Divide data into overlapping windows
            window_size = 1024  # Should be a power of 2 for UNet
            overlap = window_size // 2
            step = window_size - overlap
            
            # Create a new array to store the cleaned data
            cleaned_data = orig_data.copy()
            
            # Create a counter array to handle overlapping sections
            counter = np.zeros_like(orig_data)
            
            with torch.no_grad():
                # Process each window
                for i in range(0, n_times - window_size + 1, step):
                    # Extract window
                    window_data = orig_data[:, i:i+window_size]
                    
                    # Prepare the data for the model
                    input_data = torch.FloatTensor(window_data[np.newaxis, np.newaxis, :, :]).to(self.device)
                    
                    # Apply the model
                    cleaned_window = self.model(input_data).cpu().numpy()[0, 0]
                    
                    # Add the cleaned window to the output data (will be averaged later)
                    cleaned_data[:, i:i+window_size] += cleaned_window
                    counter[:, i:i+window_size] += 1
            
            # Average the overlapping sections
            cleaned_data = np.divide(cleaned_data, counter, out=np.zeros_like(cleaned_data), where=counter > 0)
            
            # Update the EEG object with cleaned data
            result_eeg.mne_raw._data = cleaned_data
            
        return result_eeg

    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained UNet model.
        
        Args:
            model_path: Path to the saved model checkpoint
            
        Returns:
            None
        """
        # Handle device-specific loading
        if self.device.type == "mps":
            # For Apple Silicon GPU
            checkpoint = torch.load(model_path, map_location=torch.device('mps'))
        elif not torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_path)
            
        # Initialize the model
        self.model = UNet(n_channels=1).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load data info if available
        if 'data_info' in checkpoint:
            self.data_info = checkpoint['data_info']
            
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        
        # Initialize criterion for evaluation
        self.criterion = nn.MSELoss()