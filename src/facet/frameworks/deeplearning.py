import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import os
from pathlib import Path
import mne
from ..eeg_obj import EEG

class DoubleConv(nn.Module):
    """Double Convolution block for U-Net"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """U-Net architecture for EEG artifact removal"""
    def __init__(self, n_channels: int = 1, n_classes: int = 1, base_channels: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels

        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(base_channels, base_channels * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(base_channels * 2, base_channels * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(base_channels * 4, base_channels * 8)
        )

        # Decoder
        self.up1 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up3 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(base_channels * 2, base_channels)
        
        self.outc = nn.Conv1d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder path
        x = self.up1(x4)
        x = self.up_conv1(torch.cat([x3, x], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x2, x], dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat([x1, x], dim=1))
        
        return self.outc(x)

class EEGDataset(torch.utils.data.Dataset):
    """Dataset class for EEG data"""
    def __init__(self, eeg_data: List[EEG], window_size: int = 1000):
        self.eeg_data = eeg_data
        self.window_size = window_size
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for eeg in self.eeg_data:
            data = eeg.mne_raw.get_data()
            # Create windows of data
            for i in range(0, data.shape[1] - self.window_size, self.window_size // 2):
                samples.append((data[:, i:i + self.window_size], eeg))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, eeg = self.samples[idx]
        return torch.FloatTensor(data), torch.FloatTensor(data)  # For now, using same data as target

class DeepLearningFramework:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, 
              train_data: List[EEG],
              val_data: List[EEG],
              batch_size: int = 32,
              epochs: int = 100,
              learning_rate: float = 1e-4,
              save_path: Optional[str] = None):
        """Train the U-Net model"""
        train_dataset = EEGDataset(train_data)
        val_dataset = EEGDataset(val_data)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {train_loss/len(train_loader):.6f}')
            print(f'Validation Loss: {val_loss/len(val_loader):.6f}')

            if save_path and (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))

    def predict(self, eeg_data: EEG) -> np.ndarray:
        """Apply the trained model to remove artifacts"""
        self.model.eval()
        data = torch.FloatTensor(eeg_data.mne_raw.get_data()).to(self.device)
        
        with torch.no_grad():
            # Process data in windows
            window_size = 1000
            cleaned_data = np.zeros_like(data)
            
            for i in range(0, data.shape[1] - window_size, window_size // 2):
                window = data[:, i:i + window_size].unsqueeze(0)
                output = self.model(window)
                cleaned_data[:, i:i + window_size] = output.squeeze(0).cpu().numpy()

        return cleaned_data

    def save_model(self, path: str):
        """Save the model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)

    def load_model(self, path: str):
        """Load the model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
