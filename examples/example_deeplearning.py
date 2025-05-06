from src.facet.facet import facet
from src.facet.frameworks.deeplearning import DeepLearningFramework
import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Load EEG data
facet_toolbox = facet()
eeg = facet_toolbox.import_eeg(
    path="path/to/your/eeg_file.edf",
    fmt="edf",
    preload=True
)
raw = eeg.mne_raw

# 2. Extract artifact segments and clean segments
artifact_segments = []
clean_segments = []

for annot in raw.annotations:
    if annot['description'] == "Artifact":
        onset_sample = int((annot['onset'] - raw.first_time) * raw.info['sfreq'])
        duration_samples = int(annot['duration'] * raw.info['sfreq'])
        # Get artifact segment (input)
        artifact_data = raw.get_data(start=onset_sample, stop=onset_sample + duration_samples)
        artifact_segments.append(artifact_data)
        
        # For clean segment (target) - using a simple template method for demonstration
        # In practice, replace with your best clean reference data
        clean_data = artifact_data.copy()  # Copy to preserve shape
        # Apply some correction - this is just a placeholder
        clean_data = np.zeros_like(artifact_data)  # or any other correction method
        clean_segments.append(clean_data)

# 3. Create a custom dataset class for paired data
class ArtifactDataset(Dataset):
    def __init__(self, artifact_data, clean_data):
        self.artifact_data = artifact_data
        self.clean_data = clean_data
        
    def __len__(self):
        return len(self.artifact_data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.artifact_data[idx]), torch.FloatTensor(self.clean_data[idx])

# 4. Split into train/validation sets
split = int(0.8 * len(artifact_segments))
train_artifacts = artifact_segments[:split]
train_clean = clean_segments[:split]
val_artifacts = artifact_segments[split:]
val_clean = clean_segments[split:]

# 5. Create datasets
train_dataset = ArtifactDataset(train_artifacts, train_clean)
val_dataset = ArtifactDataset(val_artifacts, val_clean)

# 6. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 7. Initialize the deep learning framework and train
dl_framework = DeepLearningFramework()

# 8. Train with custom training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = dl_framework.model
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    for batch_idx, (artifact_batch, clean_batch) in enumerate(train_loader):
        artifact_batch, clean_batch = artifact_batch.to(device), clean_batch.to(device)
        
        optimizer.zero_grad()
        output = model(artifact_batch)
        loss = criterion(output, clean_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for artifact_batch, clean_batch in val_loader:
            artifact_batch, clean_batch = artifact_batch.to(device), clean_batch.to(device)
            output = model(artifact_batch)
            val_loss += criterion(output, clean_batch).item()
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Training Loss: {train_loss/len(train_loader):.6f}')
    print(f'Validation Loss: {val_loss/len(val_loader):.6f}')

# 9. Save the trained model
dl_framework.save_model('models/trained_unet.pth')