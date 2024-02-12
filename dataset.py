# eye_tracking_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class EyeTrackingDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.samples = []

        # Load dataset
        for filename in os.listdir(dataset_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(dataset_dir, filename)
                label_path = os.path.join(dataset_dir, filename.replace('.png', '.txt'))
                with open(label_path, 'r') as f:
                    label = f.read().strip().split(',')
                    label = [float(coord) for coord in label]
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)
