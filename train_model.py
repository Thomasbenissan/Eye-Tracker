# train_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import EyeTrackingDataset
import caputre_dataset

# CNN model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 1000)  # Adjust according to your image size
        self.fc2 = nn.Linear(1000, 2)  # Output layer: x and y coordinates

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Adjust according to your image size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Transformation for the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Create dataset and dataloader
dataset = EyeTrackingDataset(dataset_dir='eye_tracking_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model, Loss, and Optimizer
model = SimpleCNN()
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == "__main__":
    pass  # This line is just a placeholder. The actual training loop is above.
