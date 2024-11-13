import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np

# Load MNIST test dataset
test_data = MNIST(root='./data', train=False, download=True, transform=ToTensor())
# It's better to set shuffle=False for consistent evaluation
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# Define the CNN model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Function to merge two models
def merge_models(model1, model2, coeff=0.5):
    """
    Merge two models by taking a weighted average of their parameters.

    Args:
        model1 (nn.Module): First model.
        model2 (nn.Module): Second model.
        coeff (float): Coefficient for model1. The coefficient for model2 is (1 - coeff).

    Returns:
        nn.Module: Merged model.
    """
    assert 0.0 <= coeff <= 1.0, "Coefficient must be between 0 and 1."
    
    # Create a new model instance
    merged_model = MNISTNet()
    
    # Get state dictionaries
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    merged_state_dict = {}
    
    # Iterate through all parameters and merge
    for key in state_dict1:
        if key in state_dict2:
            merged_state_dict[key] = coeff * state_dict1[key] + (1 - coeff) * state_dict2[key]
        else:
            raise KeyError(f"Key {key} found in model1 but not in model2.")
    
    # Load the merged state dict into the new model
    merged_model.load_state_dict(merged_state_dict)
    
    return merged_model

# Load the trained models
model1 = MNISTNet()
model1.load_state_dict(torch.load('model1.pth', map_location=torch.device('cpu')))
model1.eval()

model2 = MNISTNet()
model2.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))
model2.eval()

# Define the range of coefficients
coefficients = np.linspace(0, 1, num=21)  # 0.0, 0.05, ..., 1.0
losses = []

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Iterate over each coefficient, merge models, and compute test loss
for coeff in coefficients:
    print(f"Processing coefficient: {coeff:.2f}")
    # Merge the models
    merged_model = merge_models(model1, model2, coeff)
    merged_model.eval()
    
    total_loss = 0.0
    total_samples = 0
    num_images_to_process = 50
    images_processed = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if images_processed >= num_images_to_process:
                images_processed = 0
                break
            images_processed += 1
            outputs = merged_model(images)
            loss = criterion(outputs, labels)
            # Multiply by number of samples in the batch
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    average_loss = total_loss / total_samples
    losses.append(average_loss)
    print(f"Average Test Loss for coeff {coeff:.2f}: {average_loss:.4f}")

# Plot Test Loss vs Coefficient
plt.figure(figsize=(8, 6))
plt.plot(coefficients, losses, marker='o')
plt.title('Test Loss vs. Merging Coefficient')
plt.xlabel('Coefficient for Model1')
plt.ylabel('Average Test Loss')
plt.grid(True)
plt.show()

# Optional: Save the plot
# plt.savefig('test_loss_vs_coefficient.png')
