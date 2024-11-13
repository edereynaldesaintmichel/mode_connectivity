import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn

# Load MNIST test dataset
test_data = MNIST(root='./data', train=False,
                  download=True, transform=ToTensor())
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

def compute_svd_of_model_params(model):
    svd_params = {}
    
    for name, param in model.named_parameters():
        if param.dim() >= 2:  # SVD is applicable to 2D matrices, so check the dimension
            U, S, V = torch.svd(param.data)
            svd_params[name] = (U, S, V)
        else:
            svd_params[name] = param.data  # Store as is if it's a bias or 1D param
    
    return svd_params

# Usage
model = MNISTNet()
svd_params = compute_svd_of_model_params(model)

print(svd_params)

# Example to access SVD components
for layer, (U, S, V) in svd_params.items():
    if isinstance(U, torch.Tensor):  # Check if SVD was computed
        print(f"SVD for {layer}:")
        print("U:", U)
        print("S:", S)
        print("V:", V)
    else:
        print(f"Parameter for {layer} (bias or 1D): {U}")
