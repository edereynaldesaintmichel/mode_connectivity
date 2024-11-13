import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

# Load MNIST dataset
train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='./data', train=False, download=True, transform=ToTensor())

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

# Set initial seed for reproducibility
torch.manual_seed(4)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize and train the parent model
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

# # Create DataLoader for parent model
# train_loader_parent = DataLoader(train_data, batch_size=32, shuffle=True)

# for epoch in range(1):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader_parent, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 100 == 99:
#             print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
#             running_loss = 0.0

#         if i % 5 == 4:
#             break
# # print('Finished Training Parent Model')

# Save the trained parent model
torch.save(model.state_dict(), 'parent_model.pth')

# Initialize model1 and model2
model1 = MNISTNet().to(device)
model2 = MNISTNet().to(device)

# Load the parent model weights
model1.load_state_dict(torch.load('parent_model.pth'))
model2.load_state_dict(torch.load('parent_model.pth'))

# Define separate seeds for model1 and model2
seed_model1 = 42
seed_model2 = 100

total_size = len(train_data)
split_ratio = 0.5  # For splitting the dataset into two equal parts
size1 = int(total_size * split_ratio)
size2 = total_size - size1

# Split the dataset into two distinct subsets
train_data1, train_data2 = random_split(train_data, [size1, size2])

# Create DataLoaders for each subset
train_loader1 = DataLoader(train_data1, batch_size=32, shuffle=True)
train_loader2 = DataLoader(train_data2, batch_size=32, shuffle=True)

# Define optimizers for both models
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

# Train model1
for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader1, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer1.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer1.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Model1 Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training model1')

# Train model2
for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader2, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Model2 Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training model2')

# Save the trained models
torch.save(model1.state_dict(), 'model1.pth')
torch.save(model2.state_dict(), 'model2.pth')



