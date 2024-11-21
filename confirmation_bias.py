import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from mnist_net import MNISTNet
import math


# Define the perplexity-based loss
class PerplexityLoss(nn.Module):
    def __init__(self):
        super(PerplexityLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets):
        # Apply log softmax to logits
        log_probs = self.log_softmax(logits)
        # Gather log probabilities of the true class
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1))
        # Negative log probability becomes entropy
        entropy = -target_log_probs.mean()
        # Convert entropy to perplexity
        perplexity = torch.exp(entropy)
        return perplexity


# Load the trained model
model = MNISTNet()
model.load_state_dict(torch.load('model1.pth', map_location=torch.device('cpu')))
model.train()  # Set model to training mode

# Optimizer and custom loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = PerplexityLoss()

# Load MNIST dataset
transform = ToTensor()
dataset = MNIST(root="./data", train=True, transform=transform, download=True)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        # Move data to CPU (or GPU if available)
        images, labels = images.to(torch.device('cpu')), labels.to(torch.device('cpu'))

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Validation loop (optional)
model.eval()
val_loss = 0.0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(torch.device('cpu')), labels.to(torch.device('cpu'))
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

val_loss /= len(val_loader)
print(f"Validation Loss: {val_loss:.4f}")

# Save the updated model
torch.save(model.state_dict(), 'model1_updated.pth')
