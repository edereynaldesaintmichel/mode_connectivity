import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from mnist_net import MNISTNet

# Load the trained model
model = MNISTNet()
model.load_state_dict(torch.load('model1_updated.pth', map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Load MNIST dataset for evaluation
test_dataset = MNIST(root="./data", train=False, transform=ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Define a function to display the image and probability distribution
def display_image_with_probabilities(image, label, probabilities):
    # Prepare the image for display
    image = image.squeeze().numpy()  # Remove batch and channel dimensions for plotting
    
    # Plot the image
    plt.figure(figsize=(10, 4))
    
    # Display the MNIST image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"True Label: {label}")
    plt.axis("off")
    
    # Display the probability distribution
    plt.subplot(1, 2, 2)
    x = np.arange(10)  # Class indices 0-9
    plt.bar(x, probabilities, tick_label=x)
    plt.ylim(0, 1)
    plt.title("Probability Distribution")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    
    # Show the combined plot
    plt.tight_layout()
    plt.show()

# Evaluate the model on the test set
for i, (images, labels) in enumerate(test_loader):
    if i >= 1000:  # Display predictions for 10 samples
        break

    # Get model prediction
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()  # Convert logits to probabilities

    # Display the image and probabilities
    display_image_with_probabilities(images[0], labels.item(), probabilities)
