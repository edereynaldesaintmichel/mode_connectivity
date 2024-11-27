import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from mnist_net import MNISTNet
import random
import copy

batch_size = 100
samples_size = 60
train_data = MNIST(root='./data', train=True,
                  download=True, transform=ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = MNIST(root='./data', train=False,
                  download=True, transform=ToTensor())
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
samples = []
i = 0
sample = []
for images, labels in train_loader:
    sample.append((images, labels))
    if i % samples_size == samples_size-1:
        samples.append(sample)
        sample = []
    i+=1


def merge_models(model1: MNISTNet, model2: MNISTNet, coeffs: dict):
    # Create a new model instance
    merged_model = MNISTNet()

    # Get state dictionaries
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    merged_state_dict = {}

    # Iterate through all parameters and merge
    for key in state_dict1:
        coeff_key = key.replace('.', '')
        merged_state_dict[key] = coeffs[coeff_key] * state_dict1[key] + (1 - coeffs[coeff_key]) * state_dict2[key]

    # Load the merged state dict into the new model
    merged_model.load_state_dict(merged_state_dict)

    return merged_model


# Load the trained models
model1 = MNISTNet()
model1.load_state_dict(torch.load(
    'model1.pth', map_location=torch.device('cpu')))

model2 = MNISTNet()
model2.load_state_dict(torch.load(
    'model2.pth', map_location=torch.device('cpu')))


class MergePath(nn.Module):
    def __init__(self, path_length):
        super(MergePath, self).__init__()
        assert len(list(model1.children())) == len(list(model2.children())), \
            "Both models must have the same number of layers for layerwise merging."
        
        # model1 = model1
        # model2 = model2
        self.path_length = path_length
        
        # # Freeze the weights of model1 and model2
        # for param in model1.parameters():
        #     param.requires_grad = False
        # for param in model2.parameters():
        #     param.requires_grad = False
        
        # Initialize a list of learnable alphas for the path
        self.alphas = nn.ParameterDict(
            {
                f"{layer}_{step}": nn.Parameter(torch.tensor(float(step)* (1+(random.random())*0.1) / (path_length - 1))) 
                for layer in range(len(list(model1.children()))) for step in range(path_length)
            })

    def forward(self, x):
        """
        Outputs all the predictions of the model at each path step
        """
        output = []
        for step in range(self.path_length):
            data = x
            for i, (layer1, layer2) in enumerate(zip(model1.children(), model2.children())):
                alpha = self.alphas[f'{i}_{step}']
                
                if isinstance(layer1, nn.Linear):
                    # Blend weights and biases
                    merged_weight = alpha * layer1.weight + (1 - alpha) * layer2.weight
                    merged_bias = alpha * layer1.bias + (1 - alpha) * layer2.bias
                    # Apply the linear transformation using functional API
                    data = F.linear(data, merged_weight, merged_bias)
                elif isinstance(layer1, nn.Conv2d):
                    # Blend weights and biases
                    merged_weight = alpha * layer1.weight + (1 - alpha) * layer2.weight
                    merged_bias = alpha * layer1.bias + (1 - alpha) * layer2.bias
                    # Apply the convolution using functional API
                    data = F.conv2d(data, merged_weight, merged_bias, stride=layer1.stride, padding=layer1.padding)
                elif isinstance(layer1, nn.BatchNorm2d):
                    # Blend parameters
                    merged_weight = alpha * layer1.weight + (1 - alpha) * layer2.weight
                    merged_bias = alpha * layer1.bias + (1 - alpha) * layer2.bias
                    merged_running_mean = alpha * layer1.running_mean + (1 - alpha) * layer2.running_mean
                    merged_running_var = alpha * layer1.running_var + (1 - alpha) * layer2.running_var
                    # Apply batch normalization using functional API
                    data = F.batch_norm(
                        data, merged_running_mean, merged_running_var, 
                        weight=merged_weight, bias=merged_bias, training=self.training
                    )
                else:
                    # For layers without parameters (e.g., activation functions)
                    data = layer1(data)
            output.append(data)
    
        return output, self.alphas

merge_path = MergePath(path_length=10)
optimizer = optim.Adam(merge_path.parameters(), lr=0.0005)
distance_penalty = 0.0
std_penalty = 0.0
extremity_penalty = 0


for i, data in enumerate(train_loader, 0):
    images, labels = data
    outputs, alphas = merge_path(images)
    distance_loss = torch.norm(torch.stack([(alphas[f'{layer}_{step}'] - alphas[f'{layer}_{step+1}']) for layer in range(len(list(model1.children()))) for step in range(merge_path.path_length-1)]))
    std_loss = torch.std(torch.stack([(alphas[f'{layer}_{step}'] - alphas[f'{layer}_{step+1}']) for layer in range(len(list(model1.children()))) for step in range(merge_path.path_length-1)]))
    extremity_loss = torch.norm(torch.stack([alphas[f'{layer}_{0}'] for layer in range(len(list(model1.children())))])) + torch.norm(torch.stack([alphas[f'{layer}_{merge_path.path_length-1}'] - 1 for layer in range(len(list(model1.children())))]))
    error_loss = torch.max(torch.stack([F.cross_entropy(output, labels)**2 for output in outputs]))
    loss = 1 * error_loss + distance_penalty * distance_loss + std_penalty * std_loss + extremity_penalty * extremity_loss
    print(f"Step {i + 1}, Loss: {loss}, Distance: {distance_loss}, STD: {std_loss}, ExLoss: {extremity_loss}, Error Loss: {error_loss}")
    loss.backward(retain_graph=True)

    # Print gradients for each parameter in the model
    for name, param in merge_path.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad}")
        else:
            print(f"Gradient for {name} is None (likely frozen or unused in the computation).")

    optimizer.step()
    

torch.save(merge_path.state_dict(), 'layerwise_merge_path.pth')


