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

batch_size = 600
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
        
        # torch.matmul(torch.diag_embed(torch.full(tmp['conv1.bias'].size()[:-1], 1.0)), tmp['conv1.bias']) - tmp['conv1.bias']
        self.alphas = nn.ParameterDict()
        # {
        #     f"{param_name.replace('.', '')}_{step}": nn.Parameter(torch.full((value.size()[-1]), float(step)* (1+(random.random() - 0.5)*0.1) / (path_length - 1))) 
        #     for param_name, value in model1.state_dict() for step in range(path_length)
        # })
        random_factor = 0
        for step in range(path_length):
            for param_name, value in model1.state_dict().items():
                if len(value.size()) > 1:
                    default_value = torch.full(value.size()[:-1], float(step)* (1+(random.random() - 0.5)*random_factor) / (path_length - 1))
                else:
                    default_value = torch.full(value.size(), float(step)* (1+(random.random() - 0.5)*random_factor) / (path_length - 1))
                
                self.alphas[f"{param_name.replace('.', '')}_{step}"] = default_value
        
        
    def forward(self, x):
        """
        Outputs all the predictions of the model at each path step
        """
        output = []
        for step in range(self.path_length):
            data = x
            for i, (layer1_with_name, layer2_with_name) in enumerate(zip(model1.named_children(), model2.named_children())):
                layer_name = layer1_with_name[0]
                layer1 = layer1_with_name[1]
                layer2 = layer2_with_name[1]

                if (f'{layer_name}weight_{step}' not in self.alphas):
                    data = layer1(data)
                    continue
                weight_alpha_flat = self.alphas[f'{layer_name}weight_{step}']
                bias_alpha_flat = self.alphas[f'{layer_name}bias_{step}']

                weight_alpha = torch.diag_embed(weight_alpha_flat)
                bias_alpha = torch.diag_embed(bias_alpha_flat)

                one_minus_weight_alpha = torch.diag_embed(1 - weight_alpha_flat)
                one_minus_bias_alpha = torch.diag_embed(1 - bias_alpha_flat)
                if isinstance(layer1, nn.Linear):
                    merged_weight = torch.matmul(weight_alpha, layer1.weight) + torch.matmul(one_minus_weight_alpha, layer2.weight)
                    merged_bias = torch.matmul(bias_alpha, layer1.bias) + torch.matmul(one_minus_bias_alpha, layer2.bias)
                    data = F.linear(data, merged_weight, merged_bias)
                elif isinstance(layer1, nn.Conv2d):
                    merged_weight = torch.matmul(weight_alpha, layer1.weight) + torch.matmul(one_minus_weight_alpha, layer2.weight)
                    merged_bias = torch.matmul(bias_alpha, layer1.bias) + torch.matmul(one_minus_bias_alpha, layer2.bias)
                    data = F.conv2d(data, merged_weight, merged_bias, stride=layer1.stride, padding=layer1.padding)
                else:
                    # For layers without parameters (e.g., activation functions)
                    data = layer1(data)
            output.append(data)
        return output, self.alphas
    
    def backward(self, grad_output):
        return grad_output

merge_path = MergePath(path_length=10)
optimizer = optim.Adam(merge_path.parameters(), lr=0.00005)
distance_penalty = 0.001
std_penalty = 10
extremity_penalty = 10

dict_keys = [x for x in [key.replace('.', '') for key in model1.state_dict().keys()]]


def test_model(model):
    pre_path_finding_loss = []
    for i, data in enumerate(test_loader, 0):
        images, labels = data
        with torch.no_grad():
            outputs, alphas = model(images)
            error_loss = torch.stack([F.cross_entropy(output, labels) for output in outputs])
            pre_path_finding_loss.append(error_loss)

    return torch.mean(torch.stack(pre_path_finding_loss), dim=0)

initial_losses = test_model(merge_path)
print(initial_losses)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        outputs, alphas = merge_path(images)
        distance_loss = torch.norm(torch.stack([torch.norm((alphas[f'{layer}_{step}'] - alphas[f'{layer}_{step+1}'])) for layer in dict_keys for step in range(merge_path.path_length-1)]))
        std_loss = torch.std(torch.norm(torch.tensor([[torch.norm(alphas[f'{layer}_{step}'] - alphas[f'{layer}_{step+1}']) for layer in dict_keys] for step in range(merge_path.path_length-1)]), dim=1))
        extremity_loss = torch.norm(torch.stack([torch.norm(alphas[f'{layer}_{0}']) for layer in dict_keys])) + torch.norm(torch.stack([torch.norm(alphas[f'{layer}_{merge_path.path_length-1}'] - 1) for layer in dict_keys]))
        error_loss = torch.mean(torch.stack([F.cross_entropy(output, labels) for output in outputs]))
        loss = 1 * error_loss + distance_penalty * distance_loss + std_penalty * std_loss + extremity_penalty * extremity_loss
        print(f"Step {i + 1} (epoch {epoch}), Loss: {loss:.4f}, Distance: {distance_loss:.4f}, STD: {std_loss:.4f}, ExLoss: {extremity_loss:.4f}, Error Loss: {error_loss:.4f}")
        loss.backward(retain_graph=True)

        optimizer.step()

torch.save(merge_path.state_dict(), 'layerwise_merge_path.pth')

merge_path.eval()

after_training_losses = test_model(merge_path)

# I'd like to plot after_training_losses and initial_losses vs range(10), on the same graph


x_values = range(10)

initial_losses = initial_losses.numpy()
after_training_losses = after_training_losses.numpy()

plt.figure(figsize=(10, 6))

plt.plot(x_values, initial_losses, label='Initial Losses', marker='o')
plt.plot(x_values, after_training_losses, label='After Training Losses', marker='o')

plt.xlabel('Path Steps')
plt.ylabel('Loss')
plt.title('Comparison of Losses Before and After Training')
plt.legend()
plt.grid(True)

plt.show()