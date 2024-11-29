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

batch_size = 60
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
        
        self.path_length = path_length
        self.alphas = nn.ParameterDict()
        random_factor = 0
        for step in range(path_length):
            for param_name, value in model1.state_dict().items():
                default_value = torch.full(value.size(), float(step)* (1+(random.random() - 0.5)*random_factor) / (path_length - 1))
                
                self.alphas[f"{param_name.replace('.', '')}_{step}"] = default_value
        
        # Register gradient hooks on parameters
        for name, param in self.named_parameters():
            param.register_hook(lambda grad, param_name=name: 
                self._gradient_hook(grad, param_name))
        
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
                weight_alpha = self.alphas[f'{layer_name}weight_{step}']
                bias_alpha = self.alphas[f'{layer_name}bias_{step}']

                merged_weight = weight_alpha * layer1.weight + (1-weight_alpha) * layer2.weight
                merged_bias = bias_alpha * layer1.bias + (1-bias_alpha) * layer2.bias

                if isinstance(layer1, nn.Linear):
                    data = F.linear(data, merged_weight, merged_bias)
                elif isinstance(layer1, nn.Conv2d):
                    data = F.conv2d(data, merged_weight, merged_bias, stride=layer1.stride, padding=layer1.padding)
                else:
                    data = layer1(data)
            output.append(data)
        return output, self.alphas
    
    def _gradient_hook(self, grad, param_name: str):
        
        net_param_name, path_step_string = param_name.split('.')[1].split('_')
        path_step = int(path_step_string)
        if path_step == self.path_length-1 or path_step == 0:
            return torch.zeros(grad.size())
        output_grad = grad.clone()

        previous_step_param = self.alphas[f'{net_param_name}_{int(path_step)-1}']
        next_step_param = self.alphas[f'{net_param_name}_{int(path_step)+1}']
        forbidden_direction = next_step_param - previous_step_param
        
        projection = (torch.sum(forbidden_direction * grad) / torch.sum(forbidden_direction**2)) * forbidden_direction

        output_grad -= projection * 0.95 #keep a bit of gradient in that direction anyway  

        # print(f"Processing gradients for parameter: {param_name}")
        # Modify grad here if needed
        return output_grad

merge_path = MergePath(path_length=10)

merge_path.register_full_backward_hook
optimizer = optim.Adam(merge_path.parameters(), lr=0.001)
distance_penalty = 0.002
std_penalty = 0.1

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

for epoch in range(1):
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        outputs, alphas = merge_path(images)
        distance_loss = torch.norm(torch.stack([torch.norm((alphas[f'{layer}_{step}'] - alphas[f'{layer}_{step+1}'])) for layer in dict_keys for step in range(merge_path.path_length-1)]))
        std_loss = torch.std(torch.norm(torch.tensor([[torch.norm(alphas[f'{layer}_{step}'] - alphas[f'{layer}_{step+1}']) for layer in dict_keys] for step in range(merge_path.path_length-1)]), dim=1))
        error_loss = torch.mean(torch.stack([F.cross_entropy(output, labels)**2 for output in outputs]))
        loss = 1 * error_loss + distance_penalty * distance_loss + std_penalty * std_loss
        print(f"Step {i + 1} (epoch {epoch}), Loss: {loss:.4f}, Distance: {distance_loss:.4f}, STD: {std_loss:.4f}, Error Loss: {error_loss:.4f}")
        loss.backward(retain_graph=True)

        optimizer.step()

## Test: ability to generalize from a single small batch. It does better when there are only a few trainable parameters.
# images, labels = next(iter(train_loader))
# for epoch in range(300):   
#     outputs, alphas = merge_path(images)
#     distance_loss = torch.norm(torch.stack([torch.norm((alphas[f'{layer}_{step}'] - alphas[f'{layer}_{step+1}'])) for layer in dict_keys for step in range(merge_path.path_length-1)]))
#     std_loss = torch.std(torch.norm(torch.tensor([[torch.norm(alphas[f'{layer}_{step}'] - alphas[f'{layer}_{step+1}']) for layer in dict_keys] for step in range(merge_path.path_length-1)]), dim=1))
#     error_loss = torch.mean(torch.stack([F.cross_entropy(output, labels)**2 for output in outputs]))
#     loss = 1 * error_loss + distance_penalty * distance_loss + std_penalty * std_loss
#     print(f"(epoch {epoch}), Loss: {loss:.4f}, Distance: {distance_loss:.4f}, STD: {std_loss:.4f}, Error Loss: {error_loss:.4f}")
#     loss.backward()

#     optimizer.step()

torch.save(merge_path.state_dict(), 'full_merge.pth')

merge_path.eval()

after_training_losses = test_model(merge_path)

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