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
    

def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append((i, n // i))
    return divisors


def get_models_diff(model1: MNISTNet, model2: MNISTNet):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    diff_state_dict = {}
    diff_model = MNISTNet()

    for key in state_dict1:
        diff_state_dict[key] = state_dict1[key] - state_dict2[key]

    diff_model.load_state_dict(diff_state_dict)

def compute_svd_of_model_params(model: MNISTNet): 
    svd_params = {}
    to_flatify = set()
    
    for name, param in model.named_parameters():
        if param.dim() >= 2:  # SVD is applicable to 2D matrices, so check the dimension
            U, S, V = torch.svd(param.data)
            svd_params[name] = (U, S, V)
        else:
            # length = param.data.size()[0]
            # reshaped_param = param.data.view(get_divisors(length)[-1])
            # U, S, V = torch.svd(reshaped_param)
            # svd_params[name] = (U, S, V)
            # to_flatify.add(name)

            svd_params[name] = param.data
    
    return svd_params


# Usage
model = MNISTNet()
model.load_state_dict(torch.load('model1.pth', map_location=torch.device('cpu')))
svd_params = compute_svd_of_model_params(model)

# Example to access SVD components
total_tunable_params = 0
condition_number = 0
i = 0
for layer, svd_param in svd_params.items():
    if (isinstance(svd_param, tuple)):
        (U, S, V) = svd_param
        print(f"SVD for {layer}:")
        tunable_params = torch.prod(torch.tensor(S.size()))

        total_tunable_params += tunable_params
        s_dim = S.dim()
        max_eigh, trash = S.max(dim=s_dim -1)
        min_eigh, trash = S.min(dim=s_dim -1)
        condition_numbers = max_eigh / min_eigh
        # log_cond = torch.log(condition_numbers)
        condition_number += torch.mean(condition_numbers)
        i+=1
    else:
        total_tunable_params += svd_param.size()[0]

condition_number = condition_number / i

print(f"Total tunable_params: {total_tunable_params} \n Avg Cond: {condition_number}")


