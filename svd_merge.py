import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from mnist_net import MNISTNet
import torch.optim as optim
import random


batch_size = 10
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

class list(nn.Module):
    def __init__(self, diff_model_state_dict: dict, path_length: int):
        super(list, self).__init__()
        self.path_length = path_length
        self.parameter_keys = list(key.replace('.', '') for key in diff_model_state_dict.keys())
        self.dict = {key.replace('.', ''): value for key, value in diff_model_state_dict.items()}
        self.num_params = len(self.parameter_keys)

        # Initialize coefficients for each step and each parameter
        # We'll register each coefficient as a separate parameter with unique names
        for step in range(path_length-2):
            for key in self.parameter_keys:
                param_name = f'step{step}_{key}'
                tmp = self.dict[key]
                if isinstance(tmp, tuple):
                    U, initial_value, V = tmp
                else:
                    initial_value = tmp
                initial_value = initial_value * (step + 1) / (path_length - 1)
                self.register_parameter(param_name, nn.Parameter(initial_value))

    def forward(self):
        coeffs_path = []
        for step in range(self.path_length-2):
            coeffs_step = {}
            for key in self.parameter_keys:
                param_name = f'step{step}_{key}'
                coeff = getattr(self, param_name)
                coeffs_step[key] = coeff
            coeffs_path.append(coeffs_step)
        return coeffs_path

def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append((i, n // i))
    return divisors


def substract_state_dicts(state_dict1: dict, state_dict2: dict):
    diff_state_dict = {}
    for key in state_dict1:
        diff_state_dict[key] = state_dict1[key] - state_dict2[key]

    return diff_state_dict

def add_state_dicts(state_dict1: dict, state_dict2: dict):
    sum_state_dict = {}

    for key in state_dict1:
        sum_state_dict[key] = state_dict1[key] + state_dict2[key]

    return sum_state_dict



def compute_svd_of_model_params(state_dict: dict): 
    svd_params = {}
    to_flatify = set()
    
    for name, param in state_dict.items():
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


def get_model_state_dict_from_svd(svd_params):
    new_state_dict = {}
    for name, svd_param in svd_params.items():
        if isinstance(svd_param, tuple):
            U, S, V = svd_param
            dim_v = V.dim()
            reconstructed_param = torch.matmul(
                torch.matmul(U, torch.diag_embed(S)),
                V.transpose(dim_v-2, dim_v-1)
            )
            new_state_dict[name] = reconstructed_param
        else:
            new_state_dict[name] = svd_param
    
    return new_state_dict


def merge_svd_models(state_dict_model_1: dict, diff_model_svd_params: dict, optimized_diff_model_params: dict):
    merged_model = MNISTNet()
    updated_diff_model_svd = {}

    for name, svd_param in diff_model_svd_params.items():
        if (isinstance(svd_param, tuple)):
            (U, S, V) = svd_param
            S = optimized_diff_model_params[name]
            updated_diff_model_svd[name] = (U, S, V)
        else:
            updated_diff_model_svd[name] = optimized_diff_model_params[name]
    
    optimized_diff_model_state_dict = get_model_state_dict_from_svd(updated_diff_model_svd)
    merged_model_state_dict = add_state_dicts(state_dict1=state_dict_model_1, state_dict2=optimized_diff_model_state_dict)
    merged_model.load_state_dict(merged_model_state_dict)

    return merged_model


def get_loss(model, data_subset):
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_subset:
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Multiply by number of samples in the batch
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples
        
def get_distance(coeffs1: dict, coeffs2: dict):
    distance = 0
    for key in coeffs1:
        distance += sum((coeffs1[key] - coeffs2[key])**2)

    return distance ** 0.5


def get_merge_loss(model1: MNISTNet, diff_model_svd_params: dict, path: list, distance_penalty: float = 0.1, stdev_penalty: float = 0.2):
    subset = samples[random.randint(0, len(samples)-1)]
    total_loss = get_loss(model1, subset)
    total_distance = 0
    previous_loss = total_loss
    previous_coeffs = {key: 0 for key in path[0]}
    losses = [total_loss]
    distances = torch.zeros(len(path)+1)
    i = 0

    coeffs = [0 for _ in range(path.path_length)]

    for coeffs in path:
        svd_params = {}
        for key, param in diff_model_svd_params:
            if isinstance(param, tuple):
                U, S, V = param
                svd_params[key] = U, coeffs[key.replace('.', '')], V
            else:
                svd_params[key] = coeffs[key.replace('.', '')]

        
        merged_model = merge_models(model1, diff_model_svd_params, coeffs)
        merged_model.eval()
        loss = get_loss(merged_model, subset)
        losses.append(loss)
        distance = get_distance(coeffs1=coeffs, coeffs2=previous_coeffs)
        total_loss += (loss + previous_loss) * distance
        total_distance += distance
        distances[i] = distance

        previous_coeffs = coeffs
        previous_loss = loss
        i += 1

    last_coeffs = {key: 1 for key in path[0]}
    loss = get_loss(diff_model_svd_params, subset)
    losses.append(loss)
    distance = get_distance(coeffs1=last_coeffs, coeffs2=coeffs)
    total_loss += (loss + previous_loss) * distance
    total_distance += distance
    distances[i] = distance

    return sum(loss**2 for loss in losses) + distance_penalty * total_distance + stdev_penalty * torch.std(distances), losses, distances

# Usage
state_dict_1 = torch.load('model1.pth', map_location=torch.device('cpu'))
state_dict_2 = torch.load('model2.pth', map_location=torch.device('cpu'))


diff_state_dict = substract_state_dicts(state_dict1=state_dict_2, state_dict2=state_dict_1)

svd_params = compute_svd_of_model_params(state_dict=diff_state_dict)

# test_diff_model = get_model_from_svd(svd_params)

# Example to access SVD components
total_tunable_params = 0


path_length = 10  # Number of steps in the path
merge_path = list(svd_params, path_length)


optimizer = optim.Adam(merge_path.parameters(), lr=0.02)

total_params = sum(p.numel() for p in merge_path.parameters())

num_epochs = 100  # Adjust as needed

for layer, svd_param in svd_params.items():
    if (isinstance(svd_param, tuple)):
        (U, S, V) = svd_param
        tunable_params = torch.prod(torch.tensor(S.size()))

        total_tunable_params += tunable_params
    else:
        total_tunable_params += svd_param.size()[0]

print(f"Total tunable_params: {total_tunable_params}")


