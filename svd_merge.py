import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from mnist_net import MNISTNet

# Load MNIST test dataset
test_data = MNIST(root='./data', train=False,
                  download=True, transform=ToTensor())
# It's better to set shuffle=False for consistent evaluation
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

class MergePath(nn.Module):
    def __init__(self, diff_model_state_dict: dict, path_length: int):
        super(MergePath, self).__init__()
        self.path_length = path_length
        self.parameter_keys = list(key.replace('.', '') for key in diff_model_state_dict.keys())
        self.num_params = len(self.parameter_keys)

        # Initialize coefficients for each step and each parameter
        # We'll register each coefficient as a separate parameter with unique names
        for step in range(path_length-2):
            for key in self.parameter_keys:
                param_name = f'step{step}_{key}'
                initial_coeff = step / (path_length - 1) + 0 * torch.rand(1)
                initial_coeff = torch.clamp(torch.tensor(initial_coeff), 1e-6, 1 - 1e-6)
                initial_value = torch.log(initial_coeff / (1 - initial_coeff))
                self.register_parameter(param_name, nn.Parameter(initial_value))

    def forward(self):
        coeffs_path = []
        for step in range(path_length-2):
            coeffs_step = {}
            for key in self.parameter_keys:
                param_name = f'step{step}_{key}'
                coeff = torch.sigmoid(getattr(self, param_name))
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
        merged_state_dict[key] = coeffs[coeff_key] * \
            state_dict1[key] + (1 - coeffs[coeff_key]) * state_dict2[key]

    # Load the merged state dict into the new model
    merged_model.load_state_dict(merged_state_dict)

    return merged_model


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



        


# Usage
state_dict_1 = torch.load('model1.pth', map_location=torch.device('cpu'))
state_dict_2 = torch.load('model2.pth', map_location=torch.device('cpu'))


diff_state_dict = substract_state_dicts(state_dict_1=state_dict_2, state_dict_2=state_dict_1)

svd_params = compute_svd_of_model_params(state_dict=diff_state_dict)

# test_diff_model = get_model_from_svd(svd_params)

# Example to access SVD components
total_tunable_params = 0

for layer, svd_param in svd_params.items():
    if (isinstance(svd_param, tuple)):
        (U, S, V) = svd_param
        tunable_params = torch.prod(torch.tensor(S.size()))

        total_tunable_params += tunable_params
    else:
        total_tunable_params += svd_param.size()[0]

print(f"Total tunable_params: {total_tunable_params}")


