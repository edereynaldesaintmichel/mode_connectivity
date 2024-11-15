import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
from mnist_net import MNISTNet


# Load MNIST test dataset
test_data = MNIST(root='./data', train=False,
                  download=True, transform=ToTensor())
# It's better to set shuffle=False for consistent evaluation
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# Define the CNN model

# Function to merge two models


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


# Load the trained models
model1 = MNISTNet()
model1.load_state_dict(torch.load(
    'model1.pth', map_location=torch.device('cpu')))
model1.eval()

model2 = MNISTNet()
model2.load_state_dict(torch.load(
    'model2.pth', map_location=torch.device('cpu')))
model2.eval()


criterion = nn.CrossEntropyLoss()


def get_distance(coeffs1: dict, coeffs2: dict):
    distance = 0
    for key in coeffs1:
        distance += (coeffs1[key] - coeffs2[key])**2

    return distance ** 0.5


def get_loss(model, num_images_to_process: int = 50):
    total_loss = 0.0
    total_samples = 0
    images_processed = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if images_processed >= num_images_to_process:
                images_processed = 0
                break
            images_processed += 1
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Multiply by number of samples in the batch
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples


def get_merge_loss(model1: MNISTNet, model2: MNISTNet, path: list, distance_penalty: float = 0.1, stdev_penalty: float = 0.2):
    total_loss = get_loss(model1)
    total_distance = 0
    previous_loss = get_loss(model1)
    previous_coeffs = {key: 0 for key in path[0]}
    losses = [total_loss]
    distances = torch.zeros(len(path)+1)
    i = 0
    for coeffs in path:
        merged_model = merge_models(model1, model2, coeffs)
        merged_model.eval()
        loss = get_loss(merged_model)
        losses.append(loss)
        distance = get_distance(coeffs1=coeffs, coeffs2=previous_coeffs)
        total_loss += (loss + previous_loss) * distance
        total_distance += distance
        distances[i] = distance

        previous_coeffs = coeffs
        previous_loss = loss
        i += 1

    last_coeffs = {key: 1 for key in path[0]}
    loss = get_loss(model2)
    losses.append(loss)
    distance = get_distance(coeffs1=last_coeffs, coeffs2=coeffs)
    total_loss += (loss + previous_loss) * distance
    total_distance += distance
    distances[i] = distance

    return sum(loss**2 for loss in losses) + distance_penalty * total_distance + stdev_penalty * torch.std(distances), losses, distances


class MergePath(nn.Module):
    def __init__(self, model1: MNISTNet, model2: MNISTNet, path_length: int):
        super(MergePath, self).__init__()
        self.path_length = path_length
        self.parameter_keys = list(key.replace('.', '')
                                   for key in model1.state_dict().keys())
        self.num_params = len(self.parameter_keys)

        # Initialize coefficients for each step and each parameter
        # We'll register each coefficient as a separate parameter with unique names
        for step in range(path_length-2):
            for key in self.parameter_keys:
                param_name = f'step{step}_{key}'
                initial_coeff = step / (path_length - 1) + 0 * torch.rand(1)
                initial_coeff = torch.clamp(torch.tensor(initial_coeff), 1e-6, 1 - 1e-6)
                # initial_value = torch.log(initial_coeff / (1 - initial_coeff))
                self.register_parameter(param_name, nn.Parameter(initial_coeff))

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


path_length = 10  # Number of steps in the path
merge_path = MergePath(model1, model2, path_length)

optimizer = optim.Adam(merge_path.parameters(), lr=0.2)

num_epochs = 150  # Adjust as needed


min_theoretical_distance = len(model1.state_dict()) ** 0.5
print(f'min_theoretical_distance: {min_theoretical_distance}')

for epoch in range(num_epochs):
    optimizer.zero_grad()
    coeffs_path = merge_path()
    loss, losses, distances = get_merge_loss(model1, model2, coeffs_path, distance_penalty=0.01, stdev_penalty=0.5)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Distance: {sum(distances)}, Path Loss: {sum(losses)}')

torch.save(merge_path.state_dict(), 'path_finder.pth')




def evaluate_paths(paths, eval_steps, model1, model2):
    results = {}
    path_coeffs = {}
    
    for path_name, path_path in paths.items():
        merge_path = MergePath(model1, model2, path_length)
        merge_path.load_state_dict(torch.load(path_path, map_location=torch.device('cpu')))
        merge_path.eval()
        path = merge_path()
        to_prepend = {key: torch.tensor([0.0]) for key in path[0]}
        to_append = {key: torch.tensor([1.0]) for key in path[0]}
        path.insert(0, to_prepend)
        path.append(to_append)

        # Calculate distances along the path
        with torch.no_grad():
            dist_list = torch.tensor([get_distance(path[i], path[i+1]) for i in range(len(path)-1)])
            dist_list = torch.cat((torch.tensor([0.0]), dist_list.clone().detach()))
            dists = torch.cumsum(dist_list, 0)

        # Evaluate losses along the optimal path
        optimal_path_eval_coeffs = []
        optimal_path_losses = []
        for i in range(eval_steps):
            progress = i / (eval_steps-1) * dists[-1]
            tmp = torch.nonzero(dists > progress, as_tuple=True)
            if len(tmp[0]) == 0:
                index = len(dists)-1
            else:
                index = tmp[0][0]
            interpolation_coeff = (
                progress - dists[index-1]) / (dists[index] - dists[index-1])

            optimal_coeffs = {key: path[index-1][key] + interpolation_coeff * (path[index][key] - path[index-1][key]) for key in path[0]}

            optimal_path_model = merge_models(model1=model1, model2=model2, coeffs=optimal_coeffs)
            optimal_path_losses.append(get_loss(optimal_path_model, 500))
            optimal_path_eval_coeffs.append(optimal_coeffs)

        # Store results for this path
        results[path_name] = optimal_path_losses
        path_coeffs[path_name] = optimal_path_eval_coeffs
        linear_path = [{key: i / (eval_steps - 1) for key in path[0]} for i in range(eval_steps)]

    linear_path_losses = []
    path_coeffs['linear'] = linear_path
    for i in range(eval_steps):
        linear_path_model = merge_models(model1=model1, model2=model2, coeffs=linear_path[i])
        linear_path_losses.append(get_loss(linear_path_model, 500))
    
    results['linear'] = linear_path_losses
    
    return results, path_coeffs

def plot_results(results, path_coeffs, eval_steps):
    plt.figure(figsize=(10, 6))
    # progress_along_path = np.linspace(0, 1, eval_steps)
    zero = {key: 0 for key in path_coeffs['linear'][0]}
    one = {key: 1 for key in path_coeffs['linear'][0]}
    for path_name, data in results.items():
        with torch.no_grad():
            path = path_coeffs[path_name]
            progress_along_path = [get_distance(path[i], zero) / (get_distance(path[i], zero) + get_distance(path[i], one)) for i in range(eval_steps)]
            progress_along_path = [item.item() if torch.is_tensor(item) else item for item in progress_along_path]
            plt.plot(progress_along_path, data, label=f'{path_name} Loss')
    
    plt.xlabel('Progress Along Path')
    plt.ylabel('Loss')
    plt.title('Test Loss vs Progress Along Path')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
paths = {
    'path_finder': 'path_finder.pth',
    # 'path_finder_2': 'path_finder_2.pth',
    # Add more paths here if needed
}

# tmp, losses, distances = get_merge_loss(model1=model1, model2=model2, path=merge_path())
# plt.figure(figsize=(8, 6))
# distances = torch.cat((torch.tensor([0.0]), torch.cumsum(distances.clone().detach(), dim=0)))
# plt.plot(distances, losses, marker='o')
# plt.title('Test Loss vs. path step')
# plt.xlabel('Coefficient for Model1')
# plt.ylabel('Average Test Loss')
# plt.grid(True)
# plt.show()


eval_steps = 20
results, path_coeffs = evaluate_paths(paths, eval_steps, model1, model2)
plot_results(results, path_coeffs, eval_steps)
