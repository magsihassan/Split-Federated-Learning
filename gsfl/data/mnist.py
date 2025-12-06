import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from gsfl.config import NUM_CLIENTS, BATCH_SIZE


def get_client_datasets():
    """
    Download MNIST and split training data into NUM_CLIENTS disjoint subsets.
    Returns a list of torch.utils.data.Dataset for each client.
    """
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Equal split among clients
    data_per_client = len(full_train) // NUM_CLIENTS
    lengths = [data_per_client] * NUM_CLIENTS
    client_datasets = random_split(full_train, lengths)

    return client_datasets


def get_test_loader():
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    return test_loader
