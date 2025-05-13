import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms

class CustomMNISTDataset(Dataset):
    def __init__(self, train=True, transform=None, download=True):
        self.dataset = datasets.MNIST(root='./data', train=train, download=download, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def load_mnist_splits(batch_size=64, val_split=0.1):
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full training dataset (to split into train and val)
    full_train_dataset = CustomMNISTDataset(train=True, transform=transform)

    # Compute split sizes
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Split datasets
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Load test dataset
    test_dataset = CustomMNISTDataset(train=False, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
