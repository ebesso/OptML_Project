from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np

def load(batch_size=64, val_split=0.2, seed = 1):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    np.random.seed(seed) # Necessary for reproducabiliity

    full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform) 
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    indices = np.random.permutation(len(full_train_dataset))
    split = int(np.floor(val_split * len(full_train_dataset)))
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
