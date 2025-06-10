from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np

def load(batch_size=64, val_split=0.2, seed = 1, dataset='cifar10'):
    
    np.random.seed(seed) # Necessary for reproducabiliity

    if dataset == 'cifar10': 
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616]),
        ])
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'fashion':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530]),  # Normalization for FashionMNIST
        ])
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Choose 'cifar10' or 'fashion'.")

    indices = np.random.permutation(len(full_train_dataset))
    split = int(np.floor(val_split * len(full_train_dataset)))
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
