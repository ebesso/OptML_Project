import torch 

def infer(model, test_loader, criterion, device):
    """
    Test the model with the given parameters.
    
    Args:
        model: The model to be tested.
        test_loader: DataLoader for test data.
        criterion: Loss function.
    
    Returns:
        None
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Loss: {running_loss/len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    return running_loss / len(test_loader), 100 * correct / total