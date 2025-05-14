import torch

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    """
    Train the model with the given parameters.
    
    Args:
        model: The model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to train the model on (CPU or GPU).
        num_epochs: Number of epochs to train the model.
    
    Returns:
        Model 
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Ensure same device
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(model, loss, inputs, labels)
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        validate(model, val_loader, criterion, device)

    return model

def validate(model, val_loader, criterion, device):
    """
    Validate the model with the given parameters.
    
    Args:
        model: The model to be validated.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to validate the model on (CPU or GPU).
    
    Returns:
        None
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Ensure same device
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Validation Loss: {running_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
