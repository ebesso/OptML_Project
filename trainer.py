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
    loss_history = []
    step_size_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            def closure(backward=False):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if backward:
                    loss.backward()
                return loss
            
            loss = optimizer.step(closure)

            #loss = optimizer.step(model, inputs, labels)
            
            #optimizer.zero_grad()
            #outputs = model(inputs)
            #loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()
            #loss = loss.item()
            running_loss += loss
            loss_history.append(loss.item())
            step_size_history.append(optimizer.state["step_size"])

        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        validate(model, val_loader, criterion, device)


    return model, loss_history, step_size_history

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

    with torch.no_grad(): 
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Validation Loss: {running_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
