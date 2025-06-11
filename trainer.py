import torch
from tqdm import tqdm
from optimizer.rdls_optimizer import RDLSOptimizer

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, writer):
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
        accumulated_step_size = 0.0
        accumulated_execution_time = 0.0
        accumulated_gradient_evaluations = 0.0
        accumulated_function_evaluations = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            def closure(backward=False):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if backward:
                    loss.backward()
                return loss

            def closure_no_grad():
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    return loss
            
            if isinstance(optimizer, RDLSOptimizer):
                loss = optimizer.step(closure_no_grad)
            else:
                loss = optimizer.step(closure)

            accumulated_step_size += optimizer.state["step_size"]
            running_loss += loss
            accumulated_execution_time += optimizer.state["execution_time"]
            accumulated_function_evaluations += optimizer.state["function_evaluations"]
            accumulated_gradient_evaluations += optimizer.state["gradient_evaluations"]

            step = epoch * len(train_loader) + batch_idx

            writer.add_scalar('Loss/Train', loss, step)
            writer.add_scalar('Step Size', optimizer.state["step_size"], step)
            writer.add_scalar('Function Evaluations', optimizer.state["function_evaluations"], step)
            writer.add_scalar('Gradient Evaluations', optimizer.state["gradient_evaluations"], step)
            writer.add_scalar('Execution Time', optimizer.state["execution_time"], step)
       
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        validation_loss, accuracy = validate(model, val_loader, criterion, device)

        writer.add_scalar('Average Step Size', accumulated_step_size / len(train_loader), epoch + 1)
        writer.add_scalar('Loss/Validation', validation_loss, epoch + 1)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch + 1)
        writer.add_scalar('Average Execution Time', accumulated_execution_time / len(train_loader), epoch + 1)
        writer.add_scalar('Average Train Loss', running_loss / len(train_loader), epoch + 1)
        writer.add_scalar('Average Function Evaluations', accumulated_function_evaluations / len(train_loader), epoch + 1)
        writer.add_scalar('Average Gradient Evaluations', accumulated_gradient_evaluations / len(train_loader), epoch + 1)

        print(f"Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.2f}%")

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


    return running_loss / len(val_loader), 100 * correct / total

