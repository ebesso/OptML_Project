from torch.utils.tensorboard import SummaryWriter
from loader import load
import argparse, time, json
import torch
from model import SimpleCNN, TinyCNN
from optimizer.sls_optimizer import SLSOptimizer
from optimizer.rdls_optimizer import RDLSOptimizer
from optimizer.baseline_optimizer import BaseLineOptimizer
from trainer import train
from inferencer import infer
from torchvision.models import resnet18


def load_configuration(config_name):
    config_path = f"experiment_configs/{config_name}.json"
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

def run(config, writer):
    train_loader, val_loader, test_loader = load(batch_size=config["batch_size"])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}") 


    
    criterion = torch.nn.CrossEntropyLoss()

    # Not pretrained
    model = resnet18(num_classes=10)

    model.to(device)

    if config['line_search_fn'] == 'golden_section':
        optimizer = RDLSOptimizer(
            model.parameters(),
            initial_interval=config['initial_interval'],
            max_step_size=config['max_step_size'],
            tolerance=config['tolerance']
        )
    elif config['line_search_fn'] == 'armijo' or config['line_search_fn'] == 'strong_wolfe' or config['line_search_fn'] == 'goldstein':
        optimizer = SLSOptimizer(
            model.parameters(),
            n_batches_per_epoch=len(train_loader) // config["batch_size"],
            line_search_fn=config['line_search_fn'],
            initial_step_size=config['initial_step_size'],
            max_step_size=config['max_step_size'],
            reset_option=config['reset_option'],
            gamma=config['gamma'],
            max_iterations=config['max_iterations'],
            c1=config['c1'],
            c2=config['c2'],
            beta_b=config['beta_b'],
            beta_f=config['beta_f'] 
        )
    # If no line search is specified, use a standard optimizer
    elif config['line_search_fn'] == 'none':
        optimizer = BaseLineOptimizer(
            model.parameters(),
            learning_rate=config['learning_rate']
        )

    else:
        raise ValueError(f"Unsupported line search function: {config['line_search_fn']}")
    
    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, device, config['num_epochs'], writer)    

    print("Training completed. Evaluating on test set...")

    # Evaluate the model on the test set
    test_loss, test_accuracy = infer(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Log hyperparameters 
    writer.add_hparams(config, {
        'Loss/Test': test_loss,
        'Accuracy/Test': test_accuracy
    })

    # Log final metrics
    writer.add_scalar('Loss/Test', test_loss, 0)
    writer.add_scalar('Accuracy/Test', test_accuracy, 0)

    print("Test evaluation completed.")

if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser(description="Run the experiment with the given configuration.")
    parser.add_argument("--config", type=str, required=True, help="Configuration name to load.")
    args = parser.parse_args()
    print(f"Loading configuration: {args.config}")
    config = load_configuration(args.config)

    writer = SummaryWriter(log_dir=f"runs/{args.config}-{time.strftime('%Y%m%d-%H%M%S')}")

    # Run the experiment
    run(config, writer)

    print("Experiment completed.")

    writer.close()