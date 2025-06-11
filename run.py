from torch.utils.tensorboard import SummaryWriter
from loader import load
import argparse, time, json
import torch
from model import ResNet18, ResNet34, TinyCNN_model 
from optimizer.sls_optimizer import SLSOptimizer
from optimizer.rdls_optimizer import RDLSOptimizer
from optimizer.baseline_optimizer import BaseLineOptimizer
from trainer import train
from inferencer import infer
from torchvision.models import resnet18
import os


def load_configuration(config_name):
    config_path = f"experiment_configs/{config_name}.json"
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

def run(config, writer):
    train_loader, val_loader, test_loader = load(batch_size=config["batch_size"], dataset=config['dataset'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}") 
    
    criterion = torch.nn.CrossEntropyLoss()

    # Not pretrained
    if config['model'] == 'resnet18':
        model = ResNet18()
    elif config['model'] == 'resnet34':
        model = ResNet34()
    elif config['model'] == 'tiny_cnn':
        model = TinyCNN_model(10)
    else:
        raise ValueError(f"Unsupported model: {config['model']}")

    model.to(device)

    optimizer_config = config['optimizer']

    if optimizer_config['name'] == 'rdls':
        optimizer = RDLSOptimizer(
            model.parameters(),
            device=device, 
            initial_interval=optimizer_config['initial_interval'],
            tolerance=optimizer_config['tolerance']
        )
    elif optimizer_config['name'] == 'sls':
        optimizer = SLSOptimizer(
            model.parameters(),
            n_batches_per_epoch=len(train_loader),
            line_search_fn=optimizer_config['line_search_fn'],
            initial_step_size=optimizer_config['initial_step_size'],
            max_step_size=optimizer_config['max_step_size'],
            reset_option=optimizer_config['reset_option'],
            gamma=optimizer_config['gamma'],
            max_iterations=optimizer_config['max_iterations'],
            c1=optimizer_config['c1'],
            c2=optimizer_config['c2'],
            beta_b=optimizer_config['beta_b'],
            beta_f=optimizer_config['beta_f'],
            momentum=optimizer_config["momentum"]
        )
    # If no line search is specified, use a standard optimizer
    elif optimizer_config['name'] == 'baseline':
        optimizer = BaseLineOptimizer(
            model.parameters(),
            learning_rate=optimizer_config['learning_rate']
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
    writer.add_hparams(optimizer_config, {
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

    experiment_configs = [config.split('.')[0] for config in os.listdir("experiment_configs")]

    print(f"Available configurations: {', '.join(experiment_configs)}")

    if args.config != 'all':
        experiment_configs = args.config.split(' ')
    
    for experiment_config in experiment_configs:
        print(f"Loading configuration: {experiment_config}")
        config = load_configuration(experiment_config)

        writer = SummaryWriter(log_dir=f"runs/{config['name']}-{time.strftime('%Y%m%d-%H%M%S')}")

        # Run the experiment
        run(config, writer)

        writer.close()

        print(f"Experiment '{config['name']}' completed and logged to TensorBoard.")