import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Base directory where all logs are stored
base_log_dir = "runs_06_11"

# Dictionary of labels and their corresponding subdirectories
log_dirs_18 = {
    'cifar_resnet18_baseline-20250610-231336',
    'cifar_resnet18_armijo-20250611-011647',
    'cifar_resnet18_goldstein-20250610-233428',
    'cifar_resnet18_strongwolfe-20250611-014020',
}

log_dirs_34 = {
    'cifar_resnet34_baseline-20250610-224506',
    'cifar_resnet34_armijo-20250611-021030',
    'cifar_resnet34_goldstein-20250610-235807',
    'cifar_resnet34_strongwolfe-20250611-003131'               
}

plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    try:
        ea.Reload()
        # Replace with actual tag name (e.g., 'val_accuracy', 'Accuracy', etc.)
        tag = 'Accuracy/Validation'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f"⚠️  Tag '{tag}' not found ")
    except Exception as e:
        print(f"❌ Error reading from {full_path}: {e}")

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation/Accuracy for resnet18')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    
    try:
        ea.Reload()
        # Replace with actual tag name (e.g., 'val_accuracy', 'Accuracy', etc.)
        tag = 'Accuracy/Validation'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f"⚠️  Tag '{tag}' not found ")
    except Exception as e:
        print(f"❌ Error reading from {full_path}: {e}")

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation/Accuracy for resnet34')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()