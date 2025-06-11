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

# Plots of the validation accuracy for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Accuracy/Validation'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation/Accuracy for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the average execution time for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Execution Time'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Execution Time')
plt.title('Average Execution Time for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the average function evaluations for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Function Evaluations'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Function Evaluations')
plt.title('Average Function Evaluations for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Average Gradient Evaluations for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Gradient Evaluations'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Gradient Evaluations')
plt.title('Average Gradient Evaluations for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the average step size for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Step Size'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Step Size')
plt.title('Average Step Size for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Average Train Loss for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Train Loss'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Train Loss')
plt.title('Average Train Loss for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Execution Time for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Execution Time'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Execution Time')
plt.title('Execution Time for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Function Evaluations for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Function Evaluations'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Function Evaluations')
plt.title('Function Evaluations for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Gradient Evaluations for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Gradient Evaluations'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Gradient Evaluations')
plt.title('Gradient Evaluations for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Loss/Validation for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Loss/Validation'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Loss/Validation')
plt.title('Loss/Validation for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Step Size for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Step Size'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Step Size')
plt.title('Step Size for resnet18')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

#############



#############



# Plots of the validation accuracy for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Accuracy/Validation'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation/Accuracy for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the average execution time for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Execution Time'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Execution Time')
plt.title('Average Execution Time for resne34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the average function evaluations for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Function Evaluations'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Function Evaluations')
plt.title('Average Function Evaluations for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Average Gradient Evaluations for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Gradient Evaluations'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Gradient Evaluations')
plt.title('Average Gradient Evaluations for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the average step size for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Step Size'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Step Size')
plt.title('Average Step Size for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Average Train Loss for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Average Train Loss'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Average Train Loss')
plt.title('Average Train Loss for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Execution Time for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Execution Time'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Execution Time')
plt.title('Execution Time for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Function Evaluations for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Function Evaluations'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Function Evaluations')
plt.title('Function Evaluations for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Gradient Evaluations for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Gradient Evaluations'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Gradient Evaluations')
plt.title('Gradient Evaluations for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Loss/Validation for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Loss/Validation'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Loss/Validation')
plt.title('Loss/Validation for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plots of the Step Size for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    label = sub_dir.split('-')[0] 
    label = label.split('_')[2]
    label = label.capitalize()
    try:
        ea.Reload()
        tag = 'Step Size'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, label = label)
        else:
            print(f" Tag '{tag}' not found ")
    except Exception as e:
        print(f" Error reading from {full_path}: {e}")

plt.xlabel('Epochs')
plt.ylabel('Step Size')
plt.title('Step Size for resnet34')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

