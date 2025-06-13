import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Base directory where all logs are stored
# Change to "runs"
base_log_dir = "runs"

# Output directory for saving plots
plot_dir = "experiment_plots"
os.makedirs(plot_dir, exist_ok=True)

# Dictionary of labels and their corresponding subdirectories
#Change the log directories as per your experiments
log_dirs_18 = [
    'resnet18_baseline-20250612-210627',
    'resnet18_armijo-20250612-222046',
    'resnet18_armijo_momentum-20250612-235349',
    'resnet18_goldstein-20250612-212706',
    'resnet18_goldstein_momentum-20250612-233015',
    'resnet18_strongwolfe-20250612-215034',
    'resnet18_strongwolfe_momentum-20250613-001721'
]

log_dirs_34 = [
    'resnet34_baseline-20250612-200509',
    'resnet34_armijo-20250613-012115',
    'resnet34_armijo_momentum-20250612-193127',
    'resnet34_goldstein-20250612-203316',
    'resnet34_goldstein_momentum-20250613-004744',
    'resnet34_strongwolfe-20250612-184451',
    'resnet34_strongwolfe_momentum-20250612-224404'               
]


log_dirs_rdls_tiny = [
    'tiny_baseline-20250612-031215',
    'tiny_rdls-20250612-002529'
]

log_dirs_rdls_resnet = [
    'RDLS_resnet18-20250612-070230',
    'RDLS_resnet34-20250612-052443'
]


# Plots of the validation accuracy for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)

    label_part = sub_dir.split('resnet18_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Accuracy (%)', fontsize = 22)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnet18_validation_accuracy.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plots of the average function evaluations for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)

    label_part = sub_dir.split('resnet18_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Average Function Evaluations', fontsize = 22)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnet18_avg_funceval.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plots of the average step size for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)

    label_part = sub_dir.split('resnet18_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Average Step Size', fontsize = 22)
plt.legend(loc='upper right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnet18_avg_step.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plots of the Average Train Loss for resnet18 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_18:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    
    label_part = sub_dir.split('resnet18_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Average Train Loss', fontsize = 22)
plt.legend(loc='upper right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnet18_avg_trainloss.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()


#############



#############



# Plots of the validation accuracy for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    
    label_part = sub_dir.split('resnet34_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Accuracy (%)', fontsize = 22)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnet34_validation_accuracy.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plots of the average function evaluations for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    
    label_part = sub_dir.split('resnet34_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Average Function Evaluations', fontsize = 22)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnet34_avg_funceval.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plots of the average step size for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    
    label_part = sub_dir.split('resnet34_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Average Step Size', fontsize = 22)
plt.legend(loc='upper right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnet34_avg_stepsize.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plots of the Average Train Loss for resnet34 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_34:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    
    label_part = sub_dir.split('resnet34_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Average Train Loss', fontsize = 22)
plt.legend(loc='upper right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnet34_avg_trainloss.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()


#############



#############

#Plots for tiny rdls


# Plots of the validation accuracy for tiny-CNN 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_rdls_tiny:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)

    label_part = sub_dir.split('tiny_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Accuracy (%)', fontsize = 22)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "tinyRDLS_validation_accuracy.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plots of the Average Train Loss for tiny-CNN 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_rdls_tiny:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    
    label_part = sub_dir.split('tiny_')[1].split('-')[0]
    label = label_part.replace('_', ' + ').capitalize()
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Average Train Loss', fontsize = 22)
plt.legend(loc='upper right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "tinyRDLS_avg_trainloss.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()


#############



#############

#Plots for rdls resnet



# Plots of the validation accuracy for tiny-CNN 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_rdls_resnet:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)

    label_part = sub_dir.split('-')[0]
    label = label_part.replace('_', ' on ')
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Accuracy (%)', fontsize = 22)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnetRDLS_validation_accuracy.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plots of the Average Train Loss for tiny-CNN 
plt.figure(figsize=(14, 7))
for sub_dir in log_dirs_rdls_resnet:
    full_path = os.path.join(base_log_dir, sub_dir)
    ea = event_accumulator.EventAccumulator(full_path)
    
    label_part = sub_dir.split('-')[0]
    label = label_part.replace('_', ' on ')
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

plt.xlabel('Epochs', fontsize = 22)
plt.ylabel('Average Train Loss', fontsize = 22)
plt.legend(loc='upper right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plot_dir, "resnetRDLS_avg_trainloss.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()