#  Learning Rate for Neural Networks via Line Search

## Setup
Using conda the setup the environment can be setup by running the following command in the terminal: 

```bash
conda env create -f environment.yml
``` 
This will install all dependencies necessary for running the experiments.

To activate the environment:

```bash
conda activate optml
```

## Run experiments

The configuration of each experiment is stored in the `experiment_configs/` directory.  

To run an experiment:

```bash
python experiment.py --config <config_name>
```

The config_name is the filename of the config without `.json`. See example

```bash
python experiment.py --config cifar10_resnet18_armijo 
```

To run all experiments use:

```bash
python experiment.py --config all 
```

The result of the experiments are logged using tensorboard and saved in the `runs/` directory. To launch tensorboard use:

```bash
tensorboard --logdir=runs 
```