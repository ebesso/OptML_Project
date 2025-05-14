import torch
from torch.optim import Optimizer
import random 

class RandomOptimizer(Optimizer):
    def __init__(self, params):
        super().__init__(params, dict(lr=0.01))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(-p.grad * random.uniform(0, 1))

class RandomOptimizer(Optimizer):
    def __init__(self, params):
        super().__init__(params, dict(lr=0.01))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(-p.grad * random.uniform(0, 0.1))