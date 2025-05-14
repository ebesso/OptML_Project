import torch
from torch.optim import Optimizer, SGD
import random 

class LineSearchOptimizer(Optimizer):
    def __init__(self, params, loss_fn, gamma, theta, alpha_max, j_0, delta_0):
        self.gamma = gamma
        self.theta = theta
        self.alpha = gamma**j_0 * alpha_max
        self.loss_fn = loss_fn

        if alpha_max <= 0:
            raise ValueError("Alpha max must be positive")
        if j_0 > 0:
            raise ValueError("J_0 must be non-positive")
        if delta_0 <= 0:
            raise ValueError("Delta_0 must be positive")

        super().__init__(params)

    @torch.no_grad()
    def step(self, batch_inputs, batch_labels):
        # Needs access to mini-batch points
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(-p.grad * random.uniform(0, 1))
    
    def move()

class RandomOptimizer(Optimizer):
    def __init__(self, params):
        super().__init__(params, dict(lr=0.01))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(-p.grad * random.uniform(0, 0.1))