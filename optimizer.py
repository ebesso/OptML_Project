import torch
from torch.optim.optimizer import Optimizer

class SimpleOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SimpleOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                # Basic SGD update
                d_p = p.grad
                p.data = p.data - lr * d_p

        return loss
