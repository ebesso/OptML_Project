import torch
from torch.optim import Optimizer
import time
from . import utils as ut

class BaseLineOptimizer(Optimizer):
    """A baseline optimizer that implements a simple gradient descent algorithm."""
    def __init__(
            self,
            params,
            learning_rate=0.01
            ):
        """
        Initializes the BaseLineOptimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
        """
        defaults = dict(learning_rate=learning_rate)
        super().__init__(params, defaults)

        # Initialize optimizer state variables
        self.state["function_evaluations"] = 0
        self.state["gradient_evaluations"] = 0
        self.state['step_size'] = 0
        self.state['execution_time'] = 0

    def step(self, closure):
        self.state["function_evaluations"] = 0
        self.state["gradient_evaluations"] = 0
        start_time = time.time()
        loss = None
        if closure is not None:
            loss = closure(backward=True)
            self.state["function_evaluations"] += 1
            self.state["gradient_evaluations"] += 1

        for group in self.param_groups:
            grad = ut.get_gradient(group["params"])
            grad_norm = ut.get_grad_norm(grad)
            direction = [-g for g in grad]
            step_size = group["learning_rate"]
            if grad_norm > 0:
                for p, d in zip(group["params"], direction):
                    p.data.add_(d, alpha=step_size)

            self.state["step_size"] = step_size

        self.state['execution_time'] = time.time() - start_time
        return loss