import torch
from torch.optim import Optimizer
from . import utils as ut
from numpy import random
import time

class RDLSOptimizer(Optimizer):
    def __init__(
            self,
            params,
            device,
            initial_interval=1,
            tolerance=1e-3,
            max_iterations=100):
        """
        Initializes the RDLSOptimizer.

        Args:
            params: Iterable of parameters to optimize.
            device: Device to use ('cpu', 'cuda', or 'mps').
            initial_interval (float, optional): Initial search interval for line search. Default is 1.
            tolerance (float, optional): Tolerance for line search stopping criterion. Default is 1e-3.
            max_iterations (int, optional): Maximum iterations for line search. Default is 100.
        """
        defaults = dict(
            initial_interval=initial_interval,
            tolerance=tolerance,
            max_iterations=max_iterations
        )

        super().__init__(params, defaults)

        # Initialize optimizer state
        self.state["function_evaluations"] = 0
        self.state["gradient_evaluations"] = 0
        self.state['step_size'] = 0
        self.state['execution_time'] = 0
        self.state['device'] = device


    def step(self, closure):
        start_time = time.time()

        self.state['gradient_evaluations'] = 0
        self.state['function_evaluations'] = 0

        if closure is None:
            raise ValueError("Closure function required")

        # calculate loss
        loss = closure()
        with torch.no_grad():
            step_size = 0

            for group in self.param_groups:
                params = group["params"]

                # Get original parameters
                orig_params = ut.copy_parameters(params)

                # Get random direction
                direction = [d.to(self.state['device']) for d in ut.get_random_direction(params)]

                # Perform Golden Section line search
                tau = (5**(0.5) - 1) / 2

                a, b = -group["initial_interval"], group["initial_interval"]

                l = b - (b - a) * tau
                r = a + (b - a) * tau

                ut.update_parameters_no_grad(params, l, orig_params, direction)
                loss_l = closure()

                ut.update_parameters_no_grad(params, r, orig_params, direction)
                loss_r = closure()

                j = 0

                while b - a > group["tolerance"] and j < group["max_iterations"]:
                    j += 1
                    if loss_l > loss_r:
                        a = l
                        l = r
                        loss_l = loss_r
                        r = a + (b - a) * tau
                        ut.update_parameters_no_grad(params, r, orig_params, direction)
                        loss_r = closure()
                        self.state['function_evaluations'] += 1
                    else:
                        b = r
                        r = l
                        loss_r = loss_l
                        l = b - (b - a) * tau
                        ut.update_parameters_no_grad(params, l, orig_params, direction)
                        loss_l = closure()
                        self.state['function_evaluations'] += 1

                step_size = (b + a) / 2
                self.state['step_size'] = step_size

                ut.update_parameters_no_grad(params, step_size, orig_params, direction)

                new_loss = closure()

                if new_loss > loss:
                    ut.set_params(params, orig_params)

                if self.state['device'] == 'mps':
                    del direction, orig_params, loss_l, loss_r
                    torch.mps.empty_cache()
                elif self.state['device'] == 'cuda':
                    del direction, orig_params, loss_l, loss_r
                    torch.cuda.empty_cache()

        end_time = time.time()
        self.state['execution_time'] = end_time - start_time
        return loss
