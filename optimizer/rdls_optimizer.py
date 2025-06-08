import torch
from torch.optim import Optimizer
from . import utils as ut
from numpy import random
import time

class RDLSOptimizer(Optimizer):
    def __init__(
            self,
            params,
            initial_interval=10,
            max_step_size=0.05,
            tolerance=1e-3):

        defaults = dict(
            initial_interval=initial_interval,
            max_step_size=max_step_size,
            tolerance=tolerance
            )

        super().__init__(params, defaults)

        self.state["function_evaluations"] = 0
        self.state["gradient_evaluations"] = 0
        self.state['step_size'] = 0
        self.state['execution_time'] = 0

    def step(self, closure):
        start_time = time.time()

        if closure is None:
            raise ValueError("Closure function required")

        loss = closure(backward=False)
        with torch.no_grad():
            step_size = 0

            for group in self.param_groups:
                params = group["params"]

                orig_params = ut.copy_parameters(params)
                direction = ut.get_random_direction(params)
                # direction = ut.get_gradient(params)

                tau = (5**(0.5) - 1) / 2

                a, b = -group["initial_interval"], group["initial_interval"]

                l = b - (b - a) * tau
                r = a + (b - a) * tau

                # params_l = [p + l * d for p, d in zip(orig_params, direction)]
                # ut.set_params(params, params_l)

                ut.update_parameters(params, l, orig_params, direction)
                loss_l = closure().item()

                ut.update_parameters(params, r, orig_params, direction)
                # params_r = [p + r * d for p, d in zip(orig_params, direction)]
                # ut.set_params(params, params_r)
                loss_r = closure().item()

                while b - a > group["tolerance"]:
                    if loss_l > loss_r:
                        a = l
                        l = r
                        loss_l = loss_r
                        r = a + (b - a) * tau
                        ut.update_parameters(params, r, orig_params, direction)
                        # params_r = [p + r*d for p, d in zip(orig_params, direction)]
                        # ut.set_params(params, params_r)
                        loss_r = closure().item()
                        self.state['function_evaluations'] += 1
                    else:
                        b = r
                        r = l
                        loss_r = loss_l
                        l = b - (b - a) * tau
                        ut.update_parameters(params, l, orig_params, direction)
                        # ut.set_params(params, params_l)
                        # params_l = [p + l*d for p, d in zip(orig_params, direction)]
                        loss_l = closure().item()
                        self.state['function_evaluations'] += 1

                    step_size = (b + a) / 2
                    self.state['step_size'] = step_size

                    ut.update_parameters(params, step_size, orig_params, direction)

                    # final_params = [p + step_size * d for p, d in zip(orig_params, direction)]
                    # ut.set_params(params, final_params)

                    new_loss = closure()

                    if new_loss > loss:
                        ut.set_params(params, orig_params)

        end_time = time.time()
        self.state['execution_time'] = end_time - start_time
        return loss









