import torch
from torch.optim import Optimizer
from . import utils as ut
from numpy import random

class RDLSOptimizer(Optimizer):
    def __init__(
            self, 
            params,
            line_search_algorithm='quadratic',
            initial_interval=1):

        defaults = dict(
            line_search_algorithm=line_search_algorithm,
            initial_interval=initial_interval
            )

        super().__init__(params, defaults)

    def step(self, closure):

        if closure is None:
            raise ValueError("Closure function required")
        
        with torch.no_grad():
            loss0 = closure()
            step_size = 0
            
            for group in self.param_groups:
                params = group["params"]

                orig_params = ut.copy_parameters(params)
                direction = ut.get_random_direction(len(params))

                new_params1 = [p + group['initial_interval'] * d for p, d in zip(orig_params, direction)]
                new_params2 = [p - group['initial_interval'] * d for p, d in zip(orig_params, direction)]

                ut.set_params(params, new_params1)
                loss1 = closure()

                ut.set_params(params, new_params2)
                loss2 = closure()

                point0, point1, point2 = (0, loss0), (group['initial_interval'], loss1), (-group['initial_interval'], loss2)

                poly = ut.get_polynomial(point0, point1, point2)

                if ut.is_concave(poly):
                    break

                step_size = ut.get_minima(poly)

                new_params = [p + step_size * d for p, d in zip(orig_params, direction)]

                ut.set_params(params, new_params)
                
        return loss0, step_size 

            






    
    