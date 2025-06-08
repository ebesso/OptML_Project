import torch
from torch.optim import Optimizer 
from . import utils as ut
    
class SLSOptimizer(Optimizer):
    def __init__(self, 
                 params, 
                 line_search_fn="armijo",
                 
                 initial_step_size=1,
                 max_step_size=10,
                 reset_option=2,
                 gamma=2,

                 max_iterations=100,
                 n_batches_per_epoch=500,

                 c1=0.1,
                 c2=0.9,

                 beta_b=0.9,
                 beta_f=2):
        """
        Initialize the SLS optimizer.
        Args:
            params (iterable): Parameters to optimize or dicts defining parameter groups.
            line_search_fn (str): Line search function to use.

            initial_step_size (float): Initial step size for the optimizer.
            max_step_size (float): Maximum step size.
            reset_option (int): Option for resetting the step size.
            gamma (float): Factor for resetting the step size.

            max_iterations (int): Maximum number of iterations for the line search.
            n_batches_per_epoch (int): Number of batches per epoch.

            c1 (float): Constant for the Armijo and Goldstein condition.
            c2 (float): Constant for Strong Wolfe curvature condition.

            beta_b (float): Decay factor for the step size in Armijo condition.
            beta_f (float): Increase factor for the step size in Goldstein condition.            
        """
        
        defaults = dict(initial_step_size=initial_step_size,
                        max_iterations=max_iterations,
                        line_search_fn=line_search_fn,
                        n_batches_per_epoch=n_batches_per_epoch,
                        c1=c1,
                        c2=c2,
                        beta_b=beta_b,
                        beta_f=beta_f,
                        gamma=gamma,
                        max_step_size=max_step_size,
                        reset_option=reset_option)
        super().__init__(params, defaults)

        self.state["step_size"] = initial_step_size

        self.state["function_evaluations"] = 0
        self.state["gradient_evaluations"] = 0

    def step(self, closure=None):
        
        if closure is None:
            raise ValueError("Closure function is required for SLS optimizer")

        # calculate loss
        loss = closure()
        self.state["function_evaluations"] += 1

        # calculate gradient
        loss.backward()
        self.state["gradient_evaluations"] += 1

        for group in self.param_groups:
            params = group["params"]

            orig_params = ut.copy_parameters(params)
            grad = ut.get_gradient(params)
            grad_norm = ut.get_grad_norm(grad)

            # Normalized direction
            direction = [-g/grad_norm for g in grad]

            # Calculate derivative of line search function at the current point
            loss_prime = sum((d * g).sum() for d, g in zip(direction, grad) if d is not None and g is not None)

            step_size = self.state["step_size"]
            
            if (-loss_prime >= 1e-16):
                # Reset the step size
                step_size = ut.reset_step(step_size=step_size,
                                          max_step_size=group["max_step_size"], 
                                          gamma=group["gamma"],
                                          reset_option=group["reset_option"], 
                                          n_batches_per_epoch=group["n_batches_per_epoch"])
                
                found = False
                def line_fn(step_size):
                    # Update parameters
                    ut.update_parameters(params=params, 
                                         step_size=step_size, 
                                         original_params=orig_params, 
                                         direction=direction)
                    return closure()
                
                # Perform armijo line search
                if group["line_search_fn"] == "armijo":
                    found, step_size, func_evals, grad_evals = ut.armijo_line_search(
                        line_fn=line_fn, 
                        orig_loss=loss, 
                        orig_loss_prime=loss_prime, 
                        step_size=step_size, 
                        group=group
                    )
                    self.state["function_evaluations"] += func_evals
                    self.state["gradient_evaluations"] += grad_evals
                
                # Perform goldstein line search
                elif group["line_search_fn"] == "goldstein":
                    found, step_size, func_evals, grad_evals = ut.goldstein_line_search(
                        line_fn=line_fn, 
                        orig_loss=loss, 
                        orig_loss_prime=loss_prime, 
                        step_size=step_size, 
                        group=group
                    )
                    self.state["function_evaluations"] += func_evals
                    self.state["gradient_evaluations"] += grad_evals
                
                # Perform Strong Wolfe line search
                elif group["line_search_fn"] == "strong_wolfe":
                    found, step_size, func_evals, grad_evals = ut.strong_wolfe_line_search(
                        line_fn=line_fn, 
                        params=params, 
                        orig_loss=loss, 
                        orig_loss_prime=loss_prime, 
                        step_size=step_size, 
                        direction=direction, 
                        group=group
                    )
                    self.state["function_evaluations"] += func_evals
                    self.state["gradient_evaluations"] += grad_evals
                
                else:
                    raise ValueError(f"Unknown line search function: {group['line_search_fn']}")
                            
                
                if not found:
                    # If no step size found, revert to original parameters
                    ut.update_parameters(params, 1e-6, orig_params, direction)
                    
            self.state["step_size"] = step_size

        # Returns the loss at the start
        return loss