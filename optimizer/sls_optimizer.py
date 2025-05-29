import torch
from torch.optim import Optimizer 
from . import utils as ut
    
class SLSOptimizer(Optimizer):
    def __init__(self, 
                 params, 
                 initial_step_size=1, 
                 line_search_fn="armijo", 
                 max_iterations=100,
                 n_batches_per_epoch=500,
                 c=0.1,
                 beta_b=0.9,
                 beta_f=2,
                 gamma=2,
                 max_step_size=10,
                 reset_option=2):
        """
        Initialize the SLS optimizer.
        Args:
            params (iterable): Parameters to optimize or dicts defining parameter groups.
            initial_step_size (float): Initial step size for the optimizer.
            line_search_fn (str): Line search function to use ("armijo" or "goldstein").
            max_iterations (int): Maximum number of iterations for the line search.
            n_batches_per_epoch (int): Number of batches per epoch.
            c (float): Constant for the Armijo condition.
            beta_b (float): Decay factor for the step size in Armijo condition.
            beta_f (float): Increase factor for the step size in Goldstein condition.
            gamma (float): Factor for resetting the step size.
            max_step_size (float): Maximum step size.
            reset_option (int): Option for resetting the step size.
        """
        
        defaults = dict(initial_step_size=initial_step_size,
                        max_iterations=max_iterations,
                        line_search_fn=line_search_fn,
                        n_batches_per_epoch=n_batches_per_epoch,
                        c=c,
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

            direction = [-g for g in grad]

            # Calculate derivative of line search function at the current point
            loss_prime = sum((d * g).sum() for d, g in zip(direction, grad) if d is not None and g is not None)

            step_size = self.state["step_size"]
            
            with torch.no_grad():
                if (-loss_prime >= 1e-16):
                    # Reset the step size
                    step_size = ut.reset_step(step_size=step_size,
                                              max_step_size=group["max_step_size"], 
                                              gamma=group["gamma"],
                                              reset_option=group["reset_option"], 
                                              n_batches_per_epoch=group["n_batches_per_epoch"],
                                              beta_f=group["beta_f"])
                    
                    found = 0

                    # Perform the line search
                    for _ in range(group['max_iterations']):
                        # Update parameters
                        new_params = [p + step_size * d for p, d in zip(orig_params, direction)]
                        ut.set_params(params, new_params)
                        
                        # Calculate new loss
                        new_loss = closure()
                        self.state["function_evaluations"] += 1

                        # Check the Armijo condition
                        if group["line_search_fn"] == "armijo":
                            found, step_size = ut.check_armijo_condition(f_new=new_loss.item(), 
                                                                         f0=loss.item(), 
                                                                         f0_prime=loss_prime,
                                                                         step_size=step_size, 
                                                                         c=group["c"], 
                                                                         beta_b=group["beta_b"])
                            if found == 1:
                                break
                        
                        # Check the Goldstein condition
                        elif group["line_search_fn"] == "goldstein":
                            found, step_size = ut.check_goldstein_condition(f_new=new_loss.item(), 
                                                                           f0=loss.item(), 
                                                                           f0_prime=loss_prime,
                                                                           max_step_size=group["max_step_size"], 
                                                                           step_size=step_size, 
                                                                           c=group["c"], 
                                                                           beta_b=group["beta_b"], 
                                                                           beta_f=group["beta_f"])
                            if found == 1:
                                break
                                
                    
                    if found == 0:
                        # If no step size found, revert to original parameters
                        ut.set_params(params, orig_params)
                        
            self.state["step_size"] = step_size

        # Returns the loss at the start
        return loss