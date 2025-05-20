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
            grad_norm_sq = ut.get_grad_norm_sq(grad)

            step_size = self.state["step_size"]
            
            with torch.no_grad():
                if grad_norm_sq >= 1e-16:
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
                        new_params = [p - step_size * g for p, g in zip(orig_params, grad)]
                        ut.set_params(params, new_params)
                        
                        # Calculate new loss
                        new_loss = closure()
                        self.state["function_evaluations"] += 1

                        # Check the Armijo condition
                        if group["line_search_fn"] == "armijo":
                            found, step_size = ut.check_armijo_condition(loss_new=new_loss.item(), 
                                                                         loss_old=loss.item(), 
                                                                         step_size=step_size, 
                                                                         grad_norm_sq=grad_norm_sq, 
                                                                         c=group["c"], 
                                                                         beta_b=group["beta_b"])
                            if found == 1:
                                break
                        
                        # Check the Goldstein condition
                        elif group["line_search_fn"] == "goldstein":
                            found, step_size = ut.check_goldstein_condition(loss_new=new_loss.item(), 
                                                                           loss_old=loss.item(), 
                                                                           max_step_size=group["max_step_size"], 
                                                                           step_size=step_size, 
                                                                           grad_norm_sq=grad_norm_sq, 
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
        return loss, step_size