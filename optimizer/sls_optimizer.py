import torch
from torch.optim import Optimizer 
from . import utils as ut

class SlsOptimizer(Optimizer):
    def __init__(self, params, loss_fn, gamma, theta, alpha_max, j_0, delta, max_iterations):
        self.gamma = gamma
        self.theta = theta
        self.alpha = gamma**j_0 * alpha_max
        self.loss_fn = loss_fn
        self.delta_sq = delta ** 2
        self.alpha_max = alpha_max
        self.max_iterations = max_iterations

        if alpha_max <= 0:
            raise ValueError("Alpha max must be positive")
        if j_0 > 0:
            raise ValueError("J_0 must be non-positive")
        if delta <= 0:
            raise ValueError("Delta_0 must be positive")
        if theta >= 1 or theta <= 0:
            raise ValueError("Theta must be in the range (0, 1)")
        if max_iterations <= 0:
            raise ValueError("Max iterations must be positive")

        super().__init__(params, dict())

    def is_successful_step(self, new_loss, loss, grad_norm_sq):
        return new_loss <= loss - self.alpha * self.theta * grad_norm_sq
    
    def is_reliable_step(self, grad_norm_sq):
        return self.alpha * grad_norm_sq >= self.delta_sq

    def step(self, model, batch_inputs, batch_labels):

        model.zero_grad()

        loss = self.loss_fn(model(batch_inputs), batch_labels)
        loss.backward()

        orig_params = [p.detach().clone() for p in model.parameters()]

        def set_params(new_params):
            with torch.no_grad():
                for p, new_p in zip(model.parameters(), new_params):
                    p.copy_(new_p)
                    
        grad = [p.grad.detach().clone() for p in model.parameters()]
        grad_flat = torch.cat([g.view(-1) for g in grad])
        grad_norm_sq = grad_flat.norm()**2

        for _ in range(self.max_iterations):
            new_params = [p - self.alpha * g for p, g in zip(orig_params, grad)]
            set_params(new_params)

            with torch.no_grad():
                new_loss = self.loss_fn(model(batch_inputs), batch_labels).item()
            
            if self.is_successful_step(new_loss, loss.item(), grad_norm_sq):
                if self.is_reliable_step(grad_norm_sq):
                    self.delta_sq = self.gamma * self.delta_sq
                else:
                    self.delta_sq = 1 / self.gamma * self.delta_sq
                
                self.alpha = min(self.alpha_max, self.gamma * self.alpha)
                break
            else:
                self.alpha = self.alpha / self.gamma
                self.delta_sq = self.delta_sq / self.gamma

        return loss
    
class SlsOptimizerTemp(Optimizer):
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
        ''' closure should be like this
        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            return loss'''
        
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

            orig_params = [p.detach().clone() for p in params]
            grad = [p.grad.detach().clone() for p in params]
            grad_flat = torch.cat([g.view(-1) for g in grad])
            grad_norm_sq = grad_flat.norm()**2

            def set_params(old_params, new_params):
                with torch.no_grad():
                    for p, new_p in zip(old_params, new_params):
                        p.copy_(new_p)
            
            step_size = self.state["step_size"]
            
            with torch.no_grad():
                if grad_norm_sq >= 1e-16:
                    # Reset the step size
                    step_size = ut.reset_step(step_size=self.state["step_size"],
                                              max_step_size=group["max_step_size"], 
                                              gamma=group["gamma"],
                                              reset_option=group["reset_option"], 
                                              n_batches_per_epoch=group["n_batches_per_epoch"],
                                              beta_f=group["beta_f"])
                    
                    found = 0

                    # Perform the line search
                    for _ in range(group['max_iterations']):
                        new_params = [p - step_size * g for p, g in zip(orig_params, grad)]
                        set_params(params, new_params)
                        
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
                        set_params(params, orig_params)
                        
            self.state["step_size"] = step_size

            
        # Returns the loss at the start
        return loss