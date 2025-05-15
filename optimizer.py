import torch
from torch.optim import Optimizer, SGD
import random 

class LineSearchOptimizer(Optimizer):
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

    @torch.no_grad()
    def step(self, model, loss, batch_inputs, batch_labels):
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
            new_loss = self.loss_fn(model(batch_inputs), batch_labels).item()
            
            if self.is_successful_step(new_loss, loss, grad_norm_sq):
                if self.is_reliable_step(grad_norm_sq):
                    self.delta_sq = self.gamma * self.delta_sq
                else:
                    self.delta_sq = 1 / self.gamma * self.delta_sq
                
                self.alpha = min(self.alpha_max, self.gamma * self.alpha)
                
                break
            else:
                set_params(orig_params)

                self.alpha = self.alpha / self.gamma
                self.delta_sq = self.delta_sq / self.gamma
