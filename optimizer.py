import torch
from torch.optim import Optimizer 

class LineSearchOptimizer(Optimizer):
    def __init__(self, params, loss_fn, init_step_size=1, max_iterations=100, c=0.1, beta=0.9, gamma=2):
        self.loss_fn = loss_fn
        self.init_step_size = init_step_size
        self.step_size= init_step_size
        self.max_iterations = max_iterations
        self.c = c
        self.beta = beta
        self.gamma = gamma


        super().__init__(params, dict())

    def is_successful_step(self, new_loss, loss, grad_norm_sq):
        return new_loss <= loss - self.step_size * self.c * grad_norm_sq
    
    def reset_step_size(self):
        self.step_size = self.init_step_size

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

        self.reset_step_size()

        for _ in range(self.max_iterations):
            new_params = [p - self.step_size * g for p, g in zip(orig_params, grad)]
            set_params(new_params)

            with torch.no_grad():
                new_loss = self.loss_fn(model(batch_inputs), batch_labels).item()
            
            if self.is_successful_step(new_loss, loss.item(), grad_norm_sq):
                break
            else:
                self.step_size *= self.beta

        return loss