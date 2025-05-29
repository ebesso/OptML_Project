import torch
import numpy as np

def reset_step(step_size, max_step_size=None, gamma=None, reset_option=2, n_batches_per_epoch=None, beta_f=None):
    """
    Reset the step size based on the specified reset option.

    Args:
        step_size (float): The current step size.
        max_step_size (float, optional): The max step size. Defaults to None.
        gamma (float, optional): The decay factor for the step size. Defaults to None.
        reset_option (int, optional): The reset option. Defaults to 1.
        n_batches_per_epoch (int, optional): The number of batches per epoch. Defaults to None.
        beta_f (float, optional): The increase factor for the step size. Defaults to None.

    Returns:
        float: The reseted step size.
    """
    if reset_option == 0:
        return max_step_size
    elif reset_option == 1:
        if max_step_size is None:
            raise ValueError("max_step_size must be provided for reset option 1")
        return max_step_size
    elif reset_option == 2:
        if n_batches_per_epoch is None:
            raise ValueError("n_batches_per_epoch must be provided for reset option 2")
        if gamma is None:
            raise ValueError("gamma must be provided for reset option 2")
        return step_size * gamma **(1 / n_batches_per_epoch)
    elif reset_option == 3:
        if beta_f is None:
            raise ValueError("beta_f must be provided for reset option 3")
        return step_size * beta_f
    else:
        raise ValueError("Invalid reset option")
    
def check_armijo_condition(f_new, f0, f0_prime, step_size, c, beta_b):
    """
    Check the Armijo condition for a given step size.

    Args:
        f_new (float): The new loss value.
        f0 (float): The old loss value.
        f0_prime (float): The derivative at the old loss value.
        step_size (float): The step size.
        c (float): The Armijo condition constant.
        beta_b (float): The decay factor for the step size.

    Returns:
        found (int): 1 if the condition is satisfied, 0 otherwise.
        step_size (float): The updated step size.
    """

    found = 0
    if f_new <= f0 + c * step_size * f0_prime:
        found = 1
    else:
        step_size *= beta_b

    return found, step_size

def check_goldstein_condition(f_new, f0, f0_prime, max_step_size, step_size, c, beta_b, beta_f):
    """
    Check the Goldstein condition for a given step size.

    Args:
        f_new (float): The new loss value.
        f0 (float): The old loss value.
        f0_prime (float): The derivative at the old loss value.
        max_step_size (float): The max step size.
        step_size (float): The step size.
        c (float): The Goldstein condition constant.
        beta_b (float): The decay factor for the step size.
        beta_f (float): The increase factor for the step size.

    Returns:
        found (int): 1 if the condition is satisfied, 0 otherwise.
        step_size (float): The updated step size.
    """

    found = 0
    if f_new > f0 + c * step_size * f0_prime:
       step_size *= beta_b
    elif f_new < f0 + (1 - c) * step_size * f0_prime:
        step_size = min(beta_f*step_size, max_step_size)
    else:
        found = 1

    return found, step_size

def copy_parameters(params):
    """
    Copy parameters from a list of tensors to a new list of tensors.

    Args:
        params (list): List of tensors to copy.

    Returns:
        list: A new list of copied tensors.
    """
    return [p.detach().clone() for p in params]

def set_params(target_params, update_params):
    """
    Set the parameters of a list of tensors to new values.

    Args:
        target_params (list): List of tensors to update.
        update_params (list): List of new values for the tensors.
    """
    with torch.no_grad():
        for p, new_p in zip(target_params, update_params):
            p.copy_(new_p)

def get_gradient(params):
    """
    Get the gradients of a list of tensors.

    Args:
        params (list): List of tensors to get gradients from.

    Returns:
        list: A new list of gradients.
    """
    return [p.grad.detach().clone() for p in params]

def get_grad_norm_sq(grad):
    """
    Get the squared norm of a list of gradients.

    Args:
        grad (list): List of gradients to get the norm from.

    Returns:
        float: The norm of the gradients.
    """
    grad_flat = torch.cat([g.view(-1) for g in grad])
    return grad_flat.norm()**2

def get_random_direction(params):
    direction = []
    for p in params:
        rand = torch.randn_like(p) 
        direction.append(rand)

    norm = torch.sqrt(sum((d**2).sum() for d in direction))
    return [d / norm for d in direction]

def is_concave(polynomial):
    return polynomial[0] < 0

def get_polynomial(point1, point2, point3):
    x0, y0 = point1
    x1, y1 = point2
    x2, y2 = point3

    return np.polyfit([x0, x1, x2], [y0, y1, y2], deg=2)


# Find minima on the interval [-max_step_size, max_step_size]
def get_minima(poly, max_step_size):

    critical_point = - poly[1] / (2 * poly[0])

    evaluate_polynomal = lambda x: sum([p * x ** (len(poly) - i - 1) for i, p in enumerate(poly)])

    if(abs(critical_point) < max_step_size):
        if evaluate_polynomal(-max_step_size) < min(evaluate_polynomal(critical_point), evaluate_polynomal(max_step_size)):
            return -max_step_size
        
        if evaluate_polynomal(max_step_size) < min(evaluate_polynomal(critical_point), evaluate_polynomal(-max_step_size)):
            return max_step_size
        
        return critical_point

    if evaluate_polynomal(-max_step_size) < evaluate_polynomal(max_step_size):
        return -max_step_size

    return max_step_size


