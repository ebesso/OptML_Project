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
    
def check_armijo_condition(loss_new, loss_old, step_size, grad_norm_sq, c, beta_b):
    """
    Check the Armijo condition for a given step size.

    Args:
        loss_new (float): The new loss value.
        loss_old (float): The old loss value.
        step_size (float): The step size.
        grad_norm_sq (float): The squared norm of the gradient.
        c (float): The Armijo condition constant.
        beta_b (float): The decay factor for the step size.

    Returns:
        found (int): 1 if the condition is satisfied, 0 otherwise.
        step_size (float): The updated step size.
    """

    found = 0
    if loss_new <= loss_old - c * step_size * grad_norm_sq:
        found = 1
    else:
        step_size *= beta_b

    return found, step_size

def check_goldstein_condition(loss_new, loss_old, max_step_size, step_size, grad_norm_sq, c, beta_b, beta_f):
    """
    Check the Goldstein condition for a given step size.

    Args:
        loss_new (float): The new loss value.
        loss_old (float): The old loss value.
        max_step_size (float): The max step size.
        step_size (float): The step size.
        grad_norm_sq (float): The squared norm of the gradient.
        c (float): The Goldstein condition constant.
        beta_b (float): The decay factor for the step size.
        beta_f (float): The increase factor for the step size.

    Returns:
        found (int): 1 if the condition is satisfied, 0 otherwise.
        step_size (float): The updated step size.
    """

    found = 0
    if loss_new > loss_old - c * step_size * grad_norm_sq:
       step_size *= beta_b
    elif loss_new < loss_old - (1 - c) * step_size * grad_norm_sq:
        step_size = min(beta_f*step_size, max_step_size)
    else:
        found = 1

    return found, step_size


