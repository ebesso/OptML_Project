import torch
import numpy as np

def reset_step(step_size, init_step_size=None, gamma=None, reset_option=2, n_batches_per_epoch=None, beta=None):
    """
    Reset the step size based on the specified reset option.

    Args:
        step_size (float): The current step size.
        init_step_size (float, optional): The initial step size. Defaults to None.
        gamma (float, optional): The decay factor for the step size. Defaults to None.
        reset_option (int, optional): The reset option. Defaults to 1.
        n_batches_per_epoch (int, optional): The number of batches per epoch. Defaults to None.

    Returns:
        float: The reseted step size.
    """
    if reset_option == 0:
        return init_step_size
    elif reset_option == 1:
        if init_step_size is None:
            raise ValueError("init_step_size must be provided for reset option 1")
        return init_step_size
    elif reset_option == 2:
        if n_batches_per_epoch is None:
            raise ValueError("n_batches_per_epoch must be provided for reset option 2")
        if gamma is None:
            raise ValueError("gamma must be provided for reset option 2")
        return step_size * gamma **(1 / n_batches_per_epoch)
    elif reset_option == 3:
        if beta is None:
            raise ValueError("beta must be provided for reset option 3")
        return step_size / beta
    else:
        raise ValueError("Invalid reset option")
    
def check_armijo_condition(loss_new, loss_old, step_size, grad_norm_sq, c, beta):
    """
    Check the Armijo condition for a given step size.

    Args:
        loss_new (float): The new loss value.
        loss_old (float): The old loss value.
        step_size (float): The step size.
        grad_norm_sq (float): The squared norm of the gradient.
        c (float): The Armijo condition constant.
        beta (float): The decay factor for the step size.

    Returns:
        found (int): 1 if the condition is satisfied, 0 otherwise.
        step_size (float): The updated step size.
    """

    found = 0
    if loss_new <= loss_old - c * step_size * grad_norm_sq:
        found = 1
    else:
        step_size *= beta

    return found, step_size

def check_goldstein_condition(loss_new, loss_old, init_step_size, step_size, grad_norm_sq, c, beta, gamma):
    """
    Check the Goldstein condition for a given step size.

    Args:
        loss_new (float): The new loss value.
        loss_old (float): The old loss value.
        init_step_size (float): The initial step size.
        step_size (float): The step size.
        grad_norm_sq (float): The squared norm of the gradient.
        c (float): The Goldstein condition constant.
        beta (float): The decay factor for the step size.
        gamma (float): The increase factor for the step size.

    Returns:
        found (int): 1 if the condition is satisfied, 0 otherwise.
        step_size (float): The updated step size.
    """

    found = 0
    if loss_new > loss_old - c * step_size * grad_norm_sq:
       step_size *= beta
    elif loss_new < loss_old - (1 - c) * step_size * grad_norm_sq:
        step_size = min(gamma*step_size, init_step_size)
    else:
        found = 1

    return found, step_size


