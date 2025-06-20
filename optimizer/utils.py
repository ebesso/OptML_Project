import torch
import numpy as np

def reset_step(step_size, max_step_size=None, gamma=None, reset_option=2, n_batches_per_epoch=None, initial_step_size=1):
    """
    Reset the step size based on the specified reset option.

    Args:
        step_size (float): The current step size.
        max_step_size (float, optional): The max step size. Defaults to None.
        gamma (float, optional): The increase factor for the step size. Defaults to None.
        reset_option (int, optional): The reset option. Defaults to 1.
        n_batches_per_epoch (int, optional): The number of batches per epoch. Defaults to None.
        initial_step_size (float): The initial step size

    Returns:
        float: The reseted step size.
    """
    if reset_option == 0:
        return step_size
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
        if gamma is None:
            raise ValueError("gamma must be provided for reset option 3")
        return step_size * gamma     
    elif reset_option == 4:
        return initial_step_size
    else:
        raise ValueError("Invalid reset option")

def armijo_line_search(line_fn, orig_loss, orig_loss_prime, step_size, group):
    '''
    Perform an Armijo line search to find a suitable step size.
    Args:
        line_fn (callable): The function to evaluate.
        orig_loss (float): The original loss value.
        orig_loss_prime (float): The derivative of the loss at the original point.
        step_size (float): The initial step size.
        group (dict): A dictionary containing parameters for the line search.
        Returns:
        tuple: A tuple containing a boolean indicating success, the step size, number of function evaluations and number of gradient evalutations.
    '''
    func_evals = 0
    grad_evals = 0
    
    for _ in range(group['max_iterations']):
        new_loss = line_fn(step_size)
        func_evals += 1

        if new_loss <= orig_loss + group["c1"]* step_size * orig_loss_prime:
            return True, step_size, func_evals, grad_evals
        else:
            step_size *= group["beta_b"]

    return False, step_size, func_evals, grad_evals

def goldstein_line_search(line_fn, orig_loss, orig_loss_prime, step_size, group):
    '''
    Perform a Goldstein line search to find a suitable step size.
    Args:
        line_fn (callable): The function to evaluate.
        orig_loss (float): The original loss value.
        orig_loss_prime (float): The derivative of the loss at the original point.
        step_size (float): The initial step size.
        group (dict): A dictionary containing parameters for the line search.
    Returns:
        tuple: A tuple containing a boolean indicating success, the step size, number of function evaluations and number of gradient evalutations.
    '''
    
    func_evals = 0
    grad_evals = 0
    
    for _ in range(group['max_iterations']):
        new_loss = line_fn(step_size)
        func_evals += 1

        if new_loss > orig_loss + group["c1"] * step_size * orig_loss_prime:
            step_size *= group["beta_b"]
        elif new_loss < orig_loss + (1 - group["c1"]) * step_size * orig_loss_prime:
            step_size = min(group["beta_f"] * step_size, group["max_step_size"])
        else:
            return True, step_size, func_evals, grad_evals

    return False, step_size, func_evals, grad_evals

def strong_wolfe_line_search(line_fn, params, orig_loss, orig_loss_prime, step_size, direction, group):
    """
    Perform a Strong Wolfe line search to find a suitable step size.

    Args:
        line_fn (callable): The function to evaluate.
        params (list): List of tensors representing the model parameters.
        orig_loss (float): The original loss value.
        orig_loss_prime (float): The derivative of the loss at the original point.
        step_size (float): The initial step size.
        direction (list): List of tensors representing the direction to update.
        group (dict): A dictionary containing parameters for the line search.

    Returns:
        tuple: A tuple containing a boolean indicating success, the step size, number of function evaluations and number of gradient evalutations.
    """
    ## https://link.springer.com/content/pdf/10.1007/978-0-387-40065-5_3.pdf
    # function and gradient evalutations
    func_evals = 0
    grad_evals = 0

    # definie function which outputs directional derivative
    def line_fn_deriv(loss):
        grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=True, allow_unused=True)
        return sum(
            torch.dot(d.view(-1), g.view(-1))
            for d, g in zip(direction, grads)
            if d is not None and g is not None
        )

    alpha = max(min(step_size, group["max_step_size"]), 0.1)
    alpha_prev = 0
    prev_loss = orig_loss

    for j in range(group["max_iterations"]):
        new_loss = line_fn(alpha)
        func_evals += 1

        if (new_loss > orig_loss + group["c1"]*alpha*orig_loss_prime or 
            (j > 0 and new_loss > prev_loss)):
            zoom_results = zoom(line_fn=line_fn, 
                                params=params, 
                                alpha_lo=alpha_prev, 
                                loss_lo=prev_loss,
                                alpha_hi=alpha, 
                                loss_hi=new_loss, 
                                orig_loss=orig_loss, 
                                orig_loss_prime=orig_loss_prime, 
                                direction=direction, 
                                group=group)
            found, step_size, extra_func_evals, extra_grad_evals = zoom_results
            func_evals += extra_func_evals
            grad_evals += extra_grad_evals
            return found, step_size, func_evals, grad_evals

        new_loss_prime = line_fn_deriv(new_loss)
        grad_evals += 1

        if abs(new_loss_prime) <= group["c2"]*abs(orig_loss_prime):
            return True, alpha, func_evals, grad_evals
        
        if new_loss_prime >= 0:
            zoom_results = zoom(line_fn=line_fn, 
                                params=params, 
                                alpha_lo=alpha, 
                                loss_lo=new_loss,
                                alpha_hi=alpha_prev, 
                                loss_hi=prev_loss, 
                                orig_loss=orig_loss, 
                                orig_loss_prime=orig_loss_prime, 
                                direction=direction, 
                                group=group)
            found, step_size, extra_func_evals, extra_grad_evals = zoom_results
            func_evals += extra_func_evals
            grad_evals += extra_grad_evals
            return found, step_size, func_evals, grad_evals

        alpha_prev = alpha
        prev_loss = new_loss

        alpha = min(alpha * group["beta_f"], group["max_step_size"])
    
    return False, alpha, func_evals, grad_evals

def zoom(line_fn, params, alpha_lo, loss_lo, alpha_hi, loss_hi, orig_loss, orig_loss_prime, direction, group):
    # function and gradient evalutations    
    func_evals = 0
    grad_evals = 0

    # definie function which outputs directional derivative
    def line_fn_deriv(loss):
        grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=True, allow_unused=True)
        return sum(
            torch.dot(d.view(-1), g.view(-1))
            for d, g in zip(direction, grads)
            if d is not None and g is not None
        )
    
    for _ in range(50):
        # Interpolate (using quadratic, cubic, or bisection) to find a trial step length alpha between alpha_lo and alpha_hi;
        #bisection
        alpha = (alpha_lo + alpha_hi) / 2

        new_loss = line_fn(alpha)
        func_evals += 1

        if new_loss > orig_loss + group["c1"]*alpha*orig_loss_prime or new_loss >= loss_lo:
            alpha_hi = alpha
            loss_hi = new_loss
        else: 
            # evalute directional derivative
            new_loss_prime = line_fn_deriv(new_loss)
            grad_evals += 1

            if abs(new_loss_prime) <= group["c2"]*abs(orig_loss_prime):
                return True, alpha, func_evals, grad_evals
            
            if new_loss_prime*(alpha_hi-alpha_lo) >= 0:
                alpha_hi = alpha_lo
                loss_hi = loss_lo

            alpha_lo = alpha
            loss_lo = new_loss
    
    return False, alpha, func_evals, grad_evals

def copy_parameters(params):
    """
    Copy parameters from a list of tensors to a new list of tensors.

    Args:
        params (list): List of tensors to copy.

    Returns:
        list: A new list of copied tensors.
    """
    return [p.detach().clone() for p in params]

def update_parameters(params, step_size, original_params, direction):
    # Ugly hack to avoid changing the original function signature
    update_parameters_no_grad(params, step_size, original_params, direction)

def update_parameters_no_grad(params, step_size, original_params, direction):
    """
    Update parameters by adding a scaled direction to the original parameters.
    
    Args:
        params (list): List of tensors to update.
        step_size (float): The step size to scale the direction.
        original_params (list): List of original tensors to use as base.
        direction (list): List of tensors representing the direction to update. If an entry in `direction` is `None`, the corresponding parameter will not be updated.
    """
    with torch.no_grad():
        for p_next, p_orig, d in zip(params, original_params, direction):
            if d is not None:
                p_next.data = p_orig + step_size * d

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
        list: A new list of gradients. If a tensor does not have a gradient (`p.grad is None`), `None` will be returned for that tensor.
    """
    return [p.grad.detach().clone() if p.grad is not None else None for p in params]

def get_grad_norm(grad):
    """
    Get the squared norm of a list of gradients.

    Args:
        grad (list): List of gradients to get the norm from.

    Returns:
        float: The norm of the gradients.
    """
    return torch.norm(torch.cat([g.view(-1) for g in grad if g is not None]))

def get_random_direction(params):
    """ Generate a random direction for each parameter tensor from a standard Gaussian distribution.
    Args:
        params (list): List of tensors representing the model parameters.
    Returns:
        list: A list of tensors with the same shape as the parameters, filled with random values.
    """
    with torch.no_grad():
        direction = []
        for p in params:
            rand = torch.randn_like(p, device='cpu') 
            direction.append(rand)

        return direction
