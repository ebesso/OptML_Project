# Project

## Title
Learning Rate for Neural Networks via Line Search

## Project goal
Empirically investigate line search-based methods for selecting the learning rate during training of a machine learning model, and compare them to fixed and adaptive learning rate strategies (e.g., SGD, Adam).

## Questions
- Can line search improve convergence speed or generalization in neural networks?

- Is the learning rate found via line search more stable than manually tuned schedules?

- How does line search compare in computational cost and performance to adaptive optimizers (e.g., Adam)?

- What variants of line search (Armijo, Wolfe, quadratic fit, etc.) work best in practice?

### Cases
Different setups to evaluate (current):
- Fixed and adaptive learning rate strategies (e.g., SGD, Adam) to compare with
- Stochastic line search, direction with SGD, using Armijo, Goldstein (?and Strong Wolfe?) conditions

Different setups to evaluate (possible future):
- Add momentum to stochastic line search
- Gradient free optimization, random direction and zero-order line search

Ways to get search direction: 
- SGD
- Along momentum direction (as in SGD with momentum)
- Adam's update direction
- Random

Line seach methods:
- Zero-order: eg, golden section, uniform
- Armijo, Goldstein and Strong Wolfe conditions
- quadratic fit


### Metrics
- Convergence rate.
- Step sizes.
- Average execution time per step.
- Generalization to bigger models
- Average function evaluations per step.
- Training loss vs. iterations and wall-clock time
- Test accuracy
- Stability (does it oscillate or explode?)






