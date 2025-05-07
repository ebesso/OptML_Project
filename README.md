# Project

## Title
Evaluate effect of different line search methods

## Experiments
Evaluate performance on MNIST hand written dataset. 

All experiments use SGD.

Models: Big model and small model.
Loss function: Cross-entropy loss.

### Cases
Different setups to evaluate:
- Baseline 1 (Fixed learning rate).
- Baseline 2 (Step decay).
- Backtracking line search.
- Stochastic line search.  (?)
- Wolfe line search.
- Approximate line search.
- Learning rate adaptation via line search at initialization only.

### Metrics
- Convergence rate.
- Step sizes.
- Average execution time per step.
- Generalization.
- Average function evaluations per step.







