# Fractional-Gradient-Descent-Framework-with-Classical-Modern-Optimizers
This repository provides a unified, extensible Python framework for experimenting with fractional-order gradient descent, modern optimization algorithms, and Armijo-based line search strategies. 

## Overview
This repository provides a unified, extensible Python framework for experimenting with fractional-order gradient descent, modern optimization algorithms, and Armijo backtracking line-search strategies.
It is designed for research in optimization theory, fractional calculus, numerical analysis, and machine learning.

The system includes:

- Fractional Gradient Descent using the Grünwald–Letnikov formulation

- Multiple optimizers including classical and modern adaptive methods

- Three Armijo Backtracking Line Search variants

- A unified Optimizer class with modular, extensible design

## Features
- Fractional Gradient (GL-based)

- Fast numerical Grünwald–Letnikov differentiation

- Adjustable fractional order α

- Gradient memory history

- Works on both weights and bias

- Armijo Line Search (3 Variants)
### Variant	Description
V1	Classical Armijo backtracking
--|-----------------------------------
V2	Adaptive Armijo with curvature-based step size correction
--|-----------------------------------------------------------
V3	Non-monotone Armijo (useful for noisy or rough losses)
--|--------------------------------------------------------
