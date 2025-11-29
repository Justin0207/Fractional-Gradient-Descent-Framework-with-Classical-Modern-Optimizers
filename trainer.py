import numpy as np
from typing import List

class GradientDescentTrainer:
    def __init__(self, optimizer, objective_function, tol=1e-9, print_every=100):
        self.optimizer = optimizer
        self.objective_function = objective_function
        self.tol = tol
        self.print_every = print_every

    def train(self, w_init, b_init, X, y, epochs=1000):
        w = w_init.copy()
        b = float(b_init)
        losses= []

        self.optimizer.init_state(w, b)

        for t in range(1, epochs+1):
            L, grads = self.objective_function(X, y, w, b)
            grad_w, grad_b = grads
            old_w, old_b = w.copy(), b

            w, b = self.optimizer.update(self.objective_function, X, y, w, b, grad_w, grad_b, t)
            losses.append(L)

            grad_norm = max(np.max(np.abs(w-old_w)), np.abs(b-old_b))
            if grad_norm < self.tol:
                print(f"Converged at epoch {t} | ||Δ|| = {grad_norm:.6e}")
                break

            if t % self.print_every == 0:
                print(f"Epoch {t} | Loss={L:.6f} | ||Δ||={grad_norm:.6e}")

        print(f"Final loss: {L:.6f}")
        return w, b, losses

    def predict(self, X, w, b, threshold=0.5, regression=False):
        z = X @ w + b
        y_pred = 1 / (1 + np.exp(-z))
        if regression:
            return z
        else:
            return (y_pred >= threshold).astype(int)
