import numpy as np

class ArmijoLineSearch:
    def __init__(self, version="V1"):
        self.version = version

    def search(self, objective, X, y, w, b, grad_w, grad_b, direction_w, direction_b, eta0=1.0, **kwargs):
        if self.version == "V1":
            return self._V1(objective, X, y, w, b, grad_w, grad_b, direction_w, direction_b, eta0, **kwargs)
        elif self.version == "V2":
            return self._V2(objective, X, y, w, b, grad_w, grad_b, direction_w, direction_b, eta0, **kwargs)
        elif self.version == "V3":
            return self._V3(objective, X, y, w, b, grad_w, grad_b, direction_w, direction_b, eta0, **kwargs)
        else:
            return eta0

    # --- Variant 1 ---
    def _V1(self, objective, X, y, w, b, grad_w, grad_b, direction_w, direction_b, eta0=1.0, gamma=0.5, c=1e-4):
        eta = eta0
        L_current, _ = objective(X, y, w, b)
        inner_w = np.sum(grad_w * direction_w)
        inner_b = np.sum(grad_b * direction_b)
        armijo_term = c * (inner_w + inner_b)

        while True:
            w_new = w + eta * direction_w
            b_new = b + eta * direction_b
            L_new, _ = objective(X, y, w_new, b_new)
            if L_new <= L_current + eta * armijo_term:
                break
            eta *= gamma
            if eta < 1e-12:
                break
        return eta

    # --- Variant 2 ---
    def _V2(self, objective, X, y, w, b, grad_w, grad_b, direction_w, direction_b, eta0=1.0, c1=1e-4, max_iter=1000):
        eta = eta0
        L_current, _ = objective(X, y, w, b)
        direction_term = np.sum(grad_w * direction_w) + np.sum(grad_b * direction_b)

        if direction_term >= 0:
            print("Warning: Not a descent direction.")
            return 0.0

        for _ in range(max_iter):
            w_new = w + eta * direction_w
            b_new = b + eta * direction_b
            L_new, _ = objective(X, y, w_new, b_new)
            if L_new <= L_current + c1 * eta * direction_term:
                return eta
            denominator = 2 * (L_current + eta * direction_term - L_new)
            eta = eta * 0.5 if denominator <= 0 else min(eta, (eta**2 * direction_term)/denominator)
        print(f"Warning: Line search failed within {max_iter} iterations.")
        return eta

    # --- Variant 3 ---
    def _V3(self, objective, X, y, w, b, grad_w, grad_b, direction_w, direction_b, eta0=1.0, c1=1e-4, omega=0.5, rho=0.5, history=None):
        gTd = np.sum(grad_w * direction_w) + np.sum(grad_b * direction_b)
        if gTd >= 0:
            return 1e-8

        L_current, _ = objective(X, y, w, b)
        if history is None or len(history) == 0:
            Rk = L_current
        else:
            max_hist = np.max(history)
            grad_norm = np.sqrt(np.sum(grad_w**2) + np.sum(grad_b**2))
            gamma = 1 - omega * np.exp(-grad_norm)
            Rk = gamma * L_current + (1 - gamma) * max_hist

        eta = eta0
        while True:
            w_new = w + eta * direction_w
            b_new = b + eta * direction_b
            L_new, _ = objective(X, y, w_new, b_new)
            if L_new <= Rk + eta * c1 * gTd:
                return eta
            eta *= rho
            if eta < 1e-12:
                return eta
