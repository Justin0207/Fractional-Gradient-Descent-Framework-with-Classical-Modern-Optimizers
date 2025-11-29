import numpy as np

class FractionalGradientHelper:
    def __init__(self, alpha=0.5, memory=50):
        self.alpha = alpha
        self.memory = memory
        self.weights = self._compute_gl_weights(alpha, memory)
        self.hist_w = []
        self.hist_b = []

    def _compute_gl_weights(self, alpha, N):
        w = np.zeros(N)
        w[0] = 1.0
        for k in range(1, N):
            w[k] = -w[k-1] * (1 - alpha - (k - 1)) / k
        return w

    def apply(self, grad_w, grad_b):
        M = len(self.weights)
        self.hist_w.insert(0, grad_w.copy())
        self.hist_b.insert(0, float(grad_b))

        if len(self.hist_w) > M:
            self.hist_w.pop()
        if len(self.hist_b) > M:
            self.hist_b.pop()

        frac_w = sum(self.weights[j] * self.hist_w[j] for j in range(len(self.hist_w)))
        frac_b = sum(self.weights[j] * self.hist_b[j] for j in range(len(self.hist_b)))

        return frac_w, frac_b

    def reset(self):
        self.hist_w = []
        self.hist_b = []
