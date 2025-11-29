import numpy as np
from fractional_helper import FractionalGradientHelper
from armijo import ArmijoLineSearch

class Optimizer:
    def __init__(
        self,
        optimizer='SGD',
        lr= 0.001,
        beta= 0.9,
        beta2= 0.99,
        use_fractional= False,
        alpha= 0.5,
        memory= 50,
        lam= 1e-3,
        line_search_version = None,
        epsilon= 1e-8
    ):
        self.optimizer = optimizer.lower()
        self.lr = lr
        self.beta = beta
        self.beta2 = beta2
        self.lam = lam
        self.epsilon = epsilon

        self.frac = FractionalGradientHelper(alpha, memory) if use_fractional else None
        self.line_search = ArmijoLineSearch(line_search_version) if line_search_version else None

        self.vw = None
        self.vb = 0.0
        self.mw = None
        self.mb = 0.0

        self.v_hat_w = None
        self.v_hat_b = None

    def init_state(self, w, b):
        self.vw = np.zeros_like(w)
        self.vb = 0.0
        self.mw = np.zeros_like(w)
        self.mb = 0.0
        self.v_hat_w = np.zeros_like(w)
        self.v_hat_b = 0.0

    def _maybe_line_search(self, objective, X, y, w, b, grad_w, grad_b, dir_w, dir_b):
        if not self.line_search:
            return self.lr
        return self.line_search.search(objective, X, y, w, b, grad_w, grad_b, dir_w, dir_b, eta0=self.lr)

    def update(self, objective, X, y, w, b, grad_w, grad_b, t):
        if self.frac:
            grad_w, grad_b = self.frac.apply(grad_w, grad_b)

        lr_t = self.lr

        # ----- Nesterov -----
        if self.optimizer.lower() == 'nesterov':
            w_look = w - self.beta * self.vw
            b_look = b - self.beta * self.vb
            _, (g_w_look, g_b_look) = objective(X, y, w_look, b_look)

            if self.frac:
                frac_g_w, frac_g_b = self.frac.apply(g_w_look, g_b_look)
            else:
                frac_g_w, frac_g_b = g_w_look, g_b_look

            # line search
            if self.line_search:
                lr_t = self._maybe_line_search(objective, X, y, w, b, g_w_look, g_b_look, -frac_g_w, -frac_g_b)

            self.vw = self.beta * self.vw + lr_t * frac_g_w
            self.vb = self.beta * self.vb + lr_t * frac_g_b

            w = w - self.vw
            b = b - self.vb

            return w, b

        # ----- Momentum -----
        if self.optimizer.lower() == 'momentum':
            descent_w = grad_w
            descent_b = grad_b

            if self.line_search:
                lr_t = self._maybe_line_search(objective, X, y, w, b, grad_w, grad_b, -descent_w, -descent_b)

            self.vw = self.beta * self.vw + lr_t * descent_w
            self.vb = self.beta * self.vb + lr_t * descent_b

            w = w - self.vw
            b = b - self.vb

            return w, b

        # ----- LION -----
        if self.optimizer.lower() == 'lion':
            g_w = grad_w
            g_b = grad_b

            lr_t = self.lr

            cw = self.beta * self.vw + (1 - self.beta) * g_w
            cb = self.beta * self.vb + (1 - self.beta) * g_b

            w = w - lr_t * (np.sign(cw) + self.lam * w)
            b = b - lr_t * (np.sign(cb) + self.lam * b)

            self.vw = self.beta2 * self.vw + (1 - self.beta2) * g_w
            self.vb = self.beta2 * self.vb + (1 - self.beta2) * g_b

            return w, b

        # ----- Adagrad -----
        if self.optimizer.lower() == 'adagrad':
            self.vw = self.vw + grad_w**2
            self.vb = self.vb + grad_b**2
            denom_w = np.sqrt(self.vw) + self.epsilon
            denom_b = np.sqrt(self.vb) + self.epsilon
            w = w - (self.lr / denom_w) * grad_w
            b = b - (self.lr / denom_b) * grad_b
            return w, b

        # ----- RMSprop -----
        if self.optimizer.lower() == 'rmsprop':
            self.vw = self.beta * self.vw + (1 - self.beta) * (grad_w**2)
            self.vb = self.beta * self.vb + (1 - self.beta) * (grad_b**2)
            denom_w = np.sqrt(self.vw) + self.epsilon
            denom_b = np.sqrt(self.vb) + self.epsilon
            w = w - (self.lr / denom_w) * grad_w
            b = b - (self.lr / denom_b) * grad_b
            return w, b

        # ----- Adam -----
        if self.optimizer.lower() == 'adam':
            self.mw = self.beta * self.mw + (1 - self.beta) * grad_w
            self.mb = self.beta * self.mb + (1 - self.beta) * grad_b

            self.vw = self.beta2 * self.vw + (1 - self.beta2) * (grad_w ** 2)
            self.vb = self.beta2 * self.vb + (1 - self.beta2) * (grad_b ** 2)

            beta_t = self.beta ** t
            beta2_t = self.beta2 ** t

            mw_prime = self.mw / (1 - beta_t)
            mb_prime = self.mb / (1 - beta_t)

            vw_prime = self.vw / (1 - beta2_t)
            vb_prime = self.vb / (1 - beta2_t)

            denom_w = np.sqrt(vw_prime) + self.epsilon
            denom_b = np.sqrt(vb_prime) + self.epsilon

            w = w - self.lr * (mw_prime / denom_w)
            b = b - self.lr * (mb_prime / denom_b)

            return w, b

        # ----- Nadam -----
        if self.optimizer.lower() == 'nadam':
            self.mw = self.beta * self.mw + (1 - self.beta) * grad_w
            self.mb = self.beta * self.mb + (1 - self.beta) * grad_b

            self.vw = self.beta2 * self.vw + (1 - self.beta2) * (grad_w**2)
            self.vb = self.beta2 * self.vb + (1 - self.beta2) * (grad_b**2)

            beta_t = self.beta ** t
            vw_prime = self.vw / (1 - self.beta2)
            vb_prime = self.vb / (1 - self.beta2)

            m_hat_nesterov_w = ((self.beta * self.mw) / (1 - beta_t)) + (((1 - self.beta) * grad_w) / (1 - beta_t))
            m_hat_nesterov_b = ((self.beta * self.mb) / (1 - beta_t)) + (((1 - self.beta) * grad_b) / (1 - beta_t))

            denom_w = np.sqrt(vw_prime) + self.epsilon
            denom_b = np.sqrt(vb_prime) + self.epsilon

            w = w - self.lr * (m_hat_nesterov_w / denom_w)
            b = b - self.lr * (m_hat_nesterov_b / denom_b)

            return w, b

        # ----- Adadelta -----
        if self.optimizer.lower() == 'adadelta':
            self.vw = self.beta * self.vw + (1 - self.beta) * (grad_w**2)
            self.vb = self.beta * self.vb + (1 - self.beta) * (grad_b**2)

            dx_w = -(np.sqrt(self.mw + self.epsilon) / np.sqrt(self.vw + self.epsilon)) * grad_w
            dx_b = -(np.sqrt(self.mb + self.epsilon) / np.sqrt(self.vb + self.epsilon)) * grad_b

            self.mw = self.beta * self.mw + (1 - self.beta) * (dx_w**2)
            self.mb = self.beta * self.mb + (1 - self.beta) * (dx_b**2)

            w = w + dx_w
            b = b + dx_b

            return w, b

        # ----- AdamW -----
        if self.optimizer.lower() == 'adamw':
            self.mw = self.beta * self.mw + (1 - self.beta) * grad_w
            self.mb = self.beta * self.mb + (1 - self.beta) * grad_b

            self.vw = self.beta2 * self.vw + (1 - self.beta2) * (grad_w**2)
            self.vb = self.beta2 * self.vb + (1 - self.beta2) * (grad_b**2)

            beta_t = self.beta ** t
            beta2_t = self.beta2 ** t

            m_hat_w = self.mw / (1 - beta_t)
            m_hat_b = self.mb / (1 - beta_t)
            v_hat_w = self.vw / (1 - beta2_t)
            v_hat_b = self.vb / (1 - beta2_t)

            # weight decay step
            w = w - self.lr * self.lam * w
            w = w - self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            b = b - self.lr * self.lam * b
            b = b - self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

            return w, b

        # ----- AdaBelief -----
        if self.optimizer.lower() == 'adambelief':
            self.mw = self.beta * self.mw + (1 - self.beta) * grad_w
            self.mb = self.beta * self.mb + (1 - self.beta) * grad_b

            self.vw = self.beta2 * self.vw + (1 - self.beta2) * (grad_w - self.mw)**2
            self.vb = self.beta2 * self.vb + (1 - self.beta2) * (grad_b - self.mb)**2

            beta_t = self.beta ** t
            beta2_t = self.beta2 ** t
            m_hat_w = self.mw / (1 - beta_t)
            m_hat_b = self.mb / (1 - beta_t)

            v_hat_w = self.vw / (1 - beta2_t)
            v_hat_b = self.vb / (1 - beta2_t)

            denom_w = np.sqrt(v_hat_w) + self.epsilon
            denom_b = np.sqrt(v_hat_b) + self.epsilon

            w = w - self.lr * (m_hat_w / denom_w)
            b = b - self.lr * (m_hat_b / denom_b)

            return w, b

        if self.line_search:
            lr_t = self._maybe_line_search(objective, X, y, w, b, grad_w, grad_b, -grad_w, -grad_b)
        else:
            lr_t = self.lr

        w = w - lr_t * grad_w
        b = b - lr_t * grad_b

        return w, b