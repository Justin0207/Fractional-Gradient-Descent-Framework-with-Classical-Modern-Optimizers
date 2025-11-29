import numpy as np

def logistic_loss(X, y, w, b, reg=0.0):
    m = X.shape[0]
    z = X @ w + b
    y_pred = 1 / (1 + np.exp(-z))
    eps = 1e-15
    L = -1/m * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
    L += 0.5 * reg * np.sum(w * w)
    grad_w = 1/m * (X.T @ (y_pred - y)) + reg * w
    grad_b = 1/m * np.sum(y_pred - y)
    return L, [grad_w, grad_b]

def softmax_loss(X, y, w, b, reg=0.0, epsilon=1e-15):
    m, n = X.shape
    K = w.shape[0]
    logits = X @ w.T + b
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    log_probs = -np.log(probs[np.arange(m), y] + epsilon)
    data_loss = np.mean(log_probs)
    reg_loss = 0.5 * reg * np.sum(w * w)
    loss = data_loss + reg_loss
    dscores = probs.copy()
    dscores[np.arange(m), y] -= 1
    dscores /= m
    grad_w = dscores.T @ X + reg * w
    grad_b = np.sum(dscores, axis=0)
    return loss, [grad_w, grad_b]

def mse_loss(X, y, w, b, reg=0.0):
    m = X.shape[0]
    y_pred = X @ w + b
    loss = (1/(2*m)) * np.sum((y_pred - y)**2) + 0.5 * reg * np.sum(w*w)
    grad_w = (1/m) * (X.T @ (y_pred - y)) + reg * w
    grad_b = (1/m) * np.sum(y_pred - y)
    return loss, [grad_w, grad_b]

def hinge_loss(X, y, w, b, reg=0.0):
    m = X.shape[0]
    y_transformed = np.where(y == 1, 1, -1)
    margins = 1 - y_transformed * (X @ w + b)
    loss = np.mean(np.maximum(0, margins)) + 0.5 * reg * np.sum(w * w)
    indicator = (margins > 0).astype(float)
    grad_w = -(1/m) * (X.T @ (indicator * y_transformed)) + reg * w
    grad_b = -(1/m) * np.sum(indicator * y_transformed)
    return loss, [grad_w, grad_b]
