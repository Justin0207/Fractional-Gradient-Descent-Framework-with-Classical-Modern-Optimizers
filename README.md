# Fractional Gradient Descent Framework with Classical and Modern Optimizers

This repository provides a unified and extensible Python framework for investigating fractional-order gradient descent, modern optimization algorithms, and Armijo-based line-search strategies.  
It is designed to support research in optimization theory, fractional calculus, numerical analysis, and machine learning.

---

## 📌 Overview

This framework integrates:

- Fractional-order gradient computation using the Grünwald–Letnikov formulation  
- A suite of classical and adaptive optimizers compatible with fractional gradients  
- Three variants of Armijo Backtracking Line Search  
- A modular Optimizer class enabling seamless experimentation

The system is structured to provide clarity, extensibility, and ease of integration with machine learning models and custom loss functions.

---

## 🚀 Features

### **Fractional Gradient Descent**
- Grünwald–Letnikov–based fractional differentiation  
- Adjustable fractional order $$\( \alpha \in (0,1] \)$$
- Efficient numerical implementation  
- Gradient history for memory-aware updates  
- Applicable to both weights and bias  

### **Armijo Line Search (3 Variants)**

| Variant | Description |
|--------|-------------|
| **V1** | Classical Armijo backtracking |
| **V2** | Adaptive Armijo with curvature-based step-size correction |
| **V3** | Non-monotone Armijo (based on [1]) |

### **Modern Optimization Algorithms**

All optimizers support fractional gradients and can be optionally combined with line search:

- SGD  
- Momentum  
- Nesterov Accelerated Gradient (NAG)  
- RMSProp  
- Adagrad  
- Adadelta  
- Adam  
- AdamW  
- AdaBelief  
- Lion  
- Nadam  

Each method supports additional features such as weight decay, bias correction, and stability enhancements.

---

## 📁 Repository Structure


---

## 🧪 Example Usage  
### Pima Indians Diabetes Classification

This repository includes a full example demonstrating:

- Logistic Regression  
- Lion optimizer  
- Fractional gradient with $$\( \alpha = 0.5 \)$$

| Model                             | Optimizer | Fractional α | Accuracy |
|----------------------------------|-----------|--------------|----------|
| **Proposed Framework**           | Lion      | 0.5          | 79.22%   |
| Scikit-Learn Logistic Regression | LBFGS     | —            | 75.32%   |

### **Key Findings**
- Fractional Lion improved accuracy by +3.9%.  
- Fractional gradients reduce oscillatory behavior in Lion's update rule.  
- Demonstrates the promising potential of fractional-order optimization in machine learning.  

---

## 📘 Mathematical Foundations

This section provides the theoretical background behind our Fractional Gradient Framework, Armijo Line Search Variants, and Optimizer Integrations.

---

### 1. Fractional Calculus (Grünwald–Letnikov Derivative)

Fractional calculus extends classical differentiation by permitting non-integer derivative orders, which introduces a memory effect: gradients incorporate information from previous states.

The Grünwald–Letnikov fractional derivative of order $$\( \alpha \)$$ is defined (in the continuous limit) as:

$$
D^\alpha f(x)
= \lim_{h \to 0} \frac{1}{h^\alpha}
\sum_{k=0}^{\left\lfloor \frac{x-a}{h} \right\rfloor}
(-1)^k \binom{\alpha}{k} \, f(x - k h)
$$

In discrete settings (e.g. \(h=1\) and finite memory \(N\)), this is approximated by:

$$
D^\alpha f(x_t) \approx
\sum_{k=0}^{N} w_k^{(\alpha)} \, f(x_{t-k}),
\qquad
w_k^{(\alpha)} = (-1)^k \binom{\alpha}{k}.
$$

Here $$\( \binom{\alpha}{k} = \dfrac{\Gamma(\alpha+1)}{\Gamma(k+1)\Gamma(\alpha-k+1)} \)$$.  
When $$\( \alpha = 1 \)$$ the expression reduces to the classical first-order derivative. Values $$\( \alpha < 1 \)$$ impart smoothing and memory, which can reduce oscillation and improve stability.

---

### 2. Armijo Backtracking Line Search

To select stable step sizes we enforce an Armijo-type condition. For a (fractional) gradient $$\(g\)$$ and step size $$\(\eta\)$$:

$$
f(x - \eta g) \le f(x) - c_1 \eta \lVert g \rVert^2,
\qquad c_1 \in (0,1).
$$

We implement three variants:

- **V1 — Classical Armijo Backtracking:** standard backtracking to satisfy the Armijo inequality.  
- **V2 — Adaptive Armijo (curvature correction):** adjust \(\eta\) using curvature estimates, e.g. \(\eta \leftarrow \rho(\kappa)\eta\), with \(\rho(\kappa)\in(0,1)\).  
- **V3 — Non-monotone Armijo:** allow temporary increases in loss using a windowed max:

$$
f(x - \eta g) \le \max_{k-M \le j \le k} f(x_j) - c_1 \eta \, g^\top d,
$$

where \(M\) is the memory window and \(d\) is the search direction.

---

### 3. Fractional Gradient Descent (FGD) and Integration with Optimizers

Given a loss $$\(L(\theta)\)$$, fractional gradient descent updates are:

$$
\theta_{t+1} = \theta_t - \eta_t \, D^\alpha_\theta L(\theta_t).
$$

When used inside modern optimizers, the fractional gradient $$\(g_t = D^\alpha_\theta L(\theta_t)\)$$ replaces the ordinary gradient. Example (Lion-style signed update):

$$
\theta_{t+1} = \theta_t - \eta_t \, \mathrm{sign}(g_t).
$$

More generally, each optimizer consumes $$\(D^\alpha_\theta L\)$$:

$$
\theta_{t+1} = \theta_t - \eta_t \cdot \mathrm{Optimizer}\!\big(D^\alpha_\theta L(\theta_t)\big).
$$

---

## 4. Why the Combined Framework Works

| Component | Contribution |
|----------|--------------|
| **Fractional Derivative** | Adds memory, smooths gradients, reduces noise |
| **Armijo Line Search** | Ensures stable and reliable step-size selection |
| **Adaptive Optimizers** | Accelerate convergence and normalize gradients |
| **Combined Framework** | Produces high accuracy, robustness, and stable learning |

---

## 5. Convergence Considerations

Under standard assumptions—including Lipschitz continuity and boundedness—fractional gradient descent exhibits:

- Convergence for all $$\( 0 < \alpha \le 1 \)$$
- Guaranteed descent under Armijo conditions  
- Enhanced stability for noisy or non-convex landscapes via V2 and V3 line-search variants  

These foundations align with classical results in convex optimization and recent findings in fractional-order systems.

---

## 📄 License

MIT License. See `LICENSE` for details.

---

### References

[1] Hafshejani, S. F., Gaur, D., Hossain, S., & Benkoczi, R. (2023).  
*Fast Armijo Line Search for Stochastic Gradient Descent.*  

[2] Chen, S., Zhang, C., & Mu, H. (2024). An adaptive learning rate deep learning optimizer using long and short-term gradients based on G–L fractional-order derivative. Neural Processing Letters, 56(2), 106.

[3] Zhou, X., Zhao, C., & Huang, Y. (2023). A deep learning optimizer based on Grünwald–Letnikov fractional order definition. Mathematics, 11(2), 316.

