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
- Adjustable fractional order \( \alpha \in (0,1] \)  
- Efficient numerical implementation  
- Gradient history for memory-aware updates  
- Applicable to both weights and bias  

### **Armijo Line Search (3 Variants)**

| Variant | Description |
|--------|-------------|
| **V1** | Classical Armijo backtracking |
| **V2** | Adaptive Armijo with curvature-based step-size correction |
| **V3** | Non-monotone Armijo (suitable for noisy or rugged loss landscapes) |

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
- Fractional gradient with \( \alpha = 0.5 \)

| Model                             | Optimizer | Fractional α | Accuracy |
|----------------------------------|-----------|--------------|----------|
| **Proposed Framework**           | Lion      | 0.5          | 79.22%   |
| Scikit-Learn Logistic Regression | LBFGS     | —            | 75.32%   |

### **Key Findings**
- Fractional Lion improved accuracy by **+3.9%**.  
- Fractional gradients reduce oscillatory behavior in Lion's update rule.  
- Demonstrates the promising potential of fractional-order optimization in machine learning.  

---

## 📘 Mathematical Foundations

This section outlines the theoretical underpinnings of the framework, including fractional calculus, fractional gradient descent, line-search theory, and their interaction with modern optimizers.

---

## 1. Fractional Calculus in Optimization

Fractional calculus extends classical differentiation by permitting non-integer derivative orders.  
This introduces a memory effect, enabling gradients to incorporate information from previous states.

### **Grünwald–Letnikov Fractional Derivative**

For a function \( f(x) \), the Grünwald–Letnikov derivative of order \( \alpha \) is approximated as:

\[
D^\alpha f(x) \approx 
\frac{1}{h^\alpha} \sum_{k=0}^{N} (-1)^k 
\binom{\alpha}{k} f(x - kh)
\]

Where:

- \( \alpha \in (0,1] \) is the fractional order  
- \( \binom{\alpha}{k} \) uses the Gamma function  
- \( N \) is the memory length  
- \( h \) is the step size  

**Special Cases**

- \( \alpha = 1 \): Recovers the classical first-order derivative  
- \( \alpha < 1 \): Produces smoother and more stable gradients, useful in noisy or high-curvature regions  

---

## 2. Fractional Gradient Descent (FGD)

Parameter updates follow:

\[
\theta_{t+1} = \theta_t - \eta \, D^\alpha_\theta L(\theta_t)
\]

Benefits:

- Enhanced stability on noisy loss surfaces  
- Reduced oscillations near optima  
- Improved generalization  
- Smoother convergence trajectories  

---

## 3. Armijo Backtracking Line Search Variants

Three variants of Armijo Backtracking are implemented to compute an adaptive step size \( \eta \).

---

### **V1 — Classical Armijo Backtracking**

Finds the largest \( \eta \) satisfying:

\[
f(x - \eta g) \le f(x) - c_1 \eta \lVert g \rVert^2
\]

Where \( g \) is the (fractional) gradient.

---

### **V2 — Adaptive Armijo with Curvature Correction**

Step size adapts according to curvature:

\[
\eta \leftarrow \rho(\kappa) \cdot \eta
\]

Where:

- \( \kappa \) is a curvature estimate  
- \( \rho(\kappa) \in (0,1) \) adjusts the aggressiveness of step-size reduction  

This greatly improves stability on steep, irregular, or non-smooth loss landscapes.

---

### **V3 — Non-Monotone Armijo Search**

Allows slight increases in loss:

\[
f(x - \eta g) \le 
\max_{k-M \le j \le k} f(x_j) 
- c_1 \eta g^T d
\]

Where:

- \( M \) is the memory window  
- \( d \) is the search direction  

Useful for:

- Noisy objectives  
- Loss surfaces with many small local minima  
- Fractional gradients that produce smooth but non-monotonic paths  

---

## 4. Integrating Fractional Gradients with Modern Optimizers

The framework supports many state-of-the-art optimizers.  
Each update takes the form:

\[
\theta_{t+1} = \theta_t - \eta \cdot 
\text{Optimizer}(D^\alpha_\theta L)
\]

Incorporating fractional gradients allows optimizers to leverage historical curvature information and long-term memory, which enhances convergence behavior and stability.

---

## 5. Why the Combined Framework Works

| Component | Contribution |
|----------|--------------|
| **Fractional Derivative** | Adds memory, smooths gradients, reduces noise |
| **Armijo Line Search** | Ensures stable and reliable step-size selection |
| **Adaptive Optimizers** | Accelerate convergence and normalize gradients |
| **Combined Framework** | Produces high accuracy, robustness, and stable learning |

---

## 6. Convergence Considerations

Under standard assumptions—including Lipschitz continuity and boundedness—fractional gradient descent exhibits:

- Convergence for all \( 0 < \alpha \le 1 \)  
- Guaranteed descent under Armijo conditions  
- Enhanced stability for noisy or non-convex landscapes via V2 and V3 line-search variants  

These foundations align with classical results in convex optimization and recent findings in fractional-order systems.

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## ✉️ Contact

For questions or collaboration inquiries, please feel free to open an issue or contact the maintainer.

