## Optimization Algorithms for Training Neural Networks in RIS Systems

This repository contains the implementation and simulation scripts for my research on algorithmic optimization in Reconfigurable Intelligent Surface (RIS) systems.  
The project focuses on using optimization algorithms to improve neural-network model fitting and 
predictive accuracy in realistic RIS environments.
## Research Motivation

Modern wireless communication systems, such as RIS-assisted networks, involve complex, high-dimensional environments that cannot be easily described by explicit mathematical models.  
To capture the underlying physical behaviors, data-driven models are often used to approximate system dynamics.  
However, achieving high-fidelity model fitting to realistic RIS behavior remains a challenge.  
This project explores how optimization algorithms can be used to train neural-network models that more accurately represent real RIS systems.
## Methodology

The project was implemented through the following key steps:

1. **RIS System Modeling:** Built a realistic Reconfigurable Intelligent Surface (RIS) simulation framework to generate channel data and system features.  
2. **Algorithmic Training:** Trained neural networks using four optimization algorithms — Gradient Descent (GD), Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Simulated Annealing (SA) — to study their convergence and stability.  
3. **Model Integration:** Integrated the best-performing trained neural network into the RIS simulation for end-to-end performance evaluation.  
4. **Baseline Comparison:** Implemented a linear regression baseline model within the same RIS setup for reference.  
5. **Result Analysis:** Compared the neural-network and baseline models in terms of R², MSE, RMSE, and MAE, demonstrating improved fitting and prediction accuracy in realistic RIS environments.
## Results & Key Findings

### 1. Optimization Algorithm Comparison
Four optimization algorithms were used to train the neural network, and their convergence performance was compared in terms of **training speed** (epochs to reach Loss < 0.1) and **robustness** (standard deviation of final 50 losses).

| Algorithm | Epochs (Loss < 0.1) | Robustness (Std of Final 50 Losses) |
|------------|---------------------|-------------------------------------|
| **Gradient Descent (GD)** | **81** | **0.000515** |
| Particle Swarm Optimization (PSO) | 87 | 0.001067 |
| Genetic Algorithm (GA) | 154 | 0.003393 |
| Simulated Annealing (SA) | 1269 | 0.000000 |

**Finding:**  
Gradient Descent (GD) achieved the fastest convergence with stable performance, while GA and PSO also showed good robustness.  
SA demonstrated stability but required significantly more iterations.

---

### 2. Model Performance Comparison
The trained neural network was compared against a linear baseline model under the same RIS setup.  
Evaluation metrics include **MSE**, **RMSE**, **MAE**, and **R²**.

| Model | MSE | RMSE | MAE | R² |
|--------|------|------|------|------|
| Linear Regression (Baseline) | 0.0885 | 0.2974 | 0.1532 | 0.8415 |
| **Neural Network (Optimized)** | **0.0385** | **0.1962** | **0.1250** | **0.9292** |

**Finding:**  
The neural-network model trained with optimized algorithms improved system fitting accuracy by **over 20%** in R² compared to the baseline, demonstrating superior generalization and predictive stability in realistic RIS environments.

---

### 3. Visualization

**Figure 1.** Convergence curves of four optimization algorithms.  
**Figure 2.** Predicted vs. True Sum Rate (Baseline vs. Neural Network).  


