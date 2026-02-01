# ML Optimizer Visualizer

**Compare optimization algorithms in linear regression with synthetic data**  

This project demonstrates how different optimizers — **Gradient Descent (GD), Stochastic Gradient Descent (SGD), Momentum, RMSProp, and Adam** — converge on a 1D linear regression problem. Users can generate controllable synthetic data, run different optimizers, and visualize results.

---

## Features

- Generate **synthetic linear regression data** with controllable noise and gradient scales  
- Implement five optimizers from scratch:
  - Vanilla **Gradient Descent (GD)**  
  - **Stochastic Gradient Descent (SGD)**  
  - **Momentum-based GD**  
  - **RMSProp**  
  - **Adam**  
- Compare optimizers visually using:
  - Loss curves (linear and log scale)  
  - Weight (`w`) and bias (`b`) convergence  
- Save plots for documentation or reports  
- Quantitative summary table of final losses and iterations to threshold  

---

## Installation

```bash
git clone https://github.com/adiManethia/ML-Optimizer-Visualizer.git
cd ML-Optimizer-Visualizer
pip install -r requirements.txt  # numpy, matplotlib





