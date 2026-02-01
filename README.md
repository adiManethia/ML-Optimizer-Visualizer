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
## Data Generation

We generate a **synthetic linear dataset**:

$$y = w_{\text{true}} \cdot X + b_{\text{true}} + \epsilon$$

Where:
- $x$ = input feature, ranging across small and large values to create varying gradient magnitudes  
- $w_{\text{true}}$, $b_{\text{true}}$ = true slope and intercept (default 2.0 and 3.0)  
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ = Gaussian noise to simulate real-world imperfections  
- Mixed small and large $x$ values create **ill-conditioned gradients** → highlights optimizer differences  

## Installation

```bash
git clone https://github.com/adiManethia/ML-Optimizer-Visualizer.git
cd ML-Optimizer-Visualizer
pip install -r requirements.txt  # numpy, matplotlib
```
## Usage 
```bash
python main.py
```
This will:
- Generate synthetic data (small + medium + large X values, moderate noise)
- Train all optimizer on the data
- Save plots in the ```outputs/``` folder
- Print a summary table like:
  ```bash
  Optimizer Comparison Summary
  --------------------------------------------------
  Optimizer  | Final Loss   | Iters < 0.05
  --------------------------------------------------
  GD         | 0.032646     | 255
  SGD        | 0.039081     | 248
  Momentum   | 0.032568     | 240
  RMSProp    | 0.032864     | 308
  Adam       | 0.032512     | 99
  ```
  

## Why Adam Wins in This Demo
- Dataset has mixed gradient magnitudes --> small, medium, large X values
- Noise introduces fluctuations --> GD/SGD oscillate more
- Adam adapts learning rate per parameter --> converges faster









