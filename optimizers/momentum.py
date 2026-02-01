# Extended GD with velocity--> reduce oscillations--> convergence in consistent direction

import numpy as np 
from utils.loss import mse_loss, mse_gradient 

def momentum_gradient_descent(X, y, lr=0.01, epochs=500, beta=0.9):
    """
    GD with momentum

    Args:
        X : Input features
        y : Target values 
        lr : Learning rate 
        epochs : Number of iterations
        beta : momentum coefficient

    Returns:
        history (dict):
          {
          "w" : list of w values,
          "b" : list of b values,
          "loss" : list of loss values
          }
    """

    w, b = 0.0, 0.0 

    vw, vb = 0.0, 0.0 

    history = {"w":[], "b":[], "loss":[]} 

    for _ in range(epochs):
        dw, db = mse_gradient(X, y, w, b)

        # update velocity
        vw = beta * vw + (1 - beta) * dw 
        vb = beta * vb + (1 - beta) * db 

        # update parameters
        w -= lr * vw 
        b -= lr * vb 

        # track 
        y_pred = w * X + b 
        loss = mse_loss(y, y_pred)

        history["w"].append(w)
        history["b"].append(b)
        history["loss"].append(loss)

    return history 