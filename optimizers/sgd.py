# Stochastic gradient descent --> shuffled data, noiser updates but faster to GD 

import numpy as np 
from utils.loss import mse_gradient, mse_loss 

def stochastic_gradient_descent(X, y, lr=0.01, epochs=500):
    """
    SGD for linear reg. 

    Args:
        X : Input features
        y : Target values 
        lr : Learning rate 
        epochs : Number of iterations

    Returns:
        history (dict):
          {
          "w" : list of w values,
          "b" : list of b values,
          "loss" : list of loss values
          }

    """

    w, b = 0.0, 0.0 
    n = len(X)

    history = {"w":[], "b":[], "loss":[]} 

    for _ in range(epochs):
        indices = np.random.permutation(n)

        for i in indices:
            Xi = X[i:i+1]
            yi = y[i:i+1] 

            # compute gradient using a single random sample -> nosiy update, jitter in plots
            dw, db = mse_gradient(Xi, yi, w, b) 

            # update parameters 
            w -= lr * dw 
            b -= lr * db 

            # track 
            y_pred = w * X + b 
            loss = mse_loss(y, y_pred)

            history["w"].append(w)
            history["b"].append(b)
            history["loss"].append(loss)

    return history




