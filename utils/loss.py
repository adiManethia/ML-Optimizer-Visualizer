# Loss function --> Mean Squared Error and Gradient

import numpy as np 

def mse_loss(y_true, y_pred):
    """
    MSE loss.

    Args: 
         y_true (np.ndarray) : True tragets
         y_pred (np.ndarray) : Predicted targets

    Returns:
        float: MSE loss
    """
    return np.mean((y_true - y_pred) **2)

def mse_gradient(X, y, w, b):
    """
    Compute gradients of MSE loss w.r.t. parameters w and b for linear reg. : y_pred = w * X + b

    Args:
        X : Input features, shape (n_samples,)
        y : True targets, shape (n_samples,)
        w : weight paramter
        b : Bias parameter

    Returns:
        dw : Gradient wrt w
        db : Gradient wrt b 

    """
    n = len(X)
    y_pred = w * X + b 

    dw = (2/n) * np.sum((y_pred - y) * X)
    db = (2/n) * np.sum(y_pred - y)

    return dw, db 