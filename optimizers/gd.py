## Vanilla Gradient Descent ( batch GD)

from utils.loss import mse_loss, mse_gradient 

def gradient_descent(X, y, lr=0.01, epochs=500):
    """
    BGD for linear reg.

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

    history = {"w":[], "b":[], "loss":[]}

    for _ in range(epochs):
        # compute gradients
        dw, db = mse_gradient(X, y, w, b)

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