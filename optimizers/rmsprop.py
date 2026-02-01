# RMSprop - Root mean square propagation. -> Scale LR based on magnitude of recent gradients by normalizing gradient.

from utils.loss import mse_loss, mse_gradient

def rmsprop(X, y, lr=0.01, epochs=500, beta=0.9, eps=1e-6):
    """
    RMSprop 

    Args:
        X : Input features
        y : Target values 
        lr : Learning rate 
        epochs : Number of iterations
        beta : decay rate for squared gradients
        eps: small value to avois division by zero

    Returns:
        history (dict):
          {
          "w" : list of w values,
          "b" : list of b values,
          "loss" : list of loss values
          }
    """
    w, b = 0.0, 0.0 

    sw, sb = 0.0, 0.0 

    history = {"w":[], "b":[], "loss":[]} 

    for _ in range(epochs):
        dw, db = mse_gradient(X, y, w, b)

        # update running average of squared gradients
        sw = beta * sw + (1 - beta) * (dw ** 2)
        sb = beta * sb + (1 - beta) * (db ** 2)

        # update parameters
        w -= lr * dw / ((sw ** 0.5) + eps)
        b -= lr * db / ((sb ** 0.5) + eps)

        # track 
        y_pred = w * X + b 
        loss = mse_loss(y, y_pred)

        history["w"].append(w)
        history["b"].append(b)
        history["loss"].append(loss)
        
    return history