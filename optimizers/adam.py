# Combines Momentum + RMSprop
# Most used optimizer

from utils.loss import mse_gradient, mse_loss 

def adam(X, y, lr=0.1, epochs=500, beta1=0.9, beta2=0.999, eps=1e-6):
    """
    ADAM - Adaptive moment estimation

    Args:
        X : Input features
        y : Target values 
        lr : Learning rate 
        epochs : Number of iterations
        beta1 : Exponential decay rate for first moment
        beta2 : Exponential decay rate for seconf moment
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
    mw, mb = 0.0, 0.0 
    vw, vb = 0.0, 0.0 

    history = {"w":[], "b":[], "loss":[]} 

    for t in range(1, epochs+1):
        dw, db = mse_gradient(X, y, w, b)

        # update biased first moment estimate 
        mw = beta1 * mw + (1 - beta1) * dw 
        mb = beta1 * mb + (1 - beta1) * db 

        # update biased second moment estimate 
        vw = beta2 * vw + (1 - beta2) * (dw ** 2)
        vb = beta2 * vb + (1 - beta2) * (db ** 2)

        # bias correction 
        mw_hat = mw / (1 - beta1 ** t)
        mb_hat = mb / (1 - beta1 ** t)

        vw_hat = vw / (1 - beta2 ** t)
        vb_hat = vb / (1 - beta2 ** t)

        # update parameters
        w -= lr * mw_hat / ((vw_hat ** 0.5) + eps)
        b -= lr * mb_hat / ((vb_hat ** 0.5) + eps)

        # track 
        y_pred = w * X + b 
        loss = mse_loss(y, y_pred)

        history["w"].append(w)
        history["b"].append(b)
        history["loss"].append(loss)
    
    return history