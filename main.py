from utils.data import generate_linear_data 
from optimizers.gd import gradient_descent 
from optimizers.sgd import stochastic_gradient_descent 
from optimizers.momentum import momentum_gradient_descent 
from optimizers.rmsprop import rmsprop 
from optimizers.adam import adam 
from utils.visualize import plot_loss_log_scale, plot_loss_separately, plot_parameters, print_optimizer_summary


def main():
    # generate data 
    X, y = generate_linear_data()

    # run optimizers 
    histories = {
        "GD" : gradient_descent(X,y),
        "SGD" : stochastic_gradient_descent(X, y),
        "Momentum" : momentum_gradient_descent(X, y),
        "RMSProp" : rmsprop(X, y),
        "Adam" : adam(X, y)
    }

    # visualizations
   
    plot_loss_log_scale(histories)
    plot_loss_separately(histories)
    plot_parameters(histories)
    print_optimizer_summary(histories)
    

if __name__ == "__main__":
    main()