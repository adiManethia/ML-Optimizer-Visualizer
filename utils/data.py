import numpy as np

def generate_linear_data(n_samples=100, w_true=2.0, b_true=3.0, noise_std=0.2, seed=42):
    """
    Generate linear regression data for optimizer comparison.

    Features:
    - Small and large X values to create variable gradients.
    - Noise added to show SGD/GD fluctuations.
    - Shuffling ensures optimizers cannot rely on ordering.
    
    Returns:
        X: np.ndarray, shape (n_samples,)
        y: np.ndarray, shape (n_samples,)
    """
    np.random.seed(seed)
    
    # Small X region (gradients small)
    X_small = np.linspace(0, 0.5, n_samples // 2)
    
    # Medium X region (gradients larger but not too big)
    X_medium = np.linspace(1.5, 3, n_samples // 4)
    
    # Occasional large X values (simulate spikes)
    X_large = np.linspace(5, 6, n_samples // 4)
    
    # Combine all
    X = np.concatenate([X_small, X_medium, X_large])
    
    # True linear function
    y = w_true * X + b_true
    
    # Add noise
    noise = np.random.normal(0, noise_std, size=n_samples)
    y += noise
    
    # Shuffle dataset
    perm = np.random.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    
    return X, y
