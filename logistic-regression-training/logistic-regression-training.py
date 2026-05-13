import numpy as np

def _sigmoid(z):
    return np.where(z >= 0,
                    1/(1+np.exp(-z)),
                    np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N = X.shape[0]          # number of samples
    n_features = X.shape[1]

    w = np.zeros(n_features)
    b = 0

    for i in range(steps):

        # Linear equation
        z = X @ w + b

        # Prediction
        p = _sigmoid(z)

        # Gradients
        dl_dw = (X.T @ (p - y)) / N
        dl_db = np.mean(p - y)

        # Update
        w = w - lr * dl_dw
        b = b - lr * dl_db

    return w, b