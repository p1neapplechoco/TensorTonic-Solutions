import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def _bce_loss(y, y_pred) -> float:
    n = len(y)
    return (
        -1
        / n
        * np.sum(
            [
                y[i] * np.log(y_pred[i]) + (1 - y[i]) * np.log(1 - y_pred[i])
                for i in range(n)
            ]
        )
    )


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros((X.shape[1],))
    b = 0

    n = len(y)

    for step in range(steps):
        z = (X @ w) + b
        y_pred = _sigmoid(z)

        loss = _bce_loss(y=y, y_pred=y_pred)
        print(f"Current loss: {loss}")

        loss_grad_w = np.sum((y - y_pred) @ X)
        loss_grad_b = 1 / n * np.sum(y - y_pred)

        w = w + lr * loss_grad_w
        b = b + lr * loss_grad_b

    return (w, b)


# X = np.array([[0], [1], [2], [3]])
# y = np.array([0, 0, 1, 1])

# print(train_logistic_regression(X, y))
