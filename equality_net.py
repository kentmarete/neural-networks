"""Small neural network library: are two numbers equal?

    params = train()
    predict([[2, 2], [3, 4]], params)   # near 0: equal, near 1: not equal
"""
import numpy as np

# training data: same numbers -> 0, different -> 1
# includes very large pairs as well as small ones
X_TRAIN = np.array([
    [0, 0], [1, 1], [0, 1], [1, 0],
    [1_000_000, 1_000_000], [1_000_000, 1_000_001],
    [1_000_000_000, 1_000_000_000], [1_000_000_001, 1_000_000_000],
])
Y_TRAIN = np.array([[0], [0], [1], [1], [0], [1], [0], [1]])


# activation functions
def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip avoids overflow


def gap(inputs):
    """|x1 - x2| for each pair: exactly 0 when equal, whatever their size."""
    inputs = np.asarray(inputs)
    return np.abs(inputs[:, :1] - inputs[:, 1:])


def forward(inputs, params):
    w1, b1, w2, b2 = params
    h = relu(np.dot(gap(inputs), w1) + b1)  # hidden layer
    y_hat = sigmoid(np.dot(h, w2) + b2)  # prediction
    return h, y_hat


def train(x=X_TRAIN, y=Y_TRAIN, lr=1.0,
          steps=1500, seed=1, verbose=True):
    """Train the network and return its weights and biases.

    Keep training gaps small (about 0 to 1): huge gaps make the steps explode.
    The numbers themselves can be large.
    """
    x, y = np.asarray(x), np.asarray(y)

    # network: 1 input (the gap), 1 hidden layer (2 units), 1 output
    rng = np.random.default_rng(seed)
    w1 = rng.normal(loc=0.0, scale=1.0, size=(1, 2))
    w2 = rng.normal(loc=0.0, scale=1.0, size=(2, 1))
    b1 = np.zeros(2)
    b2 = np.zeros(1)

    if verbose:
        print(f"{'Step':>4}  {'Loss':>6}")
    for step in range(1, steps + 1):
        # forward pass
        h, y_hat = forward(x, (w1, b1, w2, b2))

        # Loss: measures how wrong we are
        p = np.clip(y_hat, 1e-12, 1 - 1e-12)  # avoids log(0)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        # backward pass: gradients
        d2 = (y_hat - y) / len(x)
        d1 = np.dot(d2, w2.T) * (h > 0)

        # update: small step downhill
        w2 -= lr * np.dot(h.T, d2)
        b2 -= lr * d2.sum(axis=0)
        w1 -= lr * np.dot(gap(x).T, d1)
        b1 -= lr * d1.sum(axis=0)
        if verbose and (step == 1 or step % 150 == 0):
            print(f"{step:4d}  {loss:6.3f}")

    return w1, b1, w2, b2


def predict(inputs, params):
    """One value per pair: near 0 means equal, near 1 means not equal."""
    return forward(inputs, params)[1][:, 0]
