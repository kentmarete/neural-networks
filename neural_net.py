# imports
import numpy as np
# x1   x2   y
#----------------
# 0    0    0
# 1    1    0
# 0    1    1
# 1    0    1

# data: if two inputs are same return 0 else 1
x = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([[0], [0], [1], [1]])

# activation functions
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(inputs, params):
    w1, b1, w2, b2 = params
    h = relu(np.dot(inputs, w1) + b1)  # hidden layer
    y_hat = sigmoid(np.dot(h, w2) + b2)  # prediction
    return h, y_hat

def predict(inputs, params):
    # near 0: equal, near 1: not equal
    return forward(inputs, params)[1]

def train():
    # network: 2 inputs, 1 hidden layer (2 units), 1 output
    rng = np.random.default_rng(1)
    w1 = rng.normal(loc=0.0, scale=1.0, size=(2, 2))
    w2 = rng.normal(loc=0.0, scale=1.0, size=(2, 1))
    b1 = np.zeros(2)
    b2 = np.zeros(1)

    lr = 1.0  # learning rate: step size
    print(f"{'Step':>4}  {'Loss':>6}  Predictions")
    for step in range(1, 1501):
        # forward pass (matrix multiplication)
        h, y_hat = forward(x, (w1, b1, w2, b2))

        # Loss: measures how wrong we are
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        # backward pass: gradients
        d2 = (y_hat - y) / 4
        d1 = (np.dot(d2, w2.T) * (h > 0))

        # update: small step downhill
        w2 -= lr * np.dot(h.T, d2)
        b2 -= lr * d2.sum(axis=0)
        w1 -= lr * np.dot(x.T, d1)
        b1 -= lr * d1.sum(axis=0)
        if step == 1 or step % 150 == 0:
            print(f"{step:4d}  {loss:6.3f}  {y_hat.ravel().round(2)}")

    # results
    print("\nx1  x2  y  prediction")
    for (x1, x2), target, pred in zip(x, y[:, 0], y_hat[:, 0]):
        print(f"{x1:2d}  {x2:2d}  {target:d}  {pred:10.2f}")

    return w1, b1, w2, b2  # what the network learned

def main():
    params = train()

    # new inputs the network never saw in training
    new_x = np.array([[2, 2], [3, 4], [10, 10], [20, 22]])
    print("\nx1  x2  prediction  result")
    for (x1, x2), pred in zip(new_x, predict(new_x, params)[:, 0]):
        result = "equal" if pred < 0.5 else "not equal"
        print(f"{x1:2d}  {x2:2d}  {pred:10.2f}  {result}")

if __name__ == "__main__":
    main()
