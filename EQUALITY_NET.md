# How `equality_net.py` works, step by step

The library answers one question: **are these two numbers equal?**
It learns from examples, then uses what it learned on new numbers.

```python
from equality_net import predict, train

params = train()                              # learn once
predict([[10, 10], [20, 22]], params)         # ask as often as you like
```

Each answer is between 0 and 1. Near 0 means equal, near 1 means not equal.
`main.py` uses 0.5 as the dividing line.

```
[x1, x2] -> gap -> hidden layer (2 units) -> output (0 to 1)
```

## Step 1: The training examples

`X_TRAIN` holds pairs and `Y_TRAIN` holds the answers (0 = equal,
1 = different). There are small pairs such as `[0, 1]` and large ones such as
`[1_000_000, 1_000_001]`.

## Step 2: Turn each pair into a gap

```python
np.abs(inputs[:, :1] - inputs[:, 1:])     # |x1 - x2|
```

| Pair                 | Gap |
|----------------------|----:|
| `[10, 10]`           |   0 |
| `[20, 22]`           |   2 |
| `[1000000, 1000001]` |   1 |

This is what makes large numbers work. Equal numbers always give exactly 0,
whatever their size. `neural_net.py` had to learn this from the raw numbers, and
its small errors grew with the size of the input.

The large training pairs become gaps of 0 and 1, the same as the small pairs.

## Step 3: Start with random weights

`train()` creates the numbers it will learn:

| Name | Shape  | Role                             |
|------|--------|----------------------------------|
| `w1` | (1, 2) | weights from gap to hidden layer |
| `b1` | (2,)   | hidden layer biases              |
| `w2` | (2, 1) | weights from hidden to output    |
| `b2` | (1,)   | output bias                      |

Weights start random and biases start at 0. `seed=1` gives the same start
every time.

## Step 4: Make a guess (`forward`)

```python
h = relu(np.dot(gap(inputs), w1) + b1)
y_hat = sigmoid(np.dot(h, w2) + b2)
```

- `relu` turns negative values into 0.
- `sigmoid` squeezes the result into 0 to 1. Its input is limited to ±500 to
  avoid overflow.

## Step 5: Measure the error (`loss`)

The loss is one number for how wrong the guesses are, averaged over all
examples. Confident wrong guesses cost the most. Guesses are kept just away
from 0 and 1 because `log(0)` is not defined.

## Step 6: Work out who is to blame (backpropagation)

```python
d2 = (y_hat - y) / len(x)             # error at the output
d1 = np.dot(d2, w2.T) * (h > 0)       # share passed back to the hidden units
```

`(h > 0)` is there because a unit that `relu` switched off gets no blame.

## Step 7: Take a small step downhill

Every weight and bias moves a little in the direction that lowers the loss.
`lr` sets the step size. Steps 4 to 7 repeat 1,500 times:

```
Step    Loss
   1   0.987
 150   0.009
1500   0.001
```

`train()` then returns `w1, b1, w2, b2`: everything the network knows.

## Step 8: Predict

`predict()` runs only step 4 with the learned values. Nothing is retrained.

With the default settings, the learned values are approximately:

```
w1 = [4.003, 0.351]    b1 = [-0.001, -0.471]
w2 = [3.956, -1.006]   b2 = -6.531
```

**`[10, 10]`, gap 0**

```
hidden = relu(0 × w1 + b1)      = [0, 0]
score  = 0 + 0 - 6.531          = -6.531
output = sigmoid(-6.531)        = 0.001   -> equal
```

**`[1000000, 1000001]`, gap 1**

```
hidden = relu(1 × w1 + b1)      = [4.003, 0]
score  = 4.003 × 3.956 - 6.531  = 9.301
output = sigmoid(9.301)         = 0.9999  -> not equal
```

In short, `b2` makes "equal" the default answer, and hidden unit 1 pushes the
score up when there is a gap. (Worked values are rounded.)

## Settings for `train()`

| Setting   | Default     | Meaning                    |
|-----------|-------------|----------------------------|
| `x`, `y`  | built-in    | training pairs and answers |
| `lr`      | 1.0         | step size                  |
| `steps`   | 1500        | training rounds            |
| `seed`    | 1           | random start               |
| `verbose` | `True`      | print the loss table       |

If you pass your own data, `y` needs one answer per row: `[[0], [1], ...]`.

## Limits

- **Close decimals:** `[2, 2.3]` is called equal (0.14). Training only has
  gaps of 0 and 1.
- **Decimals mixed with huge numbers:** one decimal makes NumPy treat the whole
  list as decimals. Above about 9 × 10¹⁵, neighbouring numbers can then look equal.
- **Training gaps:** keep them around 0 to 1. Huge gaps can make training blow up.
- **Seed:** some seeds get stuck and never learn.
- **We give it the key idea:** the gap is supplied by us. The network only learns
  how big a gap counts as "not equal".
