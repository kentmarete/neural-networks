# Neural networks from scratch

Two small neural networks, written with only NumPy, that learn to answer one
question: **are these two numbers equal?**

We never write the rule "check if two numbers are equal". We show examples with
answers, and the network adjusts itself until its answers match.

| File | What it is |
|------|------------|
| `neural_net.py` | First version. Learns from the raw numbers; fails on very large ones. |
| `equality_net.py` | Improved, reusable library. Learns from the gap between the numbers, so large integers work. |
| `main.py` | Demo of `equality_net.py`. |

## Quick start

You need Python 3.9 or newer and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/kentmarete/neural-networks.git
cd neural-networks
uv sync

uv run main.py         # equality_net demo
uv run neural_net.py   # first version
```

Use the library in your own code:

```python
from equality_net import predict, train

params = train()                              # learn once
predict([[10, 10], [20, 22]], params)         # ask as often as you like
```

Each answer is between 0 and 1. Near 0 means equal, near 1 means not equal. The
demos use 0.5 as the dividing line.

Output of `main.py`, after the training log:

```
          x1            x2  prediction  result
           2             2        0.00  equal
           3             4        1.00  not equal
          10            10        0.00  equal
          20            22        1.00  not equal
     1000000       1000000        0.00  equal
     1000000       1000001        1.00  not equal
  5000000000    5000000000        0.00  equal
  5000000000    5000000003        1.00  not equal
```

## Shared ideas

Both networks have one hidden layer with two units, and one output.

- **Weights (`w1`, `w2`)** are like volume knobs: how strongly each value affects
  the next part. They start random.
- **Biases (`b1`, `b2`)** are extra amounts added on top. They start at 0.
- **`relu`** turns negatives into 0 and leaves positives alone. It lets the
  network learn patterns a straight line can't.
- **`sigmoid`** squeezes any number into the range 0 to 1, so the output reads
  like a confidence that the numbers are different.

### Training: guess, check, adjust

`train()` repeats these steps 1,500 times:

1. **Guess (`forward`)**: compute the prediction, `y_hat`.
2. **Measure the error (`loss`)**: one number for how wrong the guesses are,
   averaged over all examples. Lower is better, and confident wrong guesses cost
   the most.
3. **Find who is to blame (`d2`, `d1`)**: `d2` is the error at the output, and
   `d1` passes it back to the hidden units (*backpropagation*). A unit that
   `relu` switched off gets no blame, which is what `(h > 0)` does.
4. **Adjust (`-=`)**: nudge every weight and bias to reduce the error.

It is like walking down a foggy hill: feel which way is down, take a step,
repeat. `lr` (learning rate) is the step size.

`train()` returns `w1, b1, w2, b2`: everything the network knows. `predict()`
runs only the guess step with those values, without any more learning.

## Version 1: `neural_net.py`

```
x1 ─┐     ┌─ hidden unit 1 ─┐
    ├──X──┤                 ├── output (0 to 1)
x2 ─┘     └─ hidden unit 2 ─┘
```

It trains on four examples, where 0 means equal and 1 means different:

```python
x = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])   # questions
y = np.array([[0], [0], [1], [1]])               # correct answers
```

Training log (shortened):

```
Step    Loss  Predictions
   1   0.676  [0.5  0.65 0.57 0.66]   <- mostly unsure
 150   0.014  [0.02 0.02 0.99 0.99]   <- nearly right
```

Pairs it never saw in training:

| x1 | x2 | prediction | result    |
|---:|---:|-----------:|-----------|
|  2 |  2 |       0.00 | equal     |
|  3 |  4 |       1.00 | not equal |
| 10 | 10 |       0.00 | equal     |
| 20 | 22 |       1.00 | not equal |

To try other pairs, edit `new_x` in `main()` in `neural_net.py`.

### Why it works on new numbers

The hidden units became roughly:

- unit 1: `3.22 × (x2 − x1)`
- unit 2: `3.32 × (x1 − x2)`

`relu` removes negatives, so one unit reacts when `x2` is bigger and the other
when `x1` is bigger. If the numbers are equal, both stay near 0. It learned to
measure the **difference**.

### Limits

- **Close decimals:** `[2, 2.3]` is called equal. Training only had gaps of 0 and 1.
- **Very large numbers:** `[1000000, 1000000]` is wrongly called not equal. The
  small errors in the learned difference grow with the size of the input.
- **Settings:** these results are for seed 1 and `lr = 1.0`. Others may differ.

## Version 2: `equality_net.py`

```
[x1, x2] -> gap -> hidden layer (2 units) -> output (0 to 1)
```

### The gap

Each pair is first turned into one number:

```python
np.abs(inputs[:, :1] - inputs[:, 1:])     # |x1 - x2|
```

| Pair                 | Gap |
|----------------------|----:|
| `[10, 10]`           |   0 |
| `[20, 22]`           |   2 |
| `[1000000, 1000001]` |   1 |

This is what makes large numbers work: equal numbers always give exactly 0,
whatever their size. `X_TRAIN` holds small and large pairs, but all become gaps
of 0 and 1.

### The network

```python
h = relu(np.dot(gap(inputs), w1) + b1)
y_hat = sigmoid(np.dot(h, w2) + b2)
```

| Name | Shape  | Role                             |
|------|--------|----------------------------------|
| `w1` | (1, 2) | weights from gap to hidden layer |
| `b1` | (2,)   | hidden layer biases              |
| `w2` | (2, 1) | weights from hidden to output    |
| `b2` | (1,)   | output bias                      |

Two safeguards keep the arithmetic stable: the `sigmoid` input is limited to ±500
to avoid overflow, and the loss keeps guesses just away from 0 and 1 because
`log(0)` is not defined.

Training log (shortened):

```
Step    Loss
   1   0.987
 150   0.009
1500   0.001
```

### A worked example

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

### Settings for `train()`

| Setting   | Default  | Meaning                    |
|-----------|----------|----------------------------|
| `x`, `y`  | built-in | training pairs and answers |
| `lr`      | `1.0`    | step size                  |
| `steps`   | `1500`   | training rounds            |
| `seed`    | `1`      | random start               |
| `verbose` | `True`   | print the loss table       |

If you pass your own data, `y` needs one answer per row: `[[0], [1], ...]`.

### Limits

- **Close decimals:** `[2, 2.3]` is called equal (0.14). Training only has gaps
  of 0 and 1.
- **Decimals mixed with huge numbers:** one decimal makes NumPy treat the whole
  list as decimals. Above about 9 × 10¹⁵, neighbouring numbers can then look equal.
- **Training gaps:** keep them around 0 to 1. Huge gaps can make training blow up.
- **Seed:** many seeds get stuck and never learn (9 of seeds 0 to 19).
- **We give it the key idea:** the gap is supplied by us. The network only learns
  how big a gap counts as "not equal".
