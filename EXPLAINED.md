# How this neural network works

This program learns by example. We never write the rule "check if two numbers
are equal". We show it four examples with answers, and it adjusts itself until
its answers match.

Run it with `.venv/bin/python neural_net.py`.

## 1. The examples

```python
x = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])   # questions
y = np.array([[0], [0], [1], [1]])               # correct answers
```

`0` means the numbers are equal, `1` means they are different.

## 2. The network

```
x1 ─┐     ┌─ hidden unit 1 ─┐
    ├──X──┤                 ├── output (0 to 1)
x2 ─┘     └─ hidden unit 2 ─┘
```

- **Weights (`w1`, `w2`)** are like volume knobs: how strongly each number
  affects the next part. They start random.
- **Biases (`b1`, `b2`)** are extra amounts added on top.

## 3. Helper functions

- **`relu`** turns negatives into 0 and leaves positives alone. It lets the
  network learn patterns a straight line can't.
- **`sigmoid`** squeezes any number into the range 0 to 1, so the output reads
  like a confidence that the numbers are different.

## 4. Training: guess, check, adjust

`train()` repeats these steps 1,500 times:

1. **Guess (`forward`)**: compute the prediction, `y_hat`.
2. **Measure the error (`loss`)**: one number for how wrong the guesses are.
   Lower is better.
3. **Find who is to blame (`d2`, `d1`)**: `d2` is the error at the output, and
   `d1` passes it back to the hidden units (*backpropagation*).
4. **Adjust (`-=`)**: nudge every weight and bias to reduce the error.

It is like walking down a foggy hill: feel which way is down, take a step,
repeat. `lr` (learning rate) is the step size.

```
Step    Loss  Predictions
   1   0.676  [0.5  0.65 0.57 0.66]   <- mostly unsure
 150   0.014  [0.02 0.02 0.99 0.99]   <- nearly right
```

## 5. Predicting new pairs

`train()` returns the learned weights and biases. `predict()` uses them without
any more learning. Below 0.5 means "equal", otherwise "not equal".

| x1 | x2 | prediction | result    |
|---:|---:|-----------:|-----------|
|  2 |  2 |       0.00 | equal     |
|  3 |  4 |       1.00 | not equal |
| 10 | 10 |       0.00 | equal     |
| 20 | 22 |       1.00 | not equal |

To try other pairs, edit `new_x` in `main()`.

## 6. Why it works on numbers it never saw

The hidden units became roughly:

- unit 1: `3.22 × (x2 − x1)`
- unit 2: `3.32 × (x1 − x2)`

`relu` removes negatives, so one unit reacts when `x2` is bigger and the other
when `x1` is bigger. If the numbers are equal, both stay near 0. It learned to
measure the **difference**.

## 7. Limits

- **Close decimals:** `[2, 2.3]` is called equal. Training only had gaps of 0 and 1.
- **Very large numbers:** `[1000000, 1000000]` is wrongly called not equal.
- **Settings:** these results are for seed 1 and `lr = 1.0`. Others may differ.
