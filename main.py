from equality_net import predict, train


def main():
    params = train()

    # values to check
    pairs = [
        [2, 2], [3, 4], [10, 10], [20, 22],
        [1_000_000, 1_000_000], [1_000_000, 1_000_001],
        [5_000_000_000, 5_000_000_000], [5_000_000_000, 5_000_000_003],
    ]
    print(f"\n{'x1':>12}  {'x2':>12}  prediction  result")
    for (x1, x2), pred in zip(pairs, predict(pairs, params)):
        result = "equal" if pred < 0.5 else "not equal"
        print(f"{x1:12d}  {x2:12d}  {pred:10.2f}  {result}")


if __name__ == "__main__":
    main()
