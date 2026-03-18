from bhm.experiment import ExperimentConfig, ExperimentResult, run_experiment


def generate_configs() -> list[tuple[str, ExperimentConfig]]:
    configs: list[tuple[str, ExperimentConfig]] = []

    # Epoch sweep: epochs in {10, 100, 1000}, depth=8, width=16
    for epochs in [10, 100, 1000]:
        configs.append(("epoch", ExperimentConfig(depth=8, width=16, epochs=epochs)))

    # Depth sweep: depth in {1, 4, 8}, epochs=100, width=16
    # (depth=8, epochs=100 already covered in epoch sweep)
    for depth in [1, 4]:
        configs.append(("depth", ExperimentConfig(depth=depth, width=16, epochs=100)))

    # Width sweep: width in {32, 64, 128}, epochs=100, depth=1
    # (depth=1, width=16 already covered in depth sweep -- but width=16 isn't in this sweep)
    for width in [32, 64, 128]:
        configs.append(("width", ExperimentConfig(depth=1, width=width, epochs=100)))

    return configs


def print_results(results: list[tuple[str, ExperimentResult]]) -> None:
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    header = f"{'Sweep':<12} {'Depth':>5} {'Width':>5} {'Epochs':>6} {'Params':>7} {'H Error':>12} {'GGN Error':>12} {'Train Loss':>10}"
    print(header)
    print("-" * len(header))

    for sweep, r in results:
        c = r.config
        print(
            f"{sweep:<12} {c.depth:>5} {c.width:>5} {c.epochs:>6} {r.num_params:>7} "
            f"{r.hessian_error:>12.6f} {r.ggn_error:>12.6f} {r.train_loss:>10.4f}"
        )


def main() -> None:
    configs = generate_configs()
    results: list[tuple[str, ExperimentResult]] = []

    for i, (sweep, config) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Experiment {i + 1}/{len(configs)} [{sweep}]: depth={config.depth}, width={config.width}, epochs={config.epochs}")
        print(f"{'='*60}")

        result = run_experiment(config)
        results.append((sweep, result))

        print(f"  -> Hessian error: {result.hessian_error:.6f}")
        print(f"  -> GGN error:     {result.ggn_error:.6f}")

    print_results(results)


if __name__ == "__main__":
    main()
