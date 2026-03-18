import json
import sys

from omegaconf import OmegaConf

from bhm.experiment import ExperimentConfig, run_experiment


def main() -> None:
    cfg = OmegaConf.load("params.yaml")

    config = ExperimentConfig(
        depth=cfg.experiment.depth,  # pyright: ignore[reportAttributeAccessIssue]
        width=cfg.experiment.width,  # pyright: ignore[reportAttributeAccessIssue]
        epochs=cfg.experiment.epochs,  # pyright: ignore[reportAttributeAccessIssue]
        lr=cfg.train.lr,  # pyright: ignore[reportAttributeAccessIssue]
        batch_size=cfg.train.batch_size,  # pyright: ignore[reportAttributeAccessIssue]
        seed=cfg.seed,  # pyright: ignore[reportAttributeAccessIssue]
        chunk_size=cfg.chunk_size,  # pyright: ignore[reportAttributeAccessIssue]
    )

    result = run_experiment(config)

    metrics = {
        "hessian_error": result.hessian_error,
        "ggn_error": result.ggn_error,
        "train_loss": result.train_loss,
        "num_params": result.num_params,
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to results/metrics.json", file=sys.stderr)


if __name__ == "__main__":
    main()
