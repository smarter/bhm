from dvclive.live import Live
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

    with Live(save_dvc_exp=False, dvcyaml=False) as live:
        live.log_metric("hessian_error", result.hessian_error, plot=False)
        live.log_metric("ggn_error", result.ggn_error, plot=False)
        live.log_metric("block_ggn_error", result.block_ggn_error, plot=False)
        live.log_metric("ekfac_error", result.ekfac_error, plot=False)
        live.log_metric("kfac_error", result.kfac_error, plot=False)
        live.log_metric("train_loss", result.train_loss, plot=False)
        live.log_metric("num_params", result.num_params, plot=False)

        live.log_plot(
            "approx_error",
            [
                {"method": "Hessian", "approx_error": result.hessian_error},
                {"method": "GGN", "approx_error": result.ggn_error},
                {"method": "B-GGN", "approx_error": result.block_ggn_error},
                {"method": "EK-FAC", "approx_error": result.ekfac_error},
                {"method": "K-FAC", "approx_error": result.kfac_error},
            ],
            x="method",
            y="approx_error",
            template="bar_horizontal",
            title="Approximation Error by Method",
            x_label="Approximation Error",
            y_label="Method",
        )


if __name__ == "__main__":
    main()
