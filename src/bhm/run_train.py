from pathlib import Path

import equinox as eqx
import jax
import jax.flatten_util
from dvclive.live import Live
from omegaconf import OmegaConf

from bhm.data import load_digits_data
from bhm.model import MLP
from bhm.train import train


def main() -> None:
    cfg = OmegaConf.load("params.yaml")

    key = jax.random.PRNGKey(cfg.seed)  # pyright: ignore[reportAttributeAccessIssue]
    key, model_key, train_key = jax.random.split(key, 3)

    data = load_digits_data(seed=cfg.seed)  # pyright: ignore[reportAttributeAccessIssue]
    model = MLP(
        depth=cfg.experiment.depth,  # pyright: ignore[reportAttributeAccessIssue]
        width=cfg.experiment.width,  # pyright: ignore[reportAttributeAccessIssue]
        key=model_key,
    )

    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    print(f"  Parameters: {flat_params.shape[0]}")

    with Live(dir="dvclive/train", save_dvc_exp=False, dvcyaml=False) as live:
        live.log_metric("num_params", flat_params.shape[0], plot=False)

        model = train(
            model,
            data.x_train,
            data.y_train,
            epochs=cfg.experiment.epochs,  # pyright: ignore[reportAttributeAccessIssue]
            batch_size=cfg.train.batch_size,  # pyright: ignore[reportAttributeAccessIssue]
            lr=cfg.train.lr,  # pyright: ignore[reportAttributeAccessIssue]
            key=train_key,
            x_test=data.x_test,
            y_test=data.y_test,
            live=live,
        )

    Path("models").mkdir(exist_ok=True)
    eqx.tree_serialise_leaves("models/model.eqx", model)
    print("  Model saved to models/model.eqx")


if __name__ == "__main__":
    main()
