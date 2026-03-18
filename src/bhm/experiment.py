from dataclasses import dataclass

import jax
import jax.flatten_util

from bhm.curvature import compute_ggn, compute_hessian
from bhm.data import load_digits_data
from bhm.evaluate import approximation_error, per_sample_gradients, pseudo_inverse
from bhm.model import MLP
from bhm.train import cross_entropy_loss, train


@dataclass
class ExperimentConfig:
    depth: int
    width: int
    epochs: int
    lr: float = 0.03
    batch_size: int = 32
    seed: int = 0
    chunk_size: int = 64


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    hessian_error: float
    ggn_error: float
    train_loss: float
    num_params: int


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    key = jax.random.PRNGKey(config.seed)
    key, model_key, train_key = jax.random.split(key, 3)

    data = load_digits_data(seed=config.seed)

    model = MLP(depth=config.depth, width=config.width, key=model_key)
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    num_params = flat_params.shape[0]
    print(f"  Parameters: {num_params}")

    model = train(
        model,
        data.x_train,
        data.y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        key=train_key,
    )

    train_loss = float(cross_entropy_loss(model, data.x_train, data.y_train))
    print(f"  Final train loss: {train_loss:.4f}")

    print("  Computing Hessian...")
    H = compute_hessian(model, data.x_train, data.y_train)

    print("  Computing GGN...")
    G = compute_ggn(model, data.x_train, data.y_train, chunk_size=config.chunk_size)

    print("  Computing per-sample gradients...")
    grads = per_sample_gradients(model, data.x_train, data.y_train)

    print("  Computing pseudo-inverses...")
    H_inv = pseudo_inverse(H)
    G_inv = pseudo_inverse(G)

    print("  Computing approximation errors...")
    hessian_err = approximation_error(H, H_inv, grads)
    ggn_err = approximation_error(H, G_inv, grads)

    return ExperimentResult(
        config=config,
        hessian_error=hessian_err,
        ggn_error=ggn_err,
        train_loss=train_loss,
        num_params=num_params,
    )
