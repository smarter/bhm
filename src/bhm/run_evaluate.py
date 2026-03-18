import equinox as eqx
import jax
from dvclive.live import Live
from omegaconf import OmegaConf

from bhm.curvature import (
    compute_block_ggn,
    compute_ekfac,
    compute_ggn,
    compute_hessian,
    compute_kfac,
)
from bhm.data import load_digits_data
from bhm.evaluate import approximation_error, per_sample_gradients, pseudo_inverse
from bhm.model import MLP


def main() -> None:
    cfg = OmegaConf.load("params.yaml")

    data = load_digits_data(seed=cfg.seed)  # pyright: ignore[reportAttributeAccessIssue]

    # Load trained model
    model_skeleton = MLP(
        depth=cfg.experiment.depth,  # pyright: ignore[reportAttributeAccessIssue]
        width=cfg.experiment.width,  # pyright: ignore[reportAttributeAccessIssue]
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_deserialise_leaves("models/model.eqx", model_skeleton)

    chunk_size: int = cfg.chunk_size  # pyright: ignore[reportAssignmentType]

    print("  Computing Hessian...")
    H = compute_hessian(model, data.x_train, data.y_train, chunk_size=chunk_size)

    print("  Computing GGN...")
    G = compute_ggn(model, data.x_train, data.y_train, chunk_size=chunk_size)

    print("  Computing Block-diagonal GGN...")
    BG = compute_block_ggn(model, data.x_train, data.y_train, chunk_size=chunk_size)

    print("  Computing K-FAC...")
    KF = compute_kfac(model, data.x_train, data.y_train)

    print("  Computing EK-FAC...")
    EKF = compute_ekfac(model, data.x_train, data.y_train)

    print("  Computing per-sample gradients...")
    grads = per_sample_gradients(model, data.x_train, data.y_train)

    print("  Computing pseudo-inverses...")
    H_inv = pseudo_inverse(H)
    G_inv = pseudo_inverse(G)
    BG_inv = pseudo_inverse(BG)
    KF_inv = pseudo_inverse(KF)
    EKF_inv = pseudo_inverse(EKF)

    print("  Computing approximation errors...")
    hessian_err = approximation_error(H, H_inv, grads)
    ggn_err = approximation_error(H, G_inv, grads)
    block_ggn_err = approximation_error(H, BG_inv, grads)
    kfac_err = approximation_error(H, KF_inv, grads)
    ekfac_err = approximation_error(H, EKF_inv, grads)

    with Live(dir="dvclive/evaluate", save_dvc_exp=False, dvcyaml=False) as live:
        live.log_metric("hessian_error", hessian_err, plot=False)
        live.log_metric("ggn_error", ggn_err, plot=False)
        live.log_metric("block_ggn_error", block_ggn_err, plot=False)
        live.log_metric("ekfac_error", ekfac_err, plot=False)
        live.log_metric("kfac_error", kfac_err, plot=False)

        live.log_plot(
            "approx_error",
            [
                {"method": "Hessian", "approx_error": hessian_err},
                {"method": "GGN", "approx_error": ggn_err},
                {"method": "B-GGN", "approx_error": block_ggn_err},
                {"method": "EK-FAC", "approx_error": ekfac_err},
                {"method": "K-FAC", "approx_error": kfac_err},
            ],
            x="method",
            y="approx_error",
            template="bar_horizontal",
            title="Approximation Error by Method",
            x_label="Approximation Error",
            y_label="Method",
        )

    print(f"  Hessian:  {hessian_err:.6f}")
    print(f"  GGN:      {ggn_err:.6f}")
    print(f"  B-GGN:    {block_ggn_err:.6f}")
    print(f"  EK-FAC:   {ekfac_err:.6f}")
    print(f"  K-FAC:    {kfac_err:.6f}")


if __name__ == "__main__":
    main()
