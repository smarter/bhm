import equinox as eqx
import jax
from dvclive.live import Live
from omegaconf import OmegaConf

from bhm.curvature import (
    compute_block_ggn,
    compute_ekfac,
    compute_eshampoo,
    compute_etkfac,
    compute_ggn,
    compute_hessian,
    compute_kfac,
    compute_shampoo,
    compute_tkfac,
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

    print("  Computing Shampoo...")
    SH = compute_shampoo(model, data.x_train, data.y_train)

    print("  Computing EShampoo...")
    ESH = compute_eshampoo(model, data.x_train, data.y_train)

    print("  Computing TKFAC...")
    TKF = compute_tkfac(model, data.x_train, data.y_train)

    print("  Computing ETKFAC...")
    ETKF = compute_etkfac(model, data.x_train, data.y_train)

    print("  Computing per-sample gradients...")
    grads = per_sample_gradients(model, data.x_train, data.y_train)

    print("  Computing pseudo-inverses...")
    H_inv = pseudo_inverse(H)
    G_inv = pseudo_inverse(G)
    BG_inv = pseudo_inverse(BG)
    KF_inv = pseudo_inverse(KF)
    EKF_inv = pseudo_inverse(EKF)
    SH_inv = pseudo_inverse(SH)
    ESH_inv = pseudo_inverse(ESH)
    TKF_inv = pseudo_inverse(TKF)
    ETKF_inv = pseudo_inverse(ETKF)

    print("  Computing approximation errors...")
    hessian_err = approximation_error(H, H_inv, grads)
    ggn_err = approximation_error(H, G_inv, grads)
    block_ggn_err = approximation_error(H, BG_inv, grads)
    kfac_err = approximation_error(H, KF_inv, grads)
    ekfac_err = approximation_error(H, EKF_inv, grads)
    shampoo_err = approximation_error(H, SH_inv, grads)
    eshampoo_err = approximation_error(H, ESH_inv, grads)
    tkfac_err = approximation_error(H, TKF_inv, grads)
    etkfac_err = approximation_error(H, ETKF_inv, grads)

    with Live(dir="dvclive/evaluate", save_dvc_exp=False, dvcyaml=False) as live:
        live.log_metric("hessian_error", hessian_err, plot=False)
        live.log_metric("ggn_error", ggn_err, plot=False)
        live.log_metric("block_ggn_error", block_ggn_err, plot=False)
        live.log_metric("ekfac_error", ekfac_err, plot=False)
        live.log_metric("kfac_error", kfac_err, plot=False)
        live.log_metric("shampoo_error", shampoo_err, plot=False)
        live.log_metric("eshampoo_error", eshampoo_err, plot=False)
        live.log_metric("tkfac_error", tkfac_err, plot=False)
        live.log_metric("etkfac_error", etkfac_err, plot=False)

        live.log_plot(
            "approx_error",
            [
                {"method": "Hessian", "approx_error": hessian_err},
                {"method": "GGN", "approx_error": ggn_err},
                {"method": "B-GGN", "approx_error": block_ggn_err},
                {"method": "EShampoo", "approx_error": eshampoo_err},
                {"method": "EK-FAC", "approx_error": ekfac_err},
                {"method": "ETKFAC", "approx_error": etkfac_err},
                {"method": "Shampoo", "approx_error": shampoo_err},
                {"method": "K-FAC", "approx_error": kfac_err},
                {"method": "TKFAC", "approx_error": tkfac_err},
            ],
            x="method",
            y="approx_error",
            template="bar_horizontal",
            title="Approximation Error by Method",
            x_label="Approximation Error",
            y_label="Method",
        )

    print(f"  Hessian:   {hessian_err:.6f}")
    print(f"  GGN:       {ggn_err:.6f}")
    print(f"  B-GGN:     {block_ggn_err:.6f}")
    print(f"  EShampoo:  {eshampoo_err:.6f}")
    print(f"  EK-FAC:    {ekfac_err:.6f}")
    print(f"  ETKFAC:    {etkfac_err:.6f}")
    print(f"  Shampoo:   {shampoo_err:.6f}")
    print(f"  K-FAC:     {kfac_err:.6f}")
    print(f"  TKFAC:     {tkfac_err:.6f}")


if __name__ == "__main__":
    main()
