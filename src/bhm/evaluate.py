import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from bhm.model import MLP


def pseudo_inverse(
    M: Float[Array, "D D"], eps: float = 1e-4
) -> Float[Array, "D D"]:
    # Use numpy float64 eigendecomposition for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(M, dtype=np.float64))
    inv_eigenvalues = np.where(np.abs(eigenvalues) > eps, 1.0 / eigenvalues, 0.0)
    result = eigenvectors @ np.diag(inv_eigenvalues) @ eigenvectors.T
    return jnp.array(result, dtype=jnp.float32)


def per_sample_gradients(
    model: MLP, x: Float[Array, "N 64"], y: Int[Array, " N"]
) -> Float[Array, "N D"]:
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(model)

    def single_loss(params: Float[Array, " D"], xi: Float[Array, " 64"], yi: Int[Array, ""]) -> Float[Array, ""]:
        m = unravel_fn(params)
        logits = m(xi)
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.sum(jax.nn.one_hot(yi, 10) * log_probs)

    grad_fn = jax.grad(single_loss)
    return jax.vmap(grad_fn, in_axes=(None, 0, 0))(flat_params, x, y)


def approximation_error(
    H: Float[Array, "D D"],
    H_hat_inv: Float[Array, "D D"],
    grads: Float[Array, "N D"],
) -> float:
    # M = H @ H_hat^{-1}; we want ||M v_i - v_i||^2 / ||v_i||^2
    M = H @ H_hat_inv  # (D, D)
    # grads[i] is v_i^T, so grads @ M.T gives rows (M v_i)^T
    transformed = grads @ M.T  # (N, D)
    residuals = transformed - grads  # (N, D)
    norms_sq = jnp.sum(residuals**2, axis=1)
    v_norms_sq = jnp.sum(grads**2, axis=1)
    return float(jnp.mean(norms_sq / v_norms_sq))
