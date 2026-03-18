import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from bhm.model import MLP
from bhm.train import cross_entropy_loss


def compute_hessian(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
    *,
    chunk_size: int = 32,
) -> Float[Array, "D D"]:
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]

    def flat_loss(params: Float[Array, " D"]) -> Float[Array, ""]:
        m = unravel_fn(params)
        return cross_entropy_loss(m, x, y)

    grad_fn = jax.grad(flat_loss)

    @jax.jit
    def hvp_batch(basis: Float[Array, "K D"]) -> Float[Array, "K D"]:
        def single_hvp(v: Float[Array, " D"]) -> Float[Array, " D"]:
            _, tangent = jax.jvp(grad_fn, (flat_params,), (v,))
            return tangent

        return jax.vmap(single_hvp)(basis)

    identity = jnp.eye(D)
    H = jnp.zeros((D, D))

    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        H = H.at[start:end].set(hvp_batch(identity[start:end]))

    return (H + H.T) / 2


def _ce_output_hessian(logits: Float[Array, " C"]) -> Float[Array, "C C"]:
    p = jax.nn.softmax(logits)
    return jnp.diag(p) - jnp.outer(p, p)


def compute_ggn(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
    *,
    chunk_size: int = 32,
) -> Float[Array, "D D"]:
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]

    def flat_forward(params: Float[Array, " D"], xi: Float[Array, " 64"]) -> Float[Array, " 10"]:
        m = unravel_fn(params)
        return m(xi)

    jacobian_fn = jax.jacobian(flat_forward)

    N = x.shape[0]
    G = jnp.zeros((D, D))

    for start in range(0, N, chunk_size):
        x_chunk = x[start : start + chunk_size]

        # Per-sample Jacobians: (chunk, C, D)
        J = jax.vmap(jacobian_fn, in_axes=(None, 0))(flat_params, x_chunk)

        # Per-sample logits for output Hessian
        logits = jax.vmap(flat_forward, in_axes=(None, 0))(flat_params, x_chunk)
        H_out = jax.vmap(_ce_output_hessian)(logits)  # (chunk, C, C)

        # G += sum_i J_i^T @ H_out_i @ J_i
        temp = jnp.einsum("nce,ned->ncd", H_out, J)  # (chunk, C, D)
        G = G + jnp.einsum("ncd,ncf->df", J, temp)  # (D, D)

    G = G / N
    return (G + G.T) / 2
