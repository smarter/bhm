import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from bhm.model import MLP
from bhm.train import cross_entropy_loss


def _get_layer_param_ranges(model: MLP) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each layer's parameters in the flat vector."""
    ranges = []
    offset = 0
    for layer in model.layers:
        size = layer.weight.size
        ranges.append((offset, offset + size))
        offset += size
    return ranges


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


def _forward_with_activations(
    model: MLP, x: Float[Array, "N 64"]
) -> tuple[
    Float[Array, "N 10"],
    list[Float[Array, "N _in"]],
    list[Float[Array, "N _out"]],
]:
    """Forward pass recording layer inputs and pre-activations per layer."""
    activations: list[Float[Array, "N _in"]] = []
    pre_acts: list[Float[Array, "N _out"]] = []

    h = x
    for layer in model.layers[:-1]:
        activations.append(h)
        s = h @ layer.weight.T
        pre_acts.append(s)
        h = jnp.tanh(s)

    last = model.layers[-1]
    activations.append(h)
    s = h @ last.weight.T
    pre_acts.append(s)

    return s, activations, pre_acts


def _backprop_preact_grads(
    model: MLP,
    logits: Float[Array, "N 10"],
    y: Int[Array, " N"],
    pre_acts: list[Float[Array, "N _out"]],
) -> list[Float[Array, "N _out"]]:
    """Compute per-sample pre-activation gradients dL/ds_l for each layer via backprop."""
    p = jax.nn.softmax(logits, axis=-1)
    y_oh = jax.nn.one_hot(y, 10)

    # Output layer gradient: dL/d(logits) = softmax - one_hot
    delta = p - y_oh
    ds_list = [delta]

    # Backprop through hidden layers (reverse order)
    for l in range(len(model.layers) - 2, -1, -1):
        W_next = model.layers[l + 1].weight  # (out_next, in_next)
        delta = delta @ W_next  # (N, in_next = width)
        # tanh'(s) = 1 - tanh(s)^2 = 1 - a^2
        a = jnp.tanh(pre_acts[l])
        delta = delta * (1 - a**2)
        ds_list.append(delta)

    ds_list.reverse()
    return ds_list


def compute_kfac(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """K-FAC: Kronecker-factored approximate curvature.

    For each layer l, approximates the GGN/Fisher block as A_{l-1} ⊗ S_l where:
      A_{l-1} = (1/N) Σ a_{l-1} a_{l-1}^T  (input covariance)
      S_l     = (1/N) Σ Ds_l Ds_l^T          (pre-activation gradient covariance)

    With weight shape (out, in) and C-order flattening, the Kronecker product
    is S ⊗ A (gradient factor first, then activation factor).
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]

    logits, activations, pre_acts = _forward_with_activations(model, x)
    ds_list = _backprop_preact_grads(model, logits, y, pre_acts)

    ranges = _get_layer_param_ranges(model)
    K = jnp.zeros((D, D))

    for l, (start, end) in enumerate(ranges):
        a = activations[l]  # (N, in_l)
        ds = ds_list[l]  # (N, out_l)

        A = (a.T @ a) / N  # (in_l, in_l)
        S = (ds.T @ ds) / N  # (out_l, out_l)

        # Kronecker product: S ⊗ A (matches C-order vec of (out, in) weight)
        K = K.at[start:end, start:end].set(jnp.kron(S, A))

    return (K + K.T) / 2


def compute_ekfac(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """EK-FAC: Eigenvalue-corrected K-FAC.

    Keeps K-FAC's Kronecker eigenbasis U = U_S ⊗ U_A but replaces the
    eigenvalues with corrected values s_k* = E[(U^T g_l)_k^2], matching
    the diagonal of the true Fisher/GGN in the Kronecker eigenbasis.
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]

    logits, activations, pre_acts = _forward_with_activations(model, x)
    ds_list = _backprop_preact_grads(model, logits, y, pre_acts)

    ranges = _get_layer_param_ranges(model)
    E = jnp.zeros((D, D))

    for l, (start, end) in enumerate(ranges):
        a = activations[l]  # (N, in_l)
        ds = ds_list[l]  # (N, out_l)

        A = (a.T @ a) / N  # (in_l, in_l)
        S = (ds.T @ ds) / N  # (out_l, out_l)

        # Eigenbases of the K-FAC factors (use numpy float64 for stability)
        _, U_A_np = np.linalg.eigh(np.asarray(A, dtype=np.float64))
        _, U_S_np = np.linalg.eigh(np.asarray(S, dtype=np.float64))
        U_A = jnp.array(U_A_np, dtype=jnp.float32)
        U_S = jnp.array(U_S_np, dtype=jnp.float32)

        # Rotate per-sample activations and gradients into the eigenbasis
        q = a @ U_A  # (N, in_l)
        r = ds @ U_S  # (N, out_l)

        # Corrected eigenvalues: s*[j, m] = (1/N) sum_i r[i,j]^2 * q[i,m]^2
        s_star = (r**2).T @ (q**2) / N  # (out_l, in_l)
        s_star_flat = s_star.ravel()  # C-order

        # EK-FAC block = U diag(s*) U^T where U = U_S ⊗ U_A
        U = jnp.kron(U_S, U_A)
        block_ekfac = U @ jnp.diag(s_star_flat) @ U.T

        E = E.at[start:end, start:end].set(block_ekfac)

    return (E + E.T) / 2


def compute_block_ggn(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
    *,
    chunk_size: int = 32,
) -> Float[Array, "D D"]:
    """Block-diagonal GGN: extract only the diagonal blocks (one per layer) from the full GGN."""
    G = compute_ggn(model, x, y, chunk_size=chunk_size)
    ranges = _get_layer_param_ranges(model)
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    B = jnp.zeros((D, D))
    for start, end in ranges:
        B = B.at[start:end, start:end].set(G[start:end, start:end])
    return B
