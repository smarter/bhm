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


def _backprop_fim_preact_grads(
    model: MLP,
    logits: Float[Array, "N C"],
    pre_acts: list[Float[Array, "N _out"]],
) -> list[Float[Array, "N C _out"]]:
    """Compute per-sample, per-class pre-activation gradients for the real FIM.

    For each sample i and class c, computes Ds_l^(i,c) = d(-log p_c)/ds_l,
    i.e. the backprop gradient with pseudo-label y=c. This gives C gradient
    vectors per sample, which are combined with weights p_{i,c} by the caller.

    Returns a list (one per layer) of arrays with shape (N, C, out_l).
    """
    C = logits.shape[1]
    p = jax.nn.softmax(logits, axis=-1)  # (N, C)

    # Output layer gradient for pseudo-label c: p - e_c
    # For all classes at once: delta[i, c, :] = p[i, :] - I[c, :]
    # Shape: (N, C, C)
    identity_C = jnp.eye(C)  # (C, C)
    delta = p[:, None, :] - identity_C[None, :, :]  # (N, C, C)

    ds_list = [delta]

    # Backprop through hidden layers (reverse order)
    for l in range(len(model.layers) - 2, -1, -1):
        W_next = model.layers[l + 1].weight  # (out_next, in_next)
        # delta: (N, C, out_next) @ (out_next, in_next) -> (N, C, in_next)
        delta = delta @ W_next
        # tanh derivative
        a = jnp.tanh(pre_acts[l])  # (N, out_l)
        delta = delta * (1 - a**2)[:, None, :]  # broadcast (N, 1, out_l)
        ds_list.append(delta)

    ds_list.reverse()
    return ds_list


def compute_kfac(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """K-FAC: Kronecker-factored approximate curvature using the real FIM.

    Uses the true Fisher Information Matrix (sampling y from the model's
    predictive distribution) rather than the empirical Fisher. For each
    layer l, approximates the FIM block as S_l ⊗ A_{l-1} where:
      A_{l-1} = (1/N) Σ_i a_{l-1,i} a_{l-1,i}^T
      S_l     = (1/N) Σ_i Σ_c p_{i,c} · Ds_l^(i,c) · Ds_l^(i,c)^T

    For cross-entropy with softmax, the real FIM equals the GGN, so this
    stays well-conditioned even at convergence.
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]

    logits, activations, pre_acts = _forward_with_activations(model, x)
    p = jax.nn.softmax(logits, axis=-1)  # (N, C)
    ds_list = _backprop_fim_preact_grads(model, logits, pre_acts)

    ranges = _get_layer_param_ranges(model)
    K = jnp.zeros((D, D))

    for l, (start, end) in enumerate(ranges):
        a = activations[l]  # (N, in_l)
        ds = ds_list[l]  # (N, C, out_l)

        A = (a.T @ a) / N  # (in_l, in_l)

        # S = (1/N) Σ_i Σ_c p_{i,c} ds[i,c] ds[i,c]^T
        # Weight each class's gradient by its probability
        weighted_ds = ds * jnp.sqrt(p)[:, :, None]  # (N, C, out_l)
        wd_flat = weighted_ds.reshape(-1, ds.shape[2])  # (N*C, out_l)
        S = (wd_flat.T @ wd_flat) / N  # (out_l, out_l)

        K = K.at[start:end, start:end].set(jnp.kron(S, A))

    return (K + K.T) / 2


def compute_ekfac(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """EK-FAC: Eigenvalue-corrected K-FAC using the real FIM.

    Keeps K-FAC's Kronecker eigenbasis U = U_S ⊗ U_A but replaces the
    eigenvalues with corrected values computed from the real FIM:
      s_k* = (1/N) Σ_i Σ_c p_{i,c} · (U_S^T Ds^(i,c))_j^2 · (U_A^T a_i)_m^2
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]

    logits, activations, pre_acts = _forward_with_activations(model, x)
    p = jax.nn.softmax(logits, axis=-1)  # (N, C)
    ds_list = _backprop_fim_preact_grads(model, logits, pre_acts)

    ranges = _get_layer_param_ranges(model)
    E = jnp.zeros((D, D))

    for l, (start, end) in enumerate(ranges):
        a = activations[l]  # (N, in_l)
        ds = ds_list[l]  # (N, C, out_l)

        A = (a.T @ a) / N
        S_weighted = jnp.einsum("nc,ncd,nce->de", p, ds, ds) / N
        S = S_weighted

        # Eigenbases of the K-FAC factors (use numpy float64 for stability)
        _, U_A_np = np.linalg.eigh(np.asarray(A, dtype=np.float64))
        _, U_S_np = np.linalg.eigh(np.asarray(S, dtype=np.float64))
        U_A = jnp.array(U_A_np, dtype=jnp.float32)
        U_S = jnp.array(U_S_np, dtype=jnp.float32)

        # Rotated activations (same for all classes)
        q = a @ U_A  # (N, in_l)

        # Rotated gradients (per class)
        r = jnp.einsum("ncd,de->nce", ds, U_S)  # (N, C, out_l)

        # Corrected eigenvalues:
        #   s*[j, m] = (1/N) Σ_i Σ_c p_{i,c} * r[i,c,j]^2 * q[i,m]^2
        s_star = jnp.einsum("nc,ncj,nm->jm", p, r**2, q**2) / N  # (out_l, in_l)
        s_star_flat = s_star.ravel()

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
