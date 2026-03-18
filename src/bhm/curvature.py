import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from bhm.model import MLP
from bhm.train import cross_entropy_loss


def _get_layer_param_ranges(model: MLP) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each layer's parameters in the flat vector."""
    ranges = []
    offset = 0
    for layer in model.layers:
        assert layer.bias is not None
        size = layer.weight.size + layer.bias.size
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


def _kfac_perm(out_features: int, in_features: int) -> Float[Array, " D_layer"]:
    """Permutation from K-FAC augmented-weight ordering to ravel_pytree ordering.

    K-FAC operates on the augmented weight [W | b] of shape (out, in+1), vectorized
    row-by-row: [W[0,:], b[0], W[1,:], b[1], ...].

    ravel_pytree gives: [W.ravel(), b.ravel()] = [W[0,:], W[1,:], ..., b[0], b[1], ...].

    Returns a permutation array P such that kfac_vec[i] corresponds to ravel_vec[P[i]].
    """
    perm = []
    w_size = out_features * in_features
    for j in range(out_features):
        for k in range(in_features):
            perm.append(j * in_features + k)
        perm.append(w_size + j)
    return jnp.array(perm)


def _forward_with_activations(
    model: MLP, x: Float[Array, "N 64"]
) -> tuple[
    Float[Array, "N 10"],
    list[Float[Array, "N _in_plus_1"]],
    list[Float[Array, "N _out"]],
]:
    """Forward pass recording bias-augmented inputs and pre-activations per layer."""
    N = x.shape[0]
    a_bars: list[Float[Array, "N _in_plus_1"]] = []
    pre_acts: list[Float[Array, "N _out"]] = []

    h = x
    for layer in model.layers[:-1]:
        assert layer.bias is not None
        a_bar = jnp.concatenate([h, jnp.ones((N, 1))], axis=1)
        a_bars.append(a_bar)
        s = h @ layer.weight.T + layer.bias
        pre_acts.append(s)
        h = jnp.tanh(s)

    last = model.layers[-1]
    assert last.bias is not None
    a_bar = jnp.concatenate([h, jnp.ones((N, 1))], axis=1)
    a_bars.append(a_bar)
    s = h @ last.weight.T + last.bias
    pre_acts.append(s)

    return s, a_bars, pre_acts


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
      A_{l-1} = (1/N) Σ ā_{l-1} ā_{l-1}^T  (input covariance, bias-augmented)
      S_l     = (1/N) Σ Ds_l Ds_l^T          (pre-activation gradient covariance)
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]

    logits, a_bars, pre_acts = _forward_with_activations(model, x)
    ds_list = _backprop_preact_grads(model, logits, y, pre_acts)

    ranges = _get_layer_param_ranges(model)
    K = jnp.zeros((D, D))

    for l, (start, end) in enumerate(ranges):
        a_bar = a_bars[l]  # (N, in_l+1)
        ds = ds_list[l]  # (N, out_l)

        A = (a_bar.T @ a_bar) / N  # (in_l+1, in_l+1)
        S = (ds.T @ ds) / N  # (out_l, out_l)

        # Kronecker product in C-order (row-major) vectorization: S ⊗ A
        block_kfac = jnp.kron(S, A)  # (out*(in+1), out*(in+1))

        # Permute from augmented ordering to ravel_pytree ordering
        in_f = model.layers[l].weight.shape[1]
        out_f = model.layers[l].weight.shape[0]
        perm = _kfac_perm(out_f, in_f)
        inv_perm = jnp.argsort(perm)
        block_ravel = block_kfac[jnp.ix_(inv_perm, inv_perm)]

        K = K.at[start:end, start:end].set(block_ravel)

    return (K + K.T) / 2


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
