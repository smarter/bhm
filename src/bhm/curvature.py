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


def _sub_forward(model: MLP, l: int, s_l: Float[Array, " out_l"]) -> Float[Array, " 10"]:
    """Forward from pre-activation at layer l to output logits."""
    if l == len(model.layers) - 1:
        return s_l
    h = jnp.tanh(s_l)
    for layer in model.layers[l + 1 : -1]:
        h = jnp.tanh(h @ layer.weight.T)
    return h @ model.layers[-1].weight.T


def _preact_jacobian(
    model: MLP, l: int, pre_acts_l: Float[Array, "N out_l"]
) -> Float[Array, "N C out_l"]:
    """Compute per-sample Jacobian ∂output/∂s_l via JAX autodiff."""
    return jax.vmap(jax.jacobian(lambda s: _sub_forward(model, l, s)))(pre_acts_l)


def compute_kfac(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """K-FAC: Kronecker-factored approximate curvature using the real FIM.

    Uses the true Fisher Information Matrix (= GGN for cross-entropy with
    softmax). For each layer l, the S factor is computed from the sub-network
    Jacobian B_l = ∂u/∂s_l via JAX autodiff:
      S_l = (1/N) Σ_i B_{l,i}^T H_out_i B_{l,i}
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]

    logits, activations, pre_acts = _forward_with_activations(model, x)
    H_out = jax.vmap(_ce_output_hessian)(logits)  # (N, C, C)

    ranges = _get_layer_param_ranges(model)
    K = jnp.zeros((D, D))

    for l, (start, end) in enumerate(ranges):
        a = activations[l]
        A = (a.T @ a) / N

        # Sub-network Jacobian B = ∂u/∂s_l: (N, C, out_l)
        B = _preact_jacobian(model, l, pre_acts[l])

        # S = (1/N) Σ B^T H_out B
        temp = jnp.einsum("nce,ned->ncd", H_out, B)
        S = jnp.einsum("ncd,ncf->df", B, temp) / N

        K = K.at[start:end, start:end].set(jnp.kron(S, A))

    return (K + K.T) / 2


def compute_ekfac(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """EK-FAC: Eigenvalue-corrected K-FAC using the real FIM.

    Keeps K-FAC's Kronecker eigenbasis U = U_S ⊗ U_A but replaces the
    eigenvalues with per-sample corrected values computed from the sub-network
    Jacobian B_l = ∂u/∂s_l via JAX autodiff.
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]
    C = 10

    logits, activations, pre_acts = _forward_with_activations(model, x)
    p = jax.nn.softmax(logits, axis=-1)  # (N, C)

    ranges = _get_layer_param_ranges(model)
    E = jnp.zeros((D, D))

    for l, (start, end) in enumerate(ranges):
        a = activations[l]
        A = (a.T @ a) / N

        # Sub-network Jacobian B = ∂u/∂s_l: (N, C, out_l)
        B = _preact_jacobian(model, l, pre_acts[l])

        # S = (1/N) Σ B^T H_out B (same as K-FAC)
        H_out = jax.vmap(_ce_output_hessian)(logits)
        temp = jnp.einsum("nce,ned->ncd", H_out, B)
        S = jnp.einsum("ncd,ncf->df", B, temp) / N

        # Eigenbases of the K-FAC factors (numpy float64 for stability)
        _, U_A_np = np.linalg.eigh(np.asarray(A, dtype=np.float64))
        _, U_S_np = np.linalg.eigh(np.asarray(S, dtype=np.float64))
        U_A = jnp.array(U_A_np, dtype=jnp.float32)
        U_S = jnp.array(U_S_np, dtype=jnp.float32)

        # Per-sample, per-class pre-activation gradients from B:
        # Ds[i,c,:] = B[i]^T @ (p[i] - e_c) = Σ_k B[i,k,:] * (p[i,k] - δ_{ck})
        delta = p[:, None, :] - jnp.eye(C)[None, :, :]  # (N, C, C)
        Ds = jnp.einsum("nkd,nck->ncd", B, delta)  # (N, C, out_l)

        # Rotated activations and gradients
        q = a @ U_A  # (N, in_l)
        r = jnp.einsum("ncd,de->nce", Ds, U_S)  # (N, C, out_l)

        # Corrected eigenvalues
        s_star = jnp.einsum("nc,ncj,nm->jm", p, r**2, q**2) / N
        s_star_flat = s_star.ravel()

        U = jnp.kron(U_S, U_A)
        block_ekfac = U @ jnp.diag(s_star_flat) @ U.T

        E = E.at[start:end, start:end].set(block_ekfac)

    return (E + E.T) / 2


def _shampoo_factors(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> tuple[
    Float[Array, "N 10"],
    list[Float[Array, "N _in"]],
    list[Float[Array, "N _out"]],
    Float[Array, "N C C"],
    list[tuple[Float[Array, "din din"], Float[Array, "dout dout"], Float[Array, ""]]],
]:
    """Compute per-layer Shampoo factors (shared by Shampoo, EShampoo, TKFAC, ETKFAC).

    Returns (logits, activations, pre_acts, H_out, layer_factors) where each
    layer_factors entry is (A_tilde, S_tilde, delta_l):
      A_tilde = (1/N) Σ w_i a_i a_i^T  (reweighted by gradient norm)
      S_tilde = (1/N) Σ ||a_i||^2 S_i  (reweighted by activation norm)
      delta_l = (1/N) Σ ||a_i||^2 w_i   (trace-matching scalar)
    """
    N = x.shape[0]

    logits, activations, pre_acts = _forward_with_activations(model, x)
    H_out = jax.vmap(_ce_output_hessian)(logits)  # (N, C, C)

    ranges = _get_layer_param_ranges(model)
    layer_factors = []

    for l, (start, end) in enumerate(ranges):
        a = activations[l]  # (N, d_in)
        B = _preact_jacobian(model, l, pre_acts[l])  # (N, C, d_out)

        # Per-sample S_i = B_i^T H_out_i B_i: (N, d_out, d_out)
        temp = jnp.einsum("nce,ned->ncd", H_out, B)  # (N, C, d_out)
        S_per = jnp.einsum("ncd,ncf->ndf", B, temp)  # (N, d_out, d_out)

        # w_i = tr(S_i) = expected squared gradient norm per sample
        w = jnp.trace(S_per, axis1=-2, axis2=-1)  # (N,)

        # A_tilde = (1/N) Σ w_i a_i a_i^T  (reweighted activation covariance)
        A_tilde = jnp.einsum("n,ni,nj->ij", w, a, a) / N

        # S_tilde = (1/N) Σ ||a_i||^2 S_i  (reweighted gradient covariance)
        a_norm_sq = jnp.sum(a**2, axis=1)  # (N,)
        S_tilde = jnp.einsum("n,ndf->df", a_norm_sq, S_per) / N

        # delta = E[||a||^2 * ||g||^2] = (1/N) Σ ||a_i||^2 * w_i
        delta_l = jnp.sum(a_norm_sq * w) / N

        layer_factors.append((A_tilde, S_tilde, delta_l))

    return logits, activations, pre_acts, H_out, layer_factors


def compute_shampoo(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """Shampoo: Kronecker-factored curvature with per-sample reweighting.

    Uses Gram matrices of the per-sample gradient matrix. The activation
    factor is trace-normalised. Eigenvalues are outer products of marginal
    spectra (like K-FAC).
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]

    _, _, _, _, layer_factors = _shampoo_factors(model, x, y)

    ranges = _get_layer_param_ranges(model)
    Sh = jnp.zeros((D, D))

    for (start, end), (A_tilde, S_tilde, _) in zip(ranges, layer_factors):
        # Trace-normalise A (matching reference implementation)
        A_norm = A_tilde / jnp.trace(A_tilde)
        Sh = Sh.at[start:end, start:end].set(jnp.kron(S_tilde, A_norm))

    return (Sh + Sh.T) / 2


def compute_eshampoo(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """EShampoo: Eigenvalue-corrected Shampoo using the real FIM.

    Combines Shampoo's reweighted factors with EK-FAC-style eigenvalue
    corrections. The activation factor is trace-normalised before
    eigendecomposition.
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]
    C = 10

    logits, activations, pre_acts, _, layer_factors = _shampoo_factors(model, x, y)
    p = jax.nn.softmax(logits, axis=-1)

    ranges = _get_layer_param_ranges(model)
    ES = jnp.zeros((D, D))

    for l, ((start, end), (A_tilde, S_tilde, _)) in enumerate(zip(ranges, layer_factors)):
        a = activations[l]
        B = _preact_jacobian(model, l, pre_acts[l])

        # Trace-normalise A
        A_norm = A_tilde / jnp.trace(A_tilde)

        # Eigenbases (numpy float64 for stability)
        _, U_A_np = np.linalg.eigh(np.asarray(A_norm, dtype=np.float64))
        _, U_S_np = np.linalg.eigh(np.asarray(S_tilde, dtype=np.float64))
        U_A = jnp.array(U_A_np, dtype=jnp.float32)
        U_S = jnp.array(U_S_np, dtype=jnp.float32)

        # Eigenvalue corrections (same formula as EK-FAC)
        delta = p[:, None, :] - jnp.eye(C)[None, :, :]
        Ds = jnp.einsum("nkd,nck->ncd", B, delta)
        q = a @ U_A
        r = jnp.einsum("ncd,de->nce", Ds, U_S)
        s_star = jnp.einsum("nc,ncj,nm->jm", p, r**2, q**2) / N
        s_star_flat = s_star.ravel()

        U = jnp.kron(U_S, U_A)
        block = U @ jnp.diag(s_star_flat) @ U.T
        ES = ES.at[start:end, start:end].set(block)

    return (ES + ES.T) / 2


def compute_tkfac(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """TKFAC: Trace-restricted K-FAC (Gao & Liu 2020).

    Decomposes each Fisher block as F_l = delta_l * Phi_l ⊗ Psi_l where:
      Phi_l = E[||g||^2 a a^T] / E[||a||^2 ||g||^2]  (unit-trace)
      Psi_l = E[||a||^2 g g^T] / E[||a||^2 ||g||^2]  (unit-trace)
      delta_l = E[||a||^2 ||g||^2]  (trace-matching scalar)
    This matches the trace of the true Fisher block.
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]

    _, _, _, _, layer_factors = _shampoo_factors(model, x, y)

    ranges = _get_layer_param_ranges(model)
    T = jnp.zeros((D, D))

    for (start, end), (A_tilde, S_tilde, delta_l) in zip(ranges, layer_factors):
        # Normalise factors to unit trace (Eq. 4.9 from Gao & Liu 2020)
        Phi = A_tilde / jnp.trace(A_tilde)
        Psi = S_tilde / jnp.trace(S_tilde)

        T = T.at[start:end, start:end].set(delta_l * jnp.kron(Psi, Phi))

    return (T + T.T) / 2


def compute_etkfac(
    model: MLP,
    x: Float[Array, "N 64"],
    y: Int[Array, " N"],
) -> Float[Array, "D D"]:
    """ETKFAC: Eigenvalue-corrected TKFAC using the real FIM.

    Uses TKFAC's per-sample reweighted eigenbasis (from Shampoo factors) with
    EK-FAC-style eigenvalue corrections. The eigenbasis differs from standard
    K-FAC/EK-FAC because the factors are reweighted.
    """
    flat_params, _ = jax.flatten_util.ravel_pytree(model)
    D = flat_params.shape[0]
    N = x.shape[0]
    C = 10

    logits, activations, pre_acts, _, layer_factors = _shampoo_factors(model, x, y)
    p = jax.nn.softmax(logits, axis=-1)

    ranges = _get_layer_param_ranges(model)
    ET = jnp.zeros((D, D))

    for l, ((start, end), (A_tilde, S_tilde, _)) in enumerate(zip(ranges, layer_factors)):
        a = activations[l]
        B = _preact_jacobian(model, l, pre_acts[l])

        # Eigenbases from reweighted factors (numpy float64 for stability)
        _, U_A_np = np.linalg.eigh(np.asarray(A_tilde, dtype=np.float64))
        _, U_S_np = np.linalg.eigh(np.asarray(S_tilde, dtype=np.float64))
        U_A = jnp.array(U_A_np, dtype=jnp.float32)
        U_S = jnp.array(U_S_np, dtype=jnp.float32)

        # Eigenvalue corrections in the reweighted eigenbasis
        delta = p[:, None, :] - jnp.eye(C)[None, :, :]
        Ds = jnp.einsum("nkd,nck->ncd", B, delta)
        q = a @ U_A
        r = jnp.einsum("ncd,de->nce", Ds, U_S)
        s_star = jnp.einsum("nc,ncj,nm->jm", p, r**2, q**2) / N
        s_star_flat = s_star.ravel()

        U = jnp.kron(U_S, U_A)
        block = U @ jnp.diag(s_star_flat) @ U.T
        ET = ET.at[start:end, start:end].set(block)

    return (ET + ET.T) / 2


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
