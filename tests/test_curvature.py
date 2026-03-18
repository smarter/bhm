import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

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
from bhm.evaluate import approximation_error, per_sample_gradients, pseudo_inverse
from bhm.model import MLP


def _make_tiny_setup():
    key = jax.random.PRNGKey(42)
    model = MLP(depth=1, width=4, key=key)
    # Small synthetic data: 16 samples, 64 features, 10 classes
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 64))
    y = jax.random.randint(jax.random.PRNGKey(1), (16,), 0, 10)
    return model, x, y


def test_hessian_symmetric():
    model, x, y = _make_tiny_setup()
    H = compute_hessian(model, x, y)
    assert jnp.allclose(H, H.T, atol=1e-5), f"Hessian not symmetric: max diff {jnp.max(jnp.abs(H - H.T))}"


def test_ggn_symmetric_psd():
    model, x, y = _make_tiny_setup()
    G = compute_ggn(model, x, y, chunk_size=8)
    assert jnp.allclose(G, G.T, atol=1e-5), f"GGN not symmetric: max diff {jnp.max(jnp.abs(G - G.T))}"
    eigenvalues = jnp.linalg.eigvalsh(G)
    assert jnp.all(eigenvalues >= -1e-5), f"GGN not PSD: min eigenvalue {jnp.min(eigenvalues)}"


def test_hessian_approx_error_near_zero():
    model, x, y = _make_tiny_setup()
    H = compute_hessian(model, x, y)
    H_inv = pseudo_inverse(H)
    grads = per_sample_gradients(model, x, y)
    err = approximation_error(H, H_inv, grads)
    assert err < 1e-3, f"Hessian self-approx error too large: {err}"


def test_hessian_differs_from_ggn():
    model, x, y = _make_tiny_setup()
    H = compute_hessian(model, x, y)
    G = compute_ggn(model, x, y, chunk_size=8)
    diff = jnp.max(jnp.abs(H - G))
    assert diff > 1e-6, f"Hessian and GGN are identical (diff={diff}), expected them to differ"


def test_block_ggn_is_block_diagonal():
    model, x, y = _make_tiny_setup()
    G = compute_ggn(model, x, y, chunk_size=8)
    BG = compute_block_ggn(model, x, y, chunk_size=8)
    from bhm.curvature import _get_layer_param_ranges

    ranges = _get_layer_param_ranges(model)
    for start, end in ranges:
        assert jnp.allclose(
            BG[start:end, start:end], G[start:end, start:end], atol=1e-5
        ), f"Block [{start}:{end}] doesn't match GGN diagonal"
    for i, (s1, e1) in enumerate(ranges):
        for j, (s2, e2) in enumerate(ranges):
            if i != j:
                assert jnp.allclose(
                    BG[s1:e1, s2:e2], 0.0, atol=1e-10
                ), f"Off-diagonal block [{s1}:{e1},{s2}:{e2}] not zero"


def test_block_ggn_error_between_ggn_and_kfac():
    """B-GGN error should be >= GGN error (it's a worse approximation)."""
    model, x, y = _make_tiny_setup()
    H = compute_hessian(model, x, y)
    G = compute_ggn(model, x, y, chunk_size=8)
    BG = compute_block_ggn(model, x, y, chunk_size=8)
    grads = per_sample_gradients(model, x, y)
    G_inv = pseudo_inverse(G)
    BG_inv = pseudo_inverse(BG)
    ggn_err = approximation_error(H, G_inv, grads)
    bg_err = approximation_error(H, BG_inv, grads)
    assert bg_err >= ggn_err - 1e-4, f"B-GGN error ({bg_err}) < GGN error ({ggn_err})"


def test_kfac_symmetric_psd():
    model, x, y = _make_tiny_setup()
    K = compute_kfac(model, x, y)
    assert jnp.allclose(K, K.T, atol=1e-5), f"K-FAC not symmetric"
    eigenvalues = jnp.linalg.eigvalsh(K)
    assert jnp.all(eigenvalues >= -1e-5), f"K-FAC not PSD: min eigenvalue {jnp.min(eigenvalues)}"


def test_kfac_is_block_diagonal():
    """K-FAC should only have nonzero entries in per-layer diagonal blocks."""
    model, x, y = _make_tiny_setup()
    K = compute_kfac(model, x, y)
    from bhm.curvature import _get_layer_param_ranges

    ranges = _get_layer_param_ranges(model)
    for i, (s1, e1) in enumerate(ranges):
        for j, (s2, e2) in enumerate(ranges):
            if i != j:
                assert jnp.allclose(
                    K[s1:e1, s2:e2], 0.0, atol=1e-10
                ), f"K-FAC off-diagonal block [{s1}:{e1},{s2}:{e2}] not zero"


def test_kfac_error_ge_block_ggn():
    """K-FAC error should be >= B-GGN error when measured against the GGN.

    Uses the GGN (not Hessian) as reference to isolate the Kronecker
    factorization error from the Hessian-GGN residual.
    """
    key = jax.random.PRNGKey(7)
    model = MLP(depth=3, width=8, key=key)
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 64))
    y = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 10)

    G = compute_ggn(model, x, y, chunk_size=16)
    BG = compute_block_ggn(model, x, y, chunk_size=16)
    K = compute_kfac(model, x, y)
    grads = per_sample_gradients(model, x, y)
    BG_inv = pseudo_inverse(BG)
    K_inv = pseudo_inverse(K)
    bg_err = approximation_error(G, BG_inv, grads)
    kfac_err = approximation_error(G, K_inv, grads)
    assert kfac_err >= bg_err - 1e-3, f"K-FAC error ({kfac_err}) < B-GGN error ({bg_err})"


def test_ekfac_symmetric_psd():
    model, x, y = _make_tiny_setup()
    E = compute_ekfac(model, x, y)
    assert jnp.allclose(E, E.T, atol=1e-5), f"EK-FAC not symmetric"
    eigenvalues = jnp.linalg.eigvalsh(E)
    assert jnp.all(eigenvalues >= -1e-5), f"EK-FAC not PSD: min eigenvalue {jnp.min(eigenvalues)}"


def test_ekfac_is_block_diagonal():
    """EK-FAC should only have nonzero entries in per-layer diagonal blocks."""
    model, x, y = _make_tiny_setup()
    E = compute_ekfac(model, x, y)
    from bhm.curvature import _get_layer_param_ranges

    ranges = _get_layer_param_ranges(model)
    for i, (s1, e1) in enumerate(ranges):
        for j, (s2, e2) in enumerate(ranges):
            if i != j:
                assert jnp.allclose(
                    E[s1:e1, s2:e2], 0.0, atol=1e-10
                ), f"EK-FAC off-diagonal block [{s1}:{e1},{s2}:{e2}] not zero"


def test_ekfac_reconstructs_from_factors():
    """Verify EK-FAC block equals U diag(s*) U^T computed independently."""
    model, x, y = _make_tiny_setup()
    E = compute_ekfac(model, x, y)
    from bhm.curvature import (
        _ce_output_hessian,
        _forward_with_activations,
        _get_layer_param_ranges,
        _preact_jacobian,
    )

    logits, activations, pre_acts = _forward_with_activations(model, x)
    p = jax.nn.softmax(logits, axis=-1)
    C = p.shape[1]
    ranges = _get_layer_param_ranges(model)
    N = x.shape[0]

    for l, (start, end) in enumerate(ranges):
        a = activations[l]
        B = _preact_jacobian(model, l, pre_acts[l])
        H_out = jax.vmap(_ce_output_hessian)(logits)

        A = (a.T @ a) / N
        temp = jnp.einsum("nce,ned->ncd", H_out, B)
        S = jnp.einsum("ncd,ncf->df", B, temp) / N

        _, U_A_np = np.linalg.eigh(np.asarray(A, dtype=np.float64))
        _, U_S_np = np.linalg.eigh(np.asarray(S, dtype=np.float64))
        U_A = jnp.array(U_A_np, dtype=jnp.float32)
        U_S = jnp.array(U_S_np, dtype=jnp.float32)

        delta = p[:, None, :] - jnp.eye(C)[None, :, :]
        Ds = jnp.einsum("nkd,nck->ncd", B, delta)
        q = a @ U_A
        r = jnp.einsum("ncd,de->nce", Ds, U_S)
        s_star = (jnp.einsum("nc,ncj,nm->jm", p, r**2, q**2) / N).ravel()

        U = jnp.kron(U_S, U_A)
        expected = U @ jnp.diag(s_star) @ U.T

        E_block = E[start:end, start:end]
        assert jnp.allclose(E_block, expected, atol=1e-4), (
            f"Layer {l}: EK-FAC block doesn't match reconstruction, "
            f"max diff = {jnp.max(jnp.abs(E_block - expected))}"
        )


def test_shampoo_a_factor_unit_trace():
    """Shampoo's A factor should be trace-normalised per layer."""
    model, x, y = _make_tiny_setup()
    from bhm.curvature import _get_layer_param_ranges, _shampoo_factors

    _, _, _, _, layer_factors = _shampoo_factors(model, x, y)
    S = compute_shampoo(model, x, y)
    ranges = _get_layer_param_ranges(model)

    for (start, end), (A_tilde, S_tilde, _) in zip(ranges, layer_factors):
        block = S[start:end, start:end]
        d_in = A_tilde.shape[0]
        d_out = S_tilde.shape[0]
        # Shampoo block = S_tilde ⊗ (A_tilde / tr(A_tilde))
        # Reconstruct A factor from the block: for Kronecker S⊗A, the (0,0)
        # d_in×d_in subblock is S[0,0]*A. So A ∝ block[:d_in, :d_in].
        A_recovered = block[:d_in, :d_in]
        # If A is trace-normalised, tr(A_recovered) should equal S_tilde[0,0]
        # (since the subblock is S[0,0]*A and tr(A)=1).
        # Simpler: just check that S_tilde ⊗ (A_tilde/tr(A_tilde)) == block.
        A_norm = A_tilde / jnp.trace(A_tilde)
        expected = jnp.kron(S_tilde, A_norm)
        assert jnp.allclose(block, expected, atol=1e-4), (
            f"Shampoo block doesn't match S_tilde ⊗ (A_tilde/tr(A_tilde))"
        )


def test_tkfac_trace_matches_ggn_block():
    """TKFAC's defining property: tr(TKFAC block) == tr(true GGN block) per layer."""
    model, x, y = _make_tiny_setup()
    G = compute_ggn(model, x, y, chunk_size=8)
    T = compute_tkfac(model, x, y)
    from bhm.curvature import _get_layer_param_ranges

    ranges = _get_layer_param_ranges(model)
    for start, end in ranges:
        ggn_trace = jnp.trace(G[start:end, start:end])
        tkfac_trace = jnp.trace(T[start:end, start:end])
        assert jnp.allclose(tkfac_trace, ggn_trace, rtol=0.05), (
            f"Block [{start}:{end}]: TKFAC trace {tkfac_trace:.4f} != "
            f"GGN trace {ggn_trace:.4f}"
        )


def test_tkfac_reconstructs_from_shampoo_factors():
    """TKFAC should equal delta * (A_tilde/tr(A_tilde)) ⊗ (S_tilde/tr(S_tilde))."""
    model, x, y = _make_tiny_setup()
    T = compute_tkfac(model, x, y)
    from bhm.curvature import _get_layer_param_ranges, _shampoo_factors

    _, _, _, _, layer_factors = _shampoo_factors(model, x, y)
    ranges = _get_layer_param_ranges(model)

    for (start, end), (A_tilde, S_tilde, delta_l) in zip(ranges, layer_factors):
        Phi = A_tilde / jnp.trace(A_tilde)
        Psi = S_tilde / jnp.trace(S_tilde)
        expected = delta_l * jnp.kron(Psi, Phi)
        T_block = T[start:end, start:end]
        assert jnp.allclose(T_block, expected, atol=1e-4), (
            f"Block [{start}:{end}]: TKFAC doesn't match delta * Phi ⊗ Psi, "
            f"max diff = {jnp.max(jnp.abs(T_block - expected))}"
        )


def test_eshampoo_equals_etkfac():
    """EShampoo and ETKFAC should be identical.

    Both use eigenvalue corrections in the Shampoo eigenbasis.
    Shampoo normalises only A while TKFAC normalises both A and S,
    but scalar normalisation preserves eigenvectors, so the eigenbases
    are the same. The eigenvalue corrections depend only on the
    eigenbasis and per-sample data, so the results must match.
    """
    model, x, y = _make_tiny_setup()
    ES = compute_eshampoo(model, x, y)
    ET = compute_etkfac(model, x, y)
    assert jnp.allclose(ES, ET, atol=1e-4), (
        f"EShampoo != ETKFAC: max diff = {jnp.max(jnp.abs(ES - ET))}"
    )


def test_shampoo_and_tkfac_share_eigenvectors():
    """Shampoo and TKFAC should have the same eigenvectors per layer.

    Both use Shampoo factors (A_tilde, S_tilde). TKFAC normalises
    both to unit trace; Shampoo normalises only A. Since scalar
    normalisation preserves eigenvectors, the Kronecker eigenbases
    should be the same.
    """
    model, x, y = _make_tiny_setup()
    S = compute_shampoo(model, x, y)
    T = compute_tkfac(model, x, y)
    from bhm.curvature import _get_layer_param_ranges

    ranges = _get_layer_param_ranges(model)
    for start, end in ranges:
        S_block = S[start:end, start:end]
        T_block = T[start:end, start:end]
        # Both have form U diag(lambda) U^T with the same U.
        # If eigenvectors are the same, S_block and T_block commute.
        commutator = S_block @ T_block - T_block @ S_block
        assert jnp.allclose(commutator, 0.0, atol=1e-4), (
            f"Block [{start}:{end}]: Shampoo and TKFAC don't commute, "
            f"max |[S,T]| = {jnp.max(jnp.abs(commutator))}"
        )


def test_pseudo_inverse():
    # Known matrix: diagonal
    M = jnp.diag(jnp.array([2.0, 0.5, 1e-6, 3.0]))
    M_inv = pseudo_inverse(M, eps=1e-4)
    expected = jnp.diag(jnp.array([0.5, 2.0, 0.0, 1.0 / 3.0]))
    assert jnp.allclose(M_inv, expected, atol=1e-5), f"Pseudo-inverse incorrect"
