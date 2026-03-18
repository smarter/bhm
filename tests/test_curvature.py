import jax
import jax.flatten_util
import jax.numpy as jnp

from bhm.curvature import compute_block_ggn, compute_ggn, compute_hessian
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
    # Diagonal blocks should match GGN
    from bhm.curvature import _get_layer_param_ranges

    ranges = _get_layer_param_ranges(model)
    for start, end in ranges:
        assert jnp.allclose(
            BG[start:end, start:end], G[start:end, start:end], atol=1e-5
        ), f"Block [{start}:{end}] doesn't match GGN diagonal"
    # Off-diagonal blocks should be zero
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


def test_pseudo_inverse():
    # Known matrix: diagonal
    M = jnp.diag(jnp.array([2.0, 0.5, 1e-6, 3.0]))
    M_inv = pseudo_inverse(M, eps=1e-4)
    expected = jnp.diag(jnp.array([0.5, 2.0, 0.0, 1.0 / 3.0]))
    assert jnp.allclose(M_inv, expected, atol=1e-5), f"Pseudo-inverse incorrect"
