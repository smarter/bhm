import jax
import jax.flatten_util
import jax.numpy as jnp

from bhm.curvature import compute_ggn, compute_hessian
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


def test_pseudo_inverse():
    # Known matrix: diagonal
    M = jnp.diag(jnp.array([2.0, 0.5, 1e-6, 3.0]))
    M_inv = pseudo_inverse(M, eps=1e-4)
    expected = jnp.diag(jnp.array([0.5, 2.0, 0.0, 1.0 / 3.0]))
    assert jnp.allclose(M_inv, expected, atol=1e-5), f"Pseudo-inverse incorrect"
