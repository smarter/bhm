import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, depth: int, width: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, depth + 1)
        self.layers = []
        # Input layer
        self.layers.append(eqx.nn.Linear(64, width, key=keys[0]))
        # Hidden layers
        for i in range(1, depth):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i]))
        # Output layer
        self.layers.append(eqx.nn.Linear(width, 10, key=keys[depth]))

    def __call__(self, x: Float[Array, " 64"]) -> Float[Array, " 10"]:
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)
