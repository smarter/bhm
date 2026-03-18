from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DigitsData(NamedTuple):
    x_train: Float[Array, "N_train 64"]
    y_train: Int[Array, " N_train"]
    x_test: Float[Array, "N_test 64"]
    y_test: Int[Array, " N_test"]


def load_digits_data(seed: int = 0) -> DigitsData:
    digits = load_digits()
    x: Float[Array, "1797 64"] = jnp.array(digits.data / 16.0, dtype=jnp.float32)  # pyright: ignore[reportAttributeAccessIssue]
    y: Int[Array, " 1797"] = jnp.array(digits.target, dtype=jnp.int32)  # pyright: ignore[reportAttributeAccessIssue]

    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
        x, y, test_size=0.1, random_state=seed, stratify=y
    )

    return DigitsData(
        x_train=jnp.array(x_train_np),
        y_train=jnp.array(y_train_np),
        x_test=jnp.array(x_test_np),
        y_test=jnp.array(y_test_np),
    )
