import math

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from bhm.model import MLP


def cross_entropy_loss(
    model: MLP, x: Float[Array, "N 64"], y: Int[Array, " N"]
) -> Float[Array, ""]:
    logits = jax.vmap(model)(x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, 10) * log_probs, axis=-1))


def train(
    model: MLP,
    x_train: Float[Array, "N 64"],
    y_train: Int[Array, " N"],
    *,
    epochs: int,
    batch_size: int = 32,
    lr: float = 0.03,
    key: PRNGKeyArray,
) -> MLP:
    n = x_train.shape[0]
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = epochs * steps_per_epoch

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps)
    optimizer = optax.sgd(learning_rate=schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(
        model: MLP,
        opt_state: optax.OptState,
        x_batch: Float[Array, "B 64"],
        y_batch: Int[Array, " B"],
    ) -> tuple[MLP, optax.OptState, Float[Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(cross_entropy_loss)(
            model, x_batch, y_batch
        )
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for epoch in range(epochs):
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for i in range(0, n, batch_size):
            x_batch = x_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]
            model, opt_state, loss = step(model, opt_state, x_batch, y_batch)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            full_loss = cross_entropy_loss(model, x_train, y_train)
            print(f"  Epoch {epoch + 1}/{epochs}, loss: {full_loss:.4f}")

    return model
