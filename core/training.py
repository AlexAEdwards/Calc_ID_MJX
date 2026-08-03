"""Optimiser / TrainState construction.

Extracted verbatim from train.py in REFACTOR_PLAN.md Stage 5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


def create_train_state(rng, model, input_shape, static_shape, learning_rate=1e-4, weight_decay=0.01,
                       total_steps=None, warmup_frac=0.03, end_lr_frac=0.05, use_lr_schedule=True):
    """Build the initial train state.

    When ``use_lr_schedule`` is True and ``total_steps`` is known, the optimizer LR
    follows a warmup -> cosine-decay schedule (peak = ``learning_rate``), which is the
    standard recipe for stable transformer training. Set ``use_lr_schedule=False`` (or
    leave ``total_steps`` unset) to fall back to the original constant learning rate.

    Returns ``(state, lr_fn)`` where ``lr_fn(step)`` yields the LR at a given global
    step (a constant function when the schedule is disabled) for logging.
    """
    dummy_input = jnp.ones(input_shape)
    dummy_static = jnp.ones(static_shape)
    params = model.init(rng, dummy_input, dummy_static, train=False)["params"]

    if use_lr_schedule and total_steps is not None and int(total_steps) > 1:
        total_steps = int(total_steps)
        warmup_steps = max(1, int(warmup_frac * total_steps))
        # optax's decay_steps is the total horizon (warmup + cosine); the cosine phase
        # spans decay_steps - warmup_steps and ends exactly at end_value.
        lr_fn = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * 0.01,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=learning_rate * end_lr_frac,
        )
        lr_arg = lr_fn
    else:
        lr_arg = learning_rate
        lr_fn = (lambda _step, _lr=learning_rate: _lr)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_arg, weight_decay=weight_decay),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, lr_fn
