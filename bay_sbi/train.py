# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script for normalizing flow dark matter substructure inference."""

import bisect
import copy
import functools
import time
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
import optax
from optax import GradientTransformation, EmptyState


def initialized(
    rng: Sequence[int], context_dim: int, parameter_dim: int, model: Any,
    image_context: Optional[bool] = False
) -> Tuple[Any, Any]:
    """Initialize the model parameters

    Args:
        rng: jax PRNG key.
        context_dim: Size of the context / input image.
        parameter_dim: Size of parameter dimension.
        model: Model class to initialize.
        image_context: If true the context is assumed to be an image.

    Returns:
        Initialized model parameters and batch stats.
    """
    if image_context:
        context_shape = (1, context_dim, context_dim, 1)
    else:
        context_shape = (1, context_dim)
    y_shape = (1, parameter_dim)
    @jax.jit
    def init(*args):
        return model.init(*args)
    variables = init(
        {'params': rng}, jnp.ones(y_shape),
        jnp.ones(context_shape)
    )
    return variables['params'], variables['batch_stats']


def _get_optimizer(
    optimizer: str,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float]
) -> Any:
    """Create the optax optimizer instance.

    Args:
        optimizer: Optimizer to use.
        learning_rate_schedule: Learning rate schedule.

    Returns:
        Optimizer instance.
    """
    if optimizer == 'adam':
        tx = optax.adam(
            learning_rate=learning_rate_schedule
        )
    elif optimizer == 'sgd':
        tx = optax.sgd(
            learning_rate=learning_rate_schedule
        )
    else:
        raise ValueError(f'Optimizer {optimizer} is not an option.')
    return tx


def get_optimizer(
    optimizer: str,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float],
    params: Mapping[str, jnp.ndarray]
) -> Any:
    """Create the optax optimizer instance with masking for int parameters.

    Args:
        optimizer: Optimizer to use.
        learning_rate_schedule: Learning rate schedule.
        params: Parameters of the initialzied model. Used to extract the dtype
            of the parameters.

    Returns:
        Optimizer instance and the optimizer mask.
    """
    base_optimizer = _get_optimizer(optimizer, learning_rate_schedule)

    # Map out the int parameters and tell the optimizer it can freeze them.
    def _find_int(param):
        if (param.dtype == jnp.int32 or param.dtype == jnp.int64):
            return 'freeze'
        return 'train'

    opt_mask = jax.tree_map(_find_int, params)

    # Create a custom update function for our integer parameters.
    def _init_empty_state(params) -> EmptyState:
        del params
        return EmptyState()

    def set_to_zero() -> GradientTransformation:
        def update_fn(updates, state, params=None):
            del params  # Unused by the zero transform.
            return (
                jax.tree_util.tree_map(
                    functools.partial(jnp.zeros_like, dtype=int), updates
                ), # Force int dtype to avoid type errors from the jitted func.
                state
            )

        return GradientTransformation(_init_empty_state, update_fn)

    # Map between the two optimizers depending on the parameter.
    optimizer = optax.multi_transform(
        {'train': base_optimizer, 'freeze': set_to_zero()},
        param_labels=opt_mask
    )

    return optimizer, jax.tree_map(lambda x: x == 'freeze', opt_mask)


class TrainState(train_state.TrainState):
    """Training state class for models with optimizer mask."""
    batch_stats: Any
    opt_mask: Any


def create_train_state_nf(
    rng: Sequence[int], optimizer: str,
    model: Any, image_size: int, parameter_dim: int,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float],
    image_context: Optional[bool] = False
) -> TrainState:
    """Create initial training state for flow model.

    Args:
        rng: jax PRNG key.
        optimizer: Optimizer name.
        model: Instance of model architecture.
        image_size: Dimension of square image.
        learning_rate_schedule: Learning rate schedule.
        image_context: If true the context is assumed to be an image.

    Returns:
        Initialized TrainState for model.
    """
    params, batch_stats = initialized(
        rng, image_size, parameter_dim, model, image_context
    )
    tx, opt_mask = get_optimizer(optimizer, learning_rate_schedule, params)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        opt_mask=opt_mask
    )
    return state


def extract_flow_context(
    state: TrainState, target: jnp.ndarray, context_batch: jnp.ndarray
) -> Tuple[Mapping[str, Mapping[str, jnp.ndarray]], jnp.ndarray]:
    """Extract flow parameters and the target image context.

    Args:
        state: Current TrainState object for the model.
        target: Target context.
        context_batch: Batch of context for batch normalization purposes.

    Returns:
        Flow parameters and encoded context.
    """
    # Insert image into the training batch.
    context_batch = context_batch.at[0].set(target)
    # Extract the flow parameters.
    flow_params = {
        'params': state.params['flow_module']
    }
    # Extract the context from the model.
    context, _ = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        context_batch, mutable=['batch_stats'],
        method='embed_context'
    )
    # We only want the context for the target image.
    context = context[0]
    return flow_params, context


def gaussian_log_prob(
    mean: jnp.ndarray, prec: jnp.ndarray, truth: jnp.ndarray
) -> jnp.ndarray:
    """Gaussian log probability calculated on mean, covariance, and truth.

    Args:
        mean: Mean of Gaussian.
        prec: Precision matrix of Gaussian.
        truth: True value of the parameters.

    Returns:
        Gaussian loss including only terms that depend on truth (i.e. dropping
        determinant of the covariance matrix.)

    Notes:
        Does not inlcude terms that are constant in the truth.
    """
    error = truth - mean
    log_prob = -0.5 * jnp.einsum('...n,nm,...m->...', error, prec, error)

    return log_prob


def apt_loss(log_posterior: jnp.ndarray, log_prior: jnp.ndarray) -> jnp.ndarray:
    """APT loss with Gaussian prior.

    Args:
        log_posterior: Log posterior output by the model. Has shape (batch_size,
            n_atoms).
        log_prior: Log prior for truth values Has shape (batch_size, n_atoms).

    Returns:
        APT loss for given posterior and prior.
    """
    # Fundamental quantity is ratio of posterior to prior.
    log_prop_posterior_full = log_posterior - log_prior

    # Normalize each batch by values on remaining samples.
    log_prop_posterior = (
        log_prop_posterior_full[:, 0] -
        jax.scipy.special.logsumexp(log_prop_posterior_full, axis=1)
    )
    return -jnp.mean(log_prop_posterior)


def apt_get_atoms(
    rng: Sequence[int], truth: jnp.ndarray, n_atoms: int
) -> jnp.ndarray:
    """Return atoms for each truth in the batch.

    Args:
        rng: jax PRNG key.
        truth: Truth values to sample for atoms.
        n_atoms: Number of atoms for each truth.

    Returns:
        Atoms with shape (batch_size, n_atoms). The first atom is always the
        true value at that index.
    """
    # Different random permutation for each truth
    rng_perm = jax.random.split(rng, len(truth))

    # Select the contrastive indices for each truth.
    choice_vmap = jax.vmap(
        functools.partial(
            jax.random.choice, shape=(n_atoms - 1,), replace=False
        ),
        in_axes=[0, None]
    )
    # One less than length since we can't select the truth for contrastive.
    cont_indices = choice_vmap(rng_perm, len(truth) - 1)

    # Shift indices >= the true index for each batch to ensure the true index is
    # never choces.
    shift_mask = cont_indices < jnp.arange(len(truth))[:, None]
    cont_indices = cont_indices * shift_mask + (cont_indices + 1) * ~shift_mask

    return jnp.concatenate([truth[:, None], truth[cont_indices]], axis=1)


def train_step(
    rng: Sequence[int], state: TrainState, batch: Mapping[str, jnp.ndarray],
    mu_prior: jnp.ndarray, prec_prior: jnp.ndarray,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float],
    n_atoms: int, opt_mask: Mapping[str, jnp.ndarray]
) -> Tuple[TrainState, Mapping[str, Any]]:
    """Perform a single training step.

    Args:
        state: Current TrainState object for the model.
        batch: Dictionary of images and truths to be used for training.
        mu_prior: Mean of the prior distribution.
        prec_prior: Precision matrix for the prior distribution.
        learning_rate_schedule: Learning rate schedule to apply.

    Returns:
        Updated TrainState object and metrics for training step.
    """
    truth = batch['truth']
    context = batch['context']

    # Get atoms for each evaluation.
    truth_apt = apt_get_atoms(rng, truth, n_atoms)
    log_prior = gaussian_log_prob(mu_prior, prec_prior, truth_apt)

    # Select the thetas we will use for the apt_loss
    def loss_fn(params):
        """Loss function for training."""
        log_posterior, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            truth_apt, context, mutable=['batch_stats'], method='call_apt'
        )
        loss = apt_loss(log_posterior, log_prior)
        return loss, new_model_state

    # Extract learning rate for current step.
    step = state.step
    lr = learning_rate_schedule(step)

    # Extract gradients for weight updates and current model state / loss.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    (loss, new_model_state), grads = grad_fn(state.params)

    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    def _pmean_if_not_freeze(grad, freeze_grad):
        # Apply pmean only if it is not a frozen gradient.
        if freeze_grad:
            return grad
        return jax.lax.pmean(grad, axis_name='batch')
    grads = jax.tree_map(_pmean_if_not_freeze, grads, opt_mask)

    metrics = {'learning_rate' : lr, 'loss': loss}

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats']
    )

    return new_state, metrics
