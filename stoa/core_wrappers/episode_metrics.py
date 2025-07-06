from typing import TYPE_CHECKING, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Numeric, PRNGKey
from jax import Array

from stoa.env_types import Action, EnvParams, TimeStep
from stoa.wrappers.wrapper import StateWithKey, Wrapper

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class RecordEpisodeMetricsState(StateWithKey):
    # Temporary variables to keep track of the episode return and length.
    running_count_episode_return: Numeric
    running_count_episode_length: Numeric
    # Final episode return and length.
    episode_return: Numeric
    episode_length: Numeric


class RecordEpisodeMetrics(Wrapper[RecordEpisodeMetricsState]):
    """
    A wrapper that records episode returns and lengths for each episode.

    This wrapper tracks the cumulative reward (return) and the number of steps (length)
    for each episode. At each environment step, it updates these metrics and stores them
    in the `extras` field of the `TimeStep` object under the key "episode_metrics".

    The metrics are reset at the beginning of each episode.
    """

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """
        Resets the environment and episode metrics.

        Args:
            rng_key: A JAX PRNG key for random number generation.
            env_params: Optional environment parameters.

        Returns:
            A tuple containing the initial state and the first TimeStep, with episode metrics initialized.
        """
        rng_key, reset_key = jax.random.split(rng_key)
        base_env_state, timestep = self._env.reset(reset_key, env_params)
        state = RecordEpisodeMetricsState(
            base_env_state,
            rng_key,
            jnp.array(0.0, dtype=float),
            jnp.array(0, dtype=int),
            jnp.array(0.0, dtype=float),
            jnp.array(0, dtype=int),
        )
        episode_metrics = {
            "episode_return": jnp.array(0.0, dtype=float),
            "episode_length": jnp.array(0, dtype=int),
            "is_terminal_step": jnp.array(False, dtype=bool),
        }
        timestep.extras["episode_metrics"] = episode_metrics
        return state, timestep

    def step(
        self,
        state: RecordEpisodeMetricsState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """
        Steps the environment and updates episode metrics.

        Args:
            state: The current state, including episode metrics.
            action: The action to take in the environment.
            env_params: Optional environment parameters.

        Returns:
            A tuple containing the updated state and the new TimeStep, with updated episode metrics.
        """
        base_env_state, timestep = self._env.step(state.base_env_state, action, env_params)

        # Check if the episode has ended (done == 1 if terminal, 0 otherwise)
        done = timestep.done()
        not_done = 1 - done

        # Update the running counters for the current episode
        new_episode_return = state.running_count_episode_return + timestep.reward
        new_episode_length = state.running_count_episode_length + 1

        # If the episode is done, record the final return and length; otherwise, keep previous values
        episode_return_info = state.episode_return * not_done + new_episode_return * done
        episode_length_info = state.episode_length * not_done + new_episode_length * done

        episode_metrics = {
            "episode_return": episode_return_info,
            "episode_length": episode_length_info,
            "is_terminal_step": done,
        }
        timestep.extras["episode_metrics"] = episode_metrics

        state = RecordEpisodeMetricsState(
            base_env_state=base_env_state,
            rng_key=state.rng_key,
            running_count_episode_return=new_episode_return * not_done,
            running_count_episode_length=new_episode_length * not_done,
            episode_return=episode_return_info,
            episode_length=episode_length_info,
        )
        return state, timestep


def get_final_step_metrics(metrics: Dict[str, Array]) -> Tuple[Dict[str, Array], bool]:
    """Get the metrics for the final step of an episode and check if there was a final step
    within the provided metrics.

    Note: this is not a jittable method. We need to return variable length arrays, since
    we don't know how many episodes have been run. This is done since the logger
    expects arrays for computing summary statistics on the episode metrics.
    """
    is_final_ep = metrics.pop("is_terminal_step")
    has_final_ep_step = bool(jnp.any(is_final_ep))

    final_metrics: Dict[str, Array]
    # If it didn't make it to the final step, return zeros.
    if not has_final_ep_step:
        final_metrics = jax.tree_util.tree_map(jnp.zeros_like, metrics)
    else:
        final_metrics = jax.tree_util.tree_map(lambda x: x[is_final_ep], metrics)

    return final_metrics, has_final_ep_step
