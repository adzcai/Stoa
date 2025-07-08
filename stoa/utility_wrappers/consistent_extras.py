from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment


class ConsistentExtrasWrapper(Wrapper[State]):
    """Ensures TimeStep.extras has consistent structure for JAX scanning.

    This wrapper performs a single dummy step during initialization to discover
    all possible keys in extras, then ensures these keys are always present
    (with zero values when missing) in both reset and step.
    """

    def __init__(
        self,
        env: Environment,
        rng_key: Optional[PRNGKey] = None,
        env_params: Optional[EnvParams] = None,
    ):
        """Initialize by discovering extras structure via a dummy step.

        Args:
            env: The environment to wrap.
            rng_key: Random key for the dummy reset/step.
            env_params: Optional environment parameters.
        """
        super().__init__(env)

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Do a dummy reset and step to discover extras structure
        reset_key, step_key = jax.random.split(rng_key)
        dummy_state, reset_timestep = env.reset(reset_key, env_params)
        dummy_action = env.action_space(env_params).sample(step_key)
        _, step_timestep = env.step(dummy_state, dummy_action, env_params)

        # Merge extras from both reset and step
        all_extras = dict(reset_timestep.extras)
        all_extras.update(step_timestep.extras)

        # Create zero-filled template
        self._extras_template = jax.tree_map(lambda x: jnp.zeros_like(x), all_extras)

    def _fill_extras(self, extras: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing keys with zeros from template."""
        filled = dict(self._extras_template)
        filled.update(extras)
        return filled

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(rng_key, env_params)
        return state, timestep.replace(extras=self._fill_extras(timestep.extras))  # type: ignore

    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        new_state, timestep = self._env.step(state, action, env_params)
        return new_state, timestep.replace(extras=self._fill_extras(timestep.extras))  # type: ignore
