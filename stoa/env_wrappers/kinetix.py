from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from kinetix.environment.env import KinetixObservation
from kinetix.render.renderer_pixels import PixelsObservation

from stoa.core_wrappers.wrapper import StateWithKey
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.env_wrappers.gymnax import GymnaxToStoa


class KinetixToStoa(GymnaxToStoa):
    """Kinetix environments in Stoa interface.

    Inherits from GymnaxToStoa and only overrides Kinetix-specific behavior
    for observation handling and metrics extraction.
    """

    def _fix_obs(self, obs: KinetixObservation) -> jnp.ndarray:
        """Fix observation to handle PixelsObservation objects.

        Args:
            obs: Raw observation from Kinetix environment.

        Returns:
            Fixed observation as JAX array.
        """
        if isinstance(obs, PixelsObservation):
            return obs.image
        return jnp.asarray(obs)

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[StateWithKey, TimeStep]:
        """Reset the environment with Kinetix-specific observation handling."""
        reset_key, state_key = jax.random.split(rng_key)

        # If env_params are not provided use the default
        if env_params is None:
            env_params = self._env_params

        # Reset the environment (same as parent but we need access to raw obs)
        obs, kinetix_state = self._env.reset(reset_key, env_params)

        # Wrap the state with an rng key
        state = StateWithKey(
            base_env_state=kinetix_state,
            rng_key=state_key,
        )

        # Fix observation for Kinetix
        fixed_obs = self._fix_obs(obs)

        # Create the timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=fixed_obs,
            extras={},
        )

        return state, timestep

    def step(
        self,
        state: StateWithKey,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[StateWithKey, TimeStep]:
        """Step the environment with Kinetix-specific observation and metrics handling."""
        step_key, next_key = jax.random.split(state.rng_key)

        # If env_params are not provided use the default
        if env_params is None:
            env_params = self._env_params

        # Take a step
        obs, kinetix_state, reward, done, info = self._env.step(
            step_key, state.base_env_state, action, env_params
        )

        # Wrap the state with a new key
        new_state = StateWithKey(
            base_env_state=kinetix_state,
            rng_key=next_key,
        )

        # Fix observation for Kinetix
        fixed_obs = self._fix_obs(obs)

        # Extract Kinetix-specific metrics from info
        extras = {}
        if "GoalR" in info:
            extras["solve_rate"] = jnp.asarray(info["GoalR"], dtype=jnp.float32)
        if "distance" in info:
            extras["distance"] = jnp.asarray(info["distance"], dtype=jnp.float32)

        # Add any other info
        extras.update({k: v for k, v in info.items() if k not in ["GoalR", "distance"]})

        # Kinetix has no truncation, only termination (same as Gymnax)
        step_type = jax.lax.select(done, StepType.TERMINATED, StepType.MID)

        # Create the timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(1.0 - done, dtype=jnp.float32),
            observation=fixed_obs,
            extras=extras,
        )

        return new_state, timestep
