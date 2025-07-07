from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from mujoco_playground import MjxEnv

from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, Space


class MuJoCoPlaygroundToStoa(Environment):
    """MuJoCo Playground environments in Stoa interface."""

    def __init__(self, env: MjxEnv):
        """Initialize the MuJoCo Playground wrapper.

        Args:
            env: The MuJoCo Playground environment to wrap.
        """
        self._env = env

        # Cache action and observation dimensions
        self._action_size = env.action_size
        self._obs_size = env.observation_size

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[Any, TimeStep]:
        """Reset the environment."""
        # Reset the MuJoCo Playground environment
        mjx_state = self._env.reset(rng_key)

        # Create the initial timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=mjx_state.obs.astype(jnp.float32),
            extras={
                "metrics": mjx_state.metrics,
                **mjx_state.info,
            },
        )

        return mjx_state, timestep

    def step(
        self,
        state: Any,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[Any, TimeStep]:
        """Step the environment."""
        # Take a MuJoCo Playground step
        next_mjx_state = self._env.step(state, action)

        # Determine step type and discount
        # MuJoCo Playground uses 'done' for termination and may have 'truncated' in info
        terminated = next_mjx_state.done
        info = next_mjx_state.info
        truncated = info.get("truncated", jnp.array(False))

        # Determine step type based on termination/truncation
        step_type = jax.lax.select(
            jnp.logical_and(terminated, jnp.logical_not(truncated)),
            StepType.TERMINATED,
            jax.lax.select(truncated, StepType.TRUNCATED, StepType.MID),
        )

        # Discount is 0 for termination, 1 for truncation or continuation
        discount = jnp.where(jnp.logical_and(terminated, jnp.logical_not(truncated)), 0.0, 1.0)

        # Create the timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(next_mjx_state.reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=next_mjx_state.obs.astype(jnp.float32),
            extras={
                "metrics": next_mjx_state.metrics,
                **info,
            },
        )

        return next_mjx_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the observation space."""
        return BoundedArraySpace(
            shape=(self._obs_size,),
            dtype=jnp.float32,
            minimum=-jnp.inf,
            maximum=jnp.inf,
            name="observation",
        )

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the action space."""
        return BoundedArraySpace(
            shape=(self._action_size,),
            dtype=jnp.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the state space."""
        raise NotImplementedError(
            "MuJoCo Playground does not expose a state space. Use observation_space instead."
        )

    def render(self, state: Any, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment."""
        if hasattr(self._env, "render"):
            return self._env.render(state)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")
