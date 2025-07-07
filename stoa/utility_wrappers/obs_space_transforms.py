from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, Observation, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, BoundedArraySpace, DiscreteSpace, Space


class AddStartFlagAndPrevAction(Wrapper[State]):
    """Wrapper that adds a start flag and the previous action to the observation.

    This wrapper modifies the observation to include:
    1. A start flag (1.0 for the first step, 0.0 for subsequent steps)
    2. The previous action (zero-initialized for the first step)
    3. The original observation

    The observation must be a flat array (1D). For discrete actions, the previous
    action is one-hot encoded before concatenation.
    """

    def __init__(self, env: Environment):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.

        Raises:
            ValueError: If the observation space is not a flat array.
            ValueError: If the action space is not supported.
        """
        super().__init__(env)

        # Get action space information
        action_space = self.action_space()
        if isinstance(action_space, DiscreteSpace):
            self.action_dim = action_space.num_values
            self._discrete = True
            self._process_action = lambda a: jax.nn.one_hot(a, self.action_dim, dtype=jnp.float32)
        elif isinstance(action_space, (ArraySpace, BoundedArraySpace)):
            if len(action_space.shape) != 1:
                raise ValueError("Only 1D continuous action spaces are supported.")
            self.action_dim = action_space.shape[0]
            self._discrete = False
            self._process_action = lambda a: a.astype(jnp.float32)
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        # Check if the observation is flat (1D array)
        obs_space = self.observation_space()
        if not isinstance(obs_space, (ArraySpace, BoundedArraySpace)):
            raise ValueError("Observation space must be an ArraySpace or BoundedArraySpace.")
        if len(obs_space.shape) != 1:
            raise ValueError("The observation must be a flat (1D) array.")

        self.orig_obs_dim = obs_space.shape[0]

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment and initialize the wrapper state.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (initial_state, initial_timestep) with modified observation.
        """
        state, timestep = self._env.reset(rng_key, env_params)

        # Initialize previous action as zeros
        # Always store as float32 array for consistency - discrete actions will be one-hot encoded
        prev_action = jnp.zeros(self.action_dim, dtype=jnp.float32)

        # Modify observation: [start_flag=1.0, prev_action_zeros, original_obs]
        start_flag = jnp.array([1.0], dtype=jnp.float32)

        # For the first step, previous action is always zeros (already in correct format)
        prev_action_encoded = prev_action

        new_observation = jnp.concatenate([start_flag, prev_action_encoded, timestep.observation])

        modified_timestep = timestep.replace(observation=new_observation)  # type: ignore

        return state, modified_timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment and update the wrapper state.

        Args:
            state: Current wrapper state.
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, new_timestep) with modified observation.
        """
        # Step the base environment
        new_state, timestep = self._env.step(state, action, env_params)

        # Process the action to ensure it is in the correct format
        processed_action = self._process_action(action)

        # Modify observation: [start_flag=0.0, prev_action, original_obs]
        start_flag = jnp.array([0.0], dtype=jnp.float32)
        new_observation = jnp.concatenate([start_flag, processed_action, timestep.observation])

        modified_timestep = timestep.replace(observation=new_observation)  # type: ignore

        return new_state, modified_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the modified observation space.

        The new observation space has dimension:
        1 (start flag) + action_dim + original_observation_dim

        Args:
            env_params: Optional environment parameters.

        Returns:
            Modified observation space.
        """
        orig_obs_space = self._env.observation_space(env_params)
        new_obs_dim = 1 + self.action_dim + self.orig_obs_dim
        kwargs = orig_obs_space.__dict__
        kwargs["shape"] = (new_obs_dim,)
        # Create a new space with the modified shape
        return type(orig_obs_space)(**kwargs)


class MakeChannelLast(Wrapper[State]):
    """Simple wrapper for observations that have the channel dim first.
    This makes the channel dim last.

    This wrapper transforms observations from channel-first (e.g., CHW) to
    channel-last (e.g., HWC) format. It only modifies observations and does not
    change the environment state, so no state wrapping is needed.
    """

    def __init__(self, env: Environment) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.

        Raises:
            AssertionError: If observation is not > 2 dimensional.
            ValueError: If observation space is not supported.
        """
        super().__init__(env)

        # Get observation space and validate it's multi-dimensional
        obs_space = self.observation_space()
        if not isinstance(obs_space, (ArraySpace, BoundedArraySpace)):
            raise ValueError("Observation space must be an ArraySpace or BoundedArraySpace.")

        obs_shape = jnp.array(obs_space.shape)
        assert len(obs_shape) > 2, "MakeChannelLast requires > 2 dimensional observations"

        # Calculate new shape with channel moved to last position
        # Roll the first axis (channel) to the last position
        self._new_obs_shape = tuple(jnp.roll(obs_shape, len(obs_shape) - 1))

    def _make_channel_last(self, observation: Observation) -> Observation:
        """Transform observation from channel-first to channel-last format.

        Args:
            observation: The observation to transform.

        Returns:
            Transformed observation with channel dimension last.
        """
        # Move the first axis (channel) to the last position
        return jnp.moveaxis(observation, 0, -1)

    def _transform_timestep_observations(self, timestep: TimeStep) -> TimeStep:
        """Transform all observations in a timestep.

        Args:
            timestep: The timestep with observations to transform.

        Returns:
            Timestep with transformed observations.
        """
        # Transform main observation
        new_observation = self._make_channel_last(timestep.observation)

        return timestep.replace(  # type: ignore
            observation=new_observation,
        )

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Reset the environment and transform observations.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (state, timestep) with transformed observations.
            State is passed through unchanged.
        """
        state, timestep = self._env.reset(rng_key, env_params)
        transformed_timestep = self._transform_timestep_observations(timestep)
        return state, transformed_timestep

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Step the environment and transform observations.

        Args:
            state: Current environment state (passed through unchanged).
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, new_timestep) with transformed observations.
            State is passed through unchanged.
        """
        new_state, timestep = self._env.step(state, action, env_params)
        transformed_timestep = self._transform_timestep_observations(timestep)
        return new_state, transformed_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the transformed observation space.

        Args:
            env_params: Optional environment parameters.

        Returns:
            Observation space with channel dimension moved to last position.
        """
        orig_obs_space = self._env.observation_space(env_params)
        kwargs = orig_obs_space.__dict__
        kwargs["shape"] = self._new_obs_shape
        # Create a new space with the modified shape
        return type(orig_obs_space)(**kwargs)
