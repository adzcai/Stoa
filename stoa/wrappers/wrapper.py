from typing import TYPE_CHECKING, Any, Generic, Optional, Tuple, TypeVar

import jax
from chex import PRNGKey

from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, EnvironmentSpace, Space

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class WrapperState:
    """
    Base state class for environment wrappers.

    Attributes:
        base_env_state: The state of the underlying (wrapped) environment.
    """

    base_env_state: State

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the base environment state if not found.

        Args:
            name: The attribute name.

        Returns:
            The attribute value from base_env_state if present.

        Raises:
            AttributeError: If the attribute is not found in either self or base_env_state.
        """
        # Called only if attribute not found the normal way
        if name == "__setstate__":
            raise AttributeError(name)
        try:
            return getattr(self.base_env_state, name)
        except AttributeError as e:
            raise AttributeError(f"{name} not found in WrapperState or base_env_state") from e


S = TypeVar("S", bound="WrapperState")


class Wrapper(Environment, Generic[S]):
    """
    Base class for stoa environment wrappers.

    This class wraps an existing environment, allowing for additional
    functionality to be layered on top of the base environment. By default,
    it delegates all methods to the wrapped environment.
    """

    def __init__(self, env: Environment):
        """
        Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__()
        self._env = env

    def __repr__(self) -> str:
        """Return a string representation of the wrapper and its wrapped environment."""
        return f"{self.__class__.__name__}({repr(self._env)})"

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the wrapped environment.

        Args:
            name: The attribute name.

        Returns:
            The attribute value from the wrapped environment.

        Raises:
            AttributeError: If the attribute is not found.
        """
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> Environment:
        """
        Returns the base (unwrapped) environment.

        Returns:
            The innermost wrapped environment.
        """
        return self._env.unwrapped

    def reset(self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None) -> Tuple[S, TimeStep]:
        """
        Reset the environment.

        Args:
            rng_key: A JAX PRNG key for random number generation.
            env_params: Optional environment parameters.

        Returns:
            A tuple of the initial state and the first TimeStep.
        """
        return self._env.reset(rng_key, env_params)

    def step(
        self,
        state: S,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[S, TimeStep]:
        """
        Take a step in the environment.

        Args:
            state: The current environment state.
            action: The action to take.
            env_params: Optional environment parameters.

        Returns:
            A tuple of the new state and the resulting TimeStep.
        """
        return self._env.step(state.base_env_state, action, env_params)

    def reward_space(self, env_params: Optional[EnvParams] = None) -> BoundedArraySpace:
        return self._env.reward_space(env_params)

    def discount_space(self, env_params: Optional[EnvParams] = None) -> BoundedArraySpace:
        return self._env.discount_space(env_params)

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return self._env.observation_space(env_params)

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return self._env.action_space(env_params)

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return self._env.state_space(env_params)

    def environment_space(self, env_params: Optional[EnvParams] = None) -> EnvironmentSpace:
        return self._env.environment_space(env_params)

    def render(self, state: S, env_params: Optional[EnvParams] = None) -> Any:
        return self._env.render(state.base_env_state, env_params)

    def close(self) -> None:
        self._env.close()


@dataclass
class StateWithKey(WrapperState):
    """
    Wrapper state that includes a JAX PRNG key.
    """

    rng_key: PRNGKey


class AddRNGKey(Wrapper):
    """
    Wrapper that adds a JAX PRNG key to the environment state.

    This allows environments that do not natively manage RNG keys to be
    compatible with other stoa wrappers.
    """

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[StateWithKey, TimeStep]:
        """
        Reset the environment and initialize the RNG key in the state.

        Args:
            rng_key: A JAX PRNG key for random number generation.
            env_params: Optional environment parameters.

        Returns:
            A tuple of the initial state (with RNG key) and the first TimeStep.
        """
        rng_key, wrapped_state_key = jax.random.split(rng_key)
        base_env_state, timestep = self._env.reset(rng_key, env_params)
        wrapped_env_state = StateWithKey(base_env_state, wrapped_state_key)
        return wrapped_env_state, timestep

    def step(
        self,
        state: StateWithKey,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[StateWithKey, TimeStep]:
        """
        Take a step in the environment, updating the RNG key in the state.

        Args:
            state: The current state, including the RNG key.
            action: The action to take.
            env_params: Optional environment parameters.

        Returns:
            A tuple of the new state (with updated RNG key) and the resulting TimeStep.
        """
        rng_key = state.rng_key
        base_env_state, timestep = self._env.step(state.base_env_state, action, env_params)
        rng_key, wrapped_state_key = jax.random.split(rng_key)
        wrapped_env_state = StateWithKey(base_env_state, wrapped_state_key)
        return wrapped_env_state, timestep
