from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment


class OptimisticResetVMapWrapper(Wrapper[State]):
    """
    Efficient vectorized environment wrapper with optimistic resets.
    
    This wrapper combines environment vectorization (vmap) with automatic resets,
    using an "optimistic" strategy that pre-generates reset states for efficiency.
    Instead of resetting environments one-by-one as they terminate, it generates
    a smaller number of reset states and distributes them to terminated environments.
    
    Note: This wrapper requires the environment state to have an 'rng_key' attribute.
    Use the AddRNGKey wrapper before this one if your environment doesn't have it.
    
    Args:
        env: The base environment to vectorize.
        num_envs: Number of parallel environments to run.
        reset_ratio: Number of environments per reset state generated.
                    Higher values are more efficient but may cause duplicate resets.
                    Must divide num_envs evenly.
    """

    def __init__(
        self,
        env: Environment,
        num_envs: int,
        reset_ratio: int
    ):
        super().__init__(env)

        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if num_envs % reset_ratio != 0:
            raise ValueError(
                f"reset_ratio ({reset_ratio}) must evenly divide num_envs ({num_envs})"
            )

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        self.num_resets = num_envs // reset_ratio

        self._vmap_reset = jax.vmap(self._env.reset, in_axes=(0, None), out_axes=(0, 0))
        self._vmap_step = jax.vmap(self._env.step, in_axes=(0, 0, None), out_axes=(0, 0))

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        """Resets all vectorized environments."""
        reset_keys = jax.random.split(rng_key, self.num_envs)
        states, timesteps = self._vmap_reset(reset_keys, env_params)
        return states, timesteps

    def step(
        self,
        state: State,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[State, TimeStep]:
        """Steps all vectorized environments with robust optimistic auto-reset."""
        # Step all environments
        new_states, timesteps = self._vmap_step(state, action, env_params)

        # Extract done flags and handle RNG keys for resets
        dones = timesteps.done()
        reset_keys_parent = new_states.rng_key
        reset_keys_base, reset_keys_select = jax.vmap(jax.random.split)(reset_keys_parent)
        new_states = new_states.replace(rng_key=reset_keys_base)

        # Pre-generate reset states and timesteps
        reset_keys = reset_keys_select[:self.num_resets]
        reset_states, reset_timesteps = self._vmap_reset(reset_keys, env_params)

        # Distribute reset states to done environments
        done_indices, _ = jnp.nonzero(dones, size=self.num_envs, fill_value=-1)
        done_count = jnp.count_nonzero(dones)

        # Create a mapping that cycles through the available reset states
        reset_assignments = jnp.arange(done_count) % self.num_resets
        
        # Scatter the assignments into a map at the indices of the 'done' environments
        reset_indices = jnp.zeros_like(dones, dtype=jnp.int32)
        reset_indices = reset_indices.at[done_indices].set(reset_assignments, mode='drop')
        
        # TODO (edan): finish this logic to handle the reset states
        raise NotImplementedError("This is not finished yet.")

        # return final_states, final_timesteps