from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from stoa.core_wrappers.wrapper import StateWithKey
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, BoundedArraySpace, DictSpace, DiscreteSpace, Space


def jaxmarl_space_to_stoa_space(space: Any) -> Space:
    """Convert JaxMARL space to Stoa space."""
    # Handle discrete spaces
    if hasattr(space, "n"):
        return DiscreteSpace(num_values=space.n, dtype=getattr(space, "dtype", jnp.int32))

    # Handle box/continuous spaces
    if hasattr(space, "low") and hasattr(space, "high"):
        return BoundedArraySpace(
            shape=space.shape,
            dtype=getattr(space, "dtype", jnp.float32),
            minimum=space.low,
            maximum=space.high,
        )

    # Handle dict spaces
    if hasattr(space, "spaces") and isinstance(space.spaces, dict):
        return DictSpace(
            spaces={k: jaxmarl_space_to_stoa_space(v) for k, v in space.spaces.items()}
        )

    raise TypeError(f"Unsupported JaxMARL space type: {type(space)}")


class JaxMarlToStoa(Environment):
    """Minimal adapter for JaxMARL multi-agent environments to Stoa interface.

    Converts multi-agent environments to single-agent by batching observations/actions.
    Only supports homogeneous environments where all agents have identical spaces.
    """

    def __init__(self, env: MultiAgentEnv):
        """Initialize the JaxMARL adapter.

        Args:
            env: The JaxMARL environment to wrap

        Raises:
            ValueError: If environment is not homogeneous
        """
        self._env = env
        self._agents = list(env.agents)
        self._num_agents = len(self._agents)

        # Verify homogeneous environment
        if not self._is_homogeneous():
            raise ValueError(
                "JaxMarlToStoa only supports homogeneous environments. "
                "All agents must have identical observation and action spaces."
            )

    def _is_homogeneous(self) -> bool:
        """Check if all agents have identical spaces."""
        if self._num_agents <= 1:
            return True

        first_agent = self._agents[0]
        first_obs = self._env.observation_space(first_agent)
        first_act = self._env.action_space(first_agent)

        for agent in self._agents[1:]:
            obs_space = self._env.observation_space(agent)
            act_space = self._env.action_space(agent)

            # Compare shapes as proxy for space equality
            if getattr(obs_space, "shape", None) != getattr(first_obs, "shape", None) or getattr(
                act_space, "shape", None
            ) != getattr(first_act, "shape", None):
                return False

        return True

    def _batchify(self, agent_dict: Dict[str, Any]) -> jnp.ndarray:
        """Convert agent dictionary to batched array."""
        return jnp.stack([agent_dict[agent] for agent in self._agents])

    def _unbatchify(self, batched: jnp.ndarray) -> Dict[str, Any]:
        """Convert batched array to agent dictionary."""
        return {agent: batched[i] for i, agent in enumerate(self._agents)}

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[StateWithKey, TimeStep]:
        """Reset the environment."""
        # Reset JaxMARL environment
        rng_key, reset_key = jax.random.split(rng_key)
        obs_dict, env_state = self._env.reset(reset_key)

        # Create batched observation
        batched_obs = self._batchify(obs_dict)

        # Create initial timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.zeros(self._num_agents, dtype=jnp.float32),
            discount=jnp.ones(self._num_agents, dtype=jnp.float32),
            observation=batched_obs,
            extras={},
        )

        env_state = StateWithKey(
            base_env_state=env_state,
            rng_key=reset_key,
        )

        return env_state, timestep

    def step(
        self,
        state: StateWithKey,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[Any, TimeStep]:
        """Step the environment."""
        # Convert batched action to agent dictionary
        action_dict = self._unbatchify(action)

        # Step JaxMARL environment
        rng_key, step_key = jax.random.split(state.rng_key)
        obs_dict, env_state, reward_dict, done_dict, info_dict = self._env.step(
            key=step_key, state=state.base_env_state, actions=action_dict
        )

        # Batch observations and rewards
        batched_obs = self._batchify(obs_dict)
        batched_rewards = self._batchify(reward_dict)
        batched_dones = self._batchify(done_dict)

        # Determine step type based on done_dict
        episode_done = done_dict.get("__all__", jnp.any(batched_dones))
        step_type = jax.lax.select(episode_done, StepType.TERMINATED, StepType.MID)

        # Create timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=batched_rewards,
            discount=1.0 - batched_dones.astype(jnp.float32),
            observation=batched_obs,
            extras=info_dict,
        )

        # Wrap the new environment state with the RNG key
        env_state = StateWithKey(
            base_env_state=env_state,
            rng_key=rng_key,
        )

        return env_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the batched observation space."""
        # Get single agent space and convert it
        single_space = jaxmarl_space_to_stoa_space(self._env.observation_space(self._agents[0]))

        # Create batched version
        if isinstance(single_space, DiscreteSpace):
            # For discrete obs, keep as is but note it's batched
            return ArraySpace(
                shape=(self._num_agents,), dtype=single_space.dtype, name="batched_discrete_obs"
            )
        elif isinstance(single_space, BoundedArraySpace):
            return BoundedArraySpace(
                shape=(self._num_agents,) + single_space.shape,
                dtype=single_space.dtype,
                minimum=single_space.minimum,
                maximum=single_space.maximum,
                name="batched_obs",
            )
        elif isinstance(single_space, ArraySpace):
            return ArraySpace(
                shape=(self._num_agents,) + single_space.shape,
                dtype=single_space.dtype,
                name="batched_obs",
            )
        elif isinstance(single_space, DictSpace):
            # For dict spaces, batch each component
            batched_spaces: Dict[str, Space] = {}
            for key, subspace in single_space.spaces.items():
                if hasattr(subspace, "shape") and subspace.shape is not None:
                    batched_shape = (self._num_agents,) + subspace.shape
                    if isinstance(subspace, BoundedArraySpace):
                        batched_spaces[key] = BoundedArraySpace(
                            shape=batched_shape,
                            dtype=subspace.dtype,
                            minimum=subspace.minimum,
                            maximum=subspace.maximum,
                        )
                    else:
                        batched_spaces[key] = ArraySpace(shape=batched_shape, dtype=subspace.dtype)
                else:
                    batched_spaces[key] = subspace
            return DictSpace(batched_spaces)

        return single_space

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the batched action space."""
        # Get single agent space and convert it
        single_space = jaxmarl_space_to_stoa_space(self._env.action_space(self._agents[0]))

        # Create batched version
        if isinstance(single_space, DiscreteSpace):
            # For discrete actions, return array of discrete values
            return ArraySpace(
                shape=(self._num_agents,), dtype=single_space.dtype, name="batched_discrete_actions"
            )
        elif isinstance(single_space, BoundedArraySpace):
            return BoundedArraySpace(
                shape=(self._num_agents,) + single_space.shape,
                dtype=single_space.dtype,
                minimum=single_space.minimum,
                maximum=single_space.maximum,
                name="batched_actions",
            )
        elif isinstance(single_space, ArraySpace):
            return ArraySpace(
                shape=(self._num_agents,) + single_space.shape,
                dtype=single_space.dtype,
                name="batched_actions",
            )

        return single_space

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Get the state space."""
        # Return generic space as JaxMARL state structure varies by environment
        return ArraySpace(shape=(), dtype=jnp.int32, name="jaxmarl_state")

    def render(self, state: Any, env_params: Optional[EnvParams] = None) -> Any:
        """Render the environment if supported."""
        if hasattr(self._env, "render"):
            return self._env.render(state)
        else:
            raise NotImplementedError(f"Rendering not supported for {self._env.__class__.__name__}")
