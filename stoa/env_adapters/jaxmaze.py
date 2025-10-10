from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from housemaze.env import DIR_TO_VEC
from housemaze.human_dyna.multitask_env import EnvParams, HouseMaze
from housemaze.human_dyna.multitask_env import StepType as MazeStepType
from housemaze.human_dyna.multitask_env import TimeStep as MazeStep
from housemaze.renderer import create_image_from_grid
from housemaze.utils import load_image_dict

from stoa.core_wrappers.wrapper import StateWithKey
from stoa.env_types import Action, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import DictSpace, DiscreteSpace, Space


IMAGE_DICT = load_image_dict()

class JaxMazeToStoa(Environment):
    def __init__(self, env: HouseMaze, train_params: EnvParams, test_params: EnvParams):
        self._env = env
        self._train_params = train_params
        self._test_params = test_params

    @property
    def num_objects(self):
        return len(self._env.task_runner.task_objects)

    def params(self, eval: bool | None = None):
        return self._test_params if eval else self._train_params

    def format_obs(self, raw_timestep: MazeStep) -> dict[str, jax.Array]:
        return {
            field: getattr(raw_timestep.observation, field).astype(jnp.uint8)
            for field in (
                "image",
                "task_w",
                "state_features",
                "position",
                "direction",
                "prev_action",
            )
        }

    def reset(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, rng_key: PRNGKey, env_params: bool | None = None
    ) -> Tuple[StateWithKey, TimeStep]:
        rng_key, reset_key = jax.random.split(rng_key)
        raw_timestep = self._env.reset(reset_key, self.params(env_params))
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=self.format_obs(raw_timestep),
        )
        return StateWithKey(raw_timestep, rng_key), timestep

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: StateWithKey, action: Action, env_params: bool | None = None
    ) -> tuple[StateWithKey, TimeStep]:
        rng_key, step_key = jax.random.split(state.rng_key)
        raw_timestep = self._env.step(
            step_key,
            state.base_env_state,
            action,  # pyright: ignore[reportArgumentType]
            self.params(env_params),
        )
        t = raw_timestep.step_type
        step_type = jnp.select(
            [
                t == MazeStepType.FIRST,
                t == MazeStepType.MID,
                jnp.logical_and(t == MazeStepType.LAST, jnp.allclose(raw_timestep.discount, 0)),
            ],
            [StepType.FIRST, StepType.MID, StepType.TERMINATED],
            default=StepType.TRUNCATED,
        )
        return StateWithKey(raw_timestep, rng_key), TimeStep(
            step_type=step_type,
            reward=raw_timestep.reward.astype(jnp.float32),
            discount=raw_timestep.discount.astype(jnp.float32),
            observation=self.format_obs(raw_timestep),
        )

    def observation_space(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, env_params: bool | None = None
    ) -> Space:
        height, width = self.params(env_params).reset_params.map_init.grid.shape[-3:-1]
        UInt8Space = partial(DiscreteSpace, dtype=jnp.uint8)
        return DictSpace(
            {
                "image": UInt8Space(self.num_objects + 2, shape=(height, width)),
                "task_w": UInt8Space(2, shape=(self.num_objects,)),
                "state_features": UInt8Space(2, shape=(self.num_objects)),
                "position": UInt8Space(max(height, width), shape=(2,)),
                "direction": UInt8Space(len(DIR_TO_VEC)),
                "prev_action": UInt8Space(self._env.num_actions() + 1),
            }
        )

    def action_space(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, env_params: bool | None = None
    ) -> Space:
        return DiscreteSpace(self._env.num_actions())  # pyright: ignore[reportArgumentType]

    def state_space(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, env_params: bool | None = None
    ) -> Space:
        raise NotImplementedError

    def render(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: MazeStep, env_params: bool | None = None
    ):
        video = create_image_from_grid(
            state.observation.image,
            state.observation.position,  # pyright: ignore[reportArgumentType]
            state.observation.direction,  # pyright: ignore[reportArgumentType]
            IMAGE_DICT,
        )
        return video.transpose((2, 0, 1))  # C H W
