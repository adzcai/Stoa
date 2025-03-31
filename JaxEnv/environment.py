import abc
from typing import Tuple
from chex import PRNGKey
from JaxEnv.types import TimeStep, State, Action
from chex import dataclass


@dataclass
class Environment:
  """Abstract base class for JAX-native RL environments."""

  @abc.abstractmethod
  def reset(self, rng_key: PRNGKey) -> TimeStep:
    """Resets the environment to an initial state."""
    pass

  @abc.abstractmethod
  def step(self, state : State, action : Action) -> Tuple[State, TimeStep]:
    """Updates the environment according to the agent's action."""
    pass

  @abc.abstractmethod
  def reward_spec(self):
    """Describes the reward returned by the environment."""
    pass

  @abc.abstractmethod
  def discount_spec(self):
    """Describes the discount returned by the environment."""
    pass

  @abc.abstractmethod
  def observation_spec(self):
    """Defines the structure and bounds of the observation space."""
    pass

  @abc.abstractmethod
  def action_spec(self):
    """Defines the structure and bounds of the action space."""
    pass

  def close(self):
    """Frees any resources used by the environment."""
    pass