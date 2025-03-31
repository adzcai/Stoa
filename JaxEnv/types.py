from typing import Dict, Optional
from chex import ArrayTree, dataclass
import jax.numpy as jnp
from jax import Array

State = ArrayTree
Reward = Array
Discount = Array
Observation = ArrayTree
Action = ArrayTree
TimeStepExtras = Optional[Dict[str, ArrayTree]]

class StepType(jnp.int8):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST : Array = jnp.array(0, dtype=jnp.int8)
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID : Array = jnp.array(1, dtype=jnp.int8)
  # Denotes the last `TimeStep` in a sequence that was terminated.
  TERMINATED : Array = jnp.array(2, dtype=jnp.int8)
  # Denotes the last `TimeStep` in a sequence that was truncated.
  TRUNCATED : Array = jnp.array(3, dtype=jnp.int8)


@dataclass
class TimeStep:
  step_type: StepType
  reward: Reward
  discount: Discount
  observation: Observation
  extras: TimeStepExtras = None

  def first(self) -> bool:
    return self.step_type == StepType.FIRST

  def mid(self) -> bool:
    return self.step_type == StepType.MID

  def last(self) -> bool:
    return jnp.logical_or(self.step_type == StepType.TERMINATED, self.step_type == StepType.TRUNCATED)
  
  def terminated(self) -> bool:
    return self.step_type == StepType.TERMINATED

  def truncated(self) -> bool:
    return self.step_type == StepType.TRUNCATED
