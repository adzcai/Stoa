"""Lightweight JAX-native specs for RL environments."""

import abc
from typing import Any, Dict, Iterable, NamedTuple, Optional, Sequence, Tuple, Union

from chex import PRNGKey
from jax import Array
import jax
import jax.numpy as jnp
import numpy as np

from JaxEnv.types import Action, TimeStep

class Spec(abc.ABC):
    """Abstract base class for specs that describe RL environment spaces."""

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """The shape of values described by this spec."""
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> jnp.dtype:
        """The dtype of values described by this spec."""
        pass

    @abc.abstractmethod
    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this spec.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A JAX array containing the sampled value.
        """
        pass

    @abc.abstractmethod
    def contains(self, value: Any) -> Array:
        """Checks if a value conforms to this spec.

        Args:
            value: The value to contains.

        Returns:
            a boolean indicating whether the value conforms to the spec.

        Raises:
            ValueError: If the value is not of the correct format.
        """
        pass


class Array(Spec):
    """Describes a JAX array with a specific shape and dtype."""

    def __init__(self, shape: Iterable[int], dtype: Union[jnp.dtype, type], name: str = ""):
        """Initializes a new `Array` spec.

        Args:
            shape: An iterable specifying the array shape.
            dtype: JAX numpy dtype or type specifying the array dtype.
            name: Optional string containing a semantic name for the array.
        """
        self._shape = tuple(int(dim) for dim in shape)
        self._dtype = jnp.dtype(dtype)
        self._name = name

    def __repr__(self) -> str:
        return f"Array(shape={self.shape}, dtype={self.dtype}, name={self._name!r})"

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of values described by this spec."""
        return self._shape

    @property
    def dtype(self) -> jnp.dtype:
        """The dtype of values described by this spec."""
        return self._dtype

    @property
    def name(self) -> str:
        """The name of this spec."""
        return self._name

    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this spec.

        For an unbounded Array, samples from a standard normal distribution.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A JAX array containing the sampled value.
        """
        # For a generic Array, sample from standard normal as it's unbounded
        return jax.random.normal(rng_key, shape=self.shape, dtype=self.dtype)

    def contains(self, value: Any) -> Array:
        """Checks if value conforms to this spec.

        Args:
            value: A value to check.

        Returns:
            A boolean indicating whether the value conforms to the spec.
        """
        return jnp.logical_and(
            jnp.array(value.shape == self.shape),
            jnp.array(value.dtype == self.dtype)
        )


class BoundedArray(Array):
    """Describes a JAX array with minimum and maximum bounds."""

    def __init__(
        self,
        shape: Iterable[int],
        dtype: Union[jnp.dtype, type],
        minimum: Union[float, int, Sequence, np.ndarray, jnp.ndarray],
        maximum: Union[float, int, Sequence, np.ndarray, jnp.ndarray],
        name: str = "",
    ):
        """Initializes a new `BoundedArray` spec.

        Args:
            shape: An iterable specifying the array shape.
            dtype: JAX numpy dtype or type specifying the array dtype.
            minimum: Minimum values (inclusive). Must be broadcastable to `shape`.
            maximum: Maximum values (inclusive). Must be broadcastable to `shape`.
            name: Optional string containing a semantic name for the array.

        Raises:
            ValueError: If `minimum` or `maximum` are not broadcastable to `shape`.
            ValueError: If any values in `minimum` are greater than the corresponding
                value in `maximum`.
        """
        super().__init__(shape, dtype, name)
        
        # Convert to JAX arrays
        self._minimum = jnp.asarray(minimum, dtype=self._dtype)
        self._maximum = jnp.asarray(maximum, dtype=self._dtype)
        
        # Check that minimum and maximum are broadcastable to shape
        try:
            bcast_minimum = jnp.broadcast_to(self._minimum, shape=self._shape)
        except ValueError as e:
            raise ValueError(f"`minimum` is not broadcastable to shape {self._shape}") from e
            
        try:
            bcast_maximum = jnp.broadcast_to(self._maximum, shape=self._shape)
        except ValueError as e:
            raise ValueError(f"`maximum` is not broadcastable to shape {self._shape}") from e
            
        # Check that all minimums are <= maximums
        if jnp.any(bcast_minimum > bcast_maximum):
            raise ValueError(
                f"All values in `minimum` must be <= the corresponding value in `maximum`.\n"
                f"Got minimum={self._minimum} and maximum={self._maximum}"
            )

    def __repr__(self) -> str:
        return (
            f"BoundedArray(shape={self.shape}, dtype={self.dtype}, "
            f"minimum={self.minimum}, maximum={self.maximum}, name={self.name!r})"
        )

    @property
    def minimum(self) -> jnp.ndarray:
        """Minimum values (inclusive)."""
        return self._minimum

    @property
    def maximum(self) -> jnp.ndarray:
        """Maximum values (inclusive)."""
        return self._maximum

    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this spec within the bounds.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A JAX array containing the sampled value.
        """
        # Use uniform sampling between min and max
        return jax.random.uniform(
            rng_key,
            shape=self.shape,
            minval=self.minimum,
            maxval=self.maximum,
            dtype=self.dtype
        )

    def contains(self, value: Any) -> Array:
        """Checks if value conforms to this spec.

        Args:
            value: A value to check.

        Returns:
            A boolean indicating whether the value conforms to the spec.
        """
        # First check if the shape and dtype match
        shape_dtype_valid = super().contains(value)
        
        # Broadcast minimum and maximum to the full shape
        bcast_min = jnp.broadcast_to(self.minimum, self.shape)
        bcast_max = jnp.broadcast_to(self.maximum, self.shape)
        
        # Check if value is within bounds
        in_bounds = jnp.logical_and(
            jnp.all(value >= bcast_min),
            jnp.all(value <= bcast_max)
        )
        
        return jnp.logical_and(shape_dtype_valid, in_bounds)


class DiscreteArray(BoundedArray):
    """Describes a discrete scalar space with values from 0 to num_values-1."""

    def __init__(
        self, 
        num_values: int, 
        dtype: Union[jnp.dtype, type] = jnp.int32, 
        name: str = ""
    ):
        """Initializes a new `DiscreteArray` spec.

        Args:
            num_values: Number of values in the space.
            dtype: JAX numpy dtype. Must be an integer type.
            name: Optional string containing a semantic name for the array.

        Raises:
            ValueError: If `num_values` is not positive.
            ValueError: If `dtype` is not an integer type.
        """
        if num_values <= 0:
            raise ValueError(f"`num_values` must be positive, got {num_values}")
        
        if not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError(f"`dtype` must be an integer type, got {dtype}")
            
        super().__init__(
            shape=(),  # Discrete spaces are scalar
            dtype=dtype,
            minimum=0,
            maximum=num_values - 1,
            name=name
        )
        self._num_values = num_values

    def __repr__(self) -> str:
        return f"DiscreteArray(num_values={self.num_values}, dtype={self.dtype}, name={self.name!r})"

    @property
    def num_values(self) -> int:
        """Number of possible discrete values."""
        return self._num_values

    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this spec.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A JAX array containing the sampled value.
        """
        return jax.random.randint(
            rng_key, 
            shape=self.shape, 
            minval=0, 
            maxval=self.num_values, 
            dtype=self.dtype
        )


class MultiDiscreteArray(BoundedArray):
    """Describes a multi-dimensional discrete space.
    
    Each dimension has its own number of possible values.
    """
    
    def __init__(
        self,
        num_values: Union[Sequence[int], np.ndarray, jnp.ndarray],
        dtype: Union[jnp.dtype, type] = jnp.int32,
        name: str = "",
    ):
        """Initializes a new `MultiDiscreteArray` spec.
        
        Args:
            num_values: Number of values for each dimension.
            dtype: JAX numpy dtype. Must be an integer type.
            name: Optional string containing a semantic name for the array.
            
        Raises:
            ValueError: If any values in `num_values` are not positive.
            ValueError: If `dtype` is not an integer type.
        """
        self._num_values = jnp.asarray(num_values, dtype=jnp.int32)
        
        if jnp.any(self._num_values <= 0):
            raise ValueError(f"All values in `num_values` must be positive, got {num_values}")
            
        if not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError(f"`dtype` must be an integer type, got {dtype}")
            
        super().__init__(
            shape=self._num_values.shape,
            dtype=dtype,
            minimum=jnp.zeros_like(self._num_values),
            maximum=self._num_values - 1,
            name=name
        )
        
    def __repr__(self) -> str:
        return (
            f"MultiDiscreteArray(num_values={self.num_values}, "
            f"dtype={self.dtype}, name={self.name!r})"
        )
    
    @property
    def num_values(self) -> jnp.ndarray:
        """Number of possible values for each dimension."""
        return self._num_values
        
    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this spec.
        
        Args:
            rng_key: A JAX PRNG key.
            
        Returns:
            A JAX array containing the sampled value.
        """
        
        keys = jax.random.split(rng_key, self.shape[0])
        
        # Sample values for all dimensions in a vectorized manner
        return jax.vmap(
            lambda k, max_val: jax.random.randint(
                k, shape=(), minval=0, maxval=max_val, dtype=self.dtype
            )
        )(keys, self.num_values)


class DictSpec(Spec):
    """A dictionary of specs."""
    
    def __init__(self, specs: Dict[str, Spec], name: str = ""):
        """Initializes a new `Dict` spec.
        
        Args:
            specs: A dictionary mapping keys to specs.
            name: Optional string containing a semantic name for the dictionary.
        """
        self._specs = specs
        self._name = name
        
    def __repr__(self) -> str:
        specs_str = ", ".join(f"{k}={v}" for k, v in self._specs.items())
        return f"Dict({specs_str}, name={self._name!r})"
        
    @property
    def name(self) -> str:
        """The name of this spec."""
        return self._name
        
    @property
    def shape(self) -> None:
        """Dict specs don't have a shape."""
        return None
        
    @property
    def dtype(self) -> None:
        """Dict specs don't have a dtype."""
        return None
    
    def sample(self, rng_key: PRNGKey) -> Dict[str, Any]:
        """Sample a random value from this spec.
        
        Args:
            rng_key: A JAX PRNG key.
            
        Returns:
            A dictionary mapping keys to sampled values.
        """
        keys = jax.random.split(rng_key, len(self._specs))
        return {
            key: spec.sample(keys[i]) 
            for i, (key, spec) in enumerate(self._specs.items())
        }
        
    def contains(self, value: Dict[str, Any]) -> Array:
        """Checks if value conforms to this spec.
        
        Args:
            value: A value to check.
            
        Returns:
            A boolean indicating whether the value conforms to the spec.
        """
        if not isinstance(value, dict):
            return jnp.array(False)
            
        if set(value.keys()) != set(self._specs.keys()):
            return jnp.array(False)
            
        # Check each value against its corresponding spec
        results = []
        for key in self._specs:
            results.append(self._specs[key].contains(value[key]))
            
        # All values must conform for the dict to conform
        return jnp.all(jnp.stack(results))
        
    def __getitem__(self, key: str) -> Spec:
        """Gets the spec for a key."""
        return self._specs[key]


class TupleSpec(Spec):
    """A tuple of specs."""
    
    def __init__(self, specs: Sequence[Spec], name: str = ""):
        """Initializes a new `Tuple` spec.
        
        Args:
            specs: A sequence of specs.
            name: Optional string containing a semantic name for the tuple.
        """
        self._specs = tuple(specs)
        self._name = name
        
    def __repr__(self) -> str:
        specs_str = ", ".join(repr(spec) for spec in self._specs)
        return f"Tuple({specs_str}, name={self._name!r})"
        
    @property
    def name(self) -> str:
        """The name of this spec."""
        return self._name
        
    @property
    def shape(self) -> None:
        """Tuple specs don't have a shape."""
        return None
        
    @property
    def dtype(self) -> None:
        """Tuple specs don't have a dtype."""
        return None
    
    def sample(self, rng_key: PRNGKey) -> Tuple[Any, ...]:
        """Sample a random value from this spec.
        
        Args:
            rng_key: A JAX PRNG key.
            
        Returns:
            A tuple of sampled values.
        """
        keys = jax.random.split(rng_key, len(self._specs))
        return tuple(spec.sample(keys[i]) for i, spec in enumerate(self._specs))
        
    def contains(self, value: Tuple[Any, ...]) -> Array:
        """Checks if value conforms to this spec.
        
        Args:
            value: A value to check.
            
        Returns:
            A boolean indicating whether the value conforms to the spec.
        """
        if not isinstance(value, tuple):
            return jnp.array(False)
            
        if len(value) != len(self._specs):
            return jnp.array(False)
            
        # Check each value against its corresponding spec
        results = []
        for i, (spec, val) in enumerate(zip(self._specs, value)):
            results.append(spec.contains(val))
            
        # All values must conform for the tuple to conform
        return jnp.all(jnp.stack(results))
        
    def __getitem__(self, index: int) -> Spec:
        """Gets the spec at an index."""
        return self._specs[index]


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""

    observations: Spec
    actions: Spec
    rewards: Array
    discounts: BoundedArray

    def replace(self, **kwargs) -> 'EnvironmentSpec':
        """Returns a new `EnvironmentSpec` with the specified fields replaced."""
        return self._replace(**kwargs)

    def contains(self, timestep: TimeStep, action: Optional[Action] = None) -> Array:
        """Checks if a timestep and optional action conform to the spec.
        
        Args:
            timestep: A timestep to validate.
            action: An optional action to validate.

        Returns:
            A boolean indicating whether the timestep and action conform to the spec.
        """
        obs_valid = self.observations.contains(timestep.observation)
        rew_valid = self.rewards.contains(timestep.reward)
        disc_valid = self.discounts.contains(timestep.discount)

        # Combine the validations for timestep components
        timestep_valid = jnp.logical_and(obs_valid, jnp.logical_and(rew_valid, disc_valid))

        if action is not None:
            act_valid = self.actions.contains(action)
            return jnp.logical_and(timestep_valid, act_valid)
        else:
            return timestep_valid
        