"""Tests for JaxEnv.specs module."""

import unittest
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from JaxEnv import specs
from JaxEnv.types import Action, Discount, Observation, Reward

class ArraySpecTest(unittest.TestCase):
    """Tests for the Array spec."""

    def test_init(self):
        """Test Array initialization."""
        spec = specs.Array((1, 2, 3), jnp.float32, "obs")
        self.assertEqual(spec.shape, (1, 2, 3))
        self.assertEqual(spec.dtype, jnp.float32)
        self.assertEqual(spec.name, "obs")

    def test_sample(self):
        """Test Array sampling."""
        spec = specs.Array((2, 3), jnp.float32)
        key = jax.random.PRNGKey(0)
        sample = spec.sample(key)
        self.assertEqual(sample.shape, (2, 3))
        self.assertEqual(sample.dtype, jnp.float32)

    def test_contains_correct_shape_and_dtype(self):
        """Test Array contains with correct shape and dtype."""
        spec = specs.Array((2, 3), jnp.float32)
        value = jnp.zeros((2, 3), dtype=jnp.float32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertTrue(validated)

    def test_contains_wrong_shape(self):
        """Test Array contains with wrong shape."""
        spec = specs.Array((2, 3), jnp.float32)
        value = jnp.zeros((3, 2), dtype=jnp.float32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_contains_wrong_dtype(self):
        """Test Array contains with wrong dtype."""
        spec = specs.Array((2, 3), jnp.float32)
        value = jnp.zeros((2, 3), dtype=jnp.int32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_repr(self):
        """Test Array representation."""
        spec = specs.Array((1, 2), jnp.float32, "obs")
        self.assertEqual(repr(spec), "Array(shape=(1, 2), dtype=float32, name='obs')")

    def test_jittable(self):
        """Test that Array spec methods are jittable."""
        spec = specs.Array((2, 3), jnp.float32)
        key = jax.random.PRNGKey(0)
        
        # Test jitting sample
        jitted_sample = jax.jit(spec.sample)
        sample = jitted_sample(key)
        self.assertEqual(sample.shape, (2, 3))
        self.assertEqual(sample.dtype, jnp.float32)
        
        # Test jitting contains
        jitted_contains = jax.jit(lambda x: spec.contains(x))
        result = jitted_contains(sample)
        self.assertEqual(result.dtype, jnp.bool_)
        self.assertTrue(result)


class BoundedArraySpecTest(unittest.TestCase):
    """Tests for the BoundedArray spec."""

    def test_init(self):
        """Test BoundedArray initialization."""
        spec = specs.BoundedArray((2, 3), jnp.float32, 0, 1, "bounded_obs")
        self.assertEqual(spec.shape, (2, 3))
        self.assertEqual(spec.dtype, jnp.float32)
        self.assertEqual(spec.name, "bounded_obs")
        np.testing.assert_array_equal(spec.minimum, jnp.zeros((1,), dtype=jnp.float32))
        np.testing.assert_array_equal(spec.maximum, jnp.ones((1,), dtype=jnp.float32))

    def test_init_with_arrays(self):
        """Test BoundedArray initialization with minimum and maximum as arrays."""
        min_val = jnp.array([-1, -2])
        max_val = jnp.array([1, 2])
        spec = specs.BoundedArray((2,), jnp.float32, min_val, max_val)
        np.testing.assert_array_equal(spec.minimum, min_val)
        np.testing.assert_array_equal(spec.maximum, max_val)

    def test_init_invalid_bounds(self):
        """Test BoundedArray initialization with invalid bounds."""
        with self.assertRaises(ValueError):
            specs.BoundedArray((2,), jnp.float32, [1], [0])

    def test_sample(self):
        """Test BoundedArray sampling."""
        spec = specs.BoundedArray((2, 3), jnp.float32, 0, 1)
        key = jax.random.PRNGKey(0)
        sample = spec.sample(key)
        self.assertEqual(sample.shape, (2, 3))
        self.assertEqual(sample.dtype, jnp.float32)
        # Check that samples are within bounds
        self.assertTrue(jnp.all(sample >= 0))
        self.assertTrue(jnp.all(sample <= 1))

    def test_contains_within_bounds(self):
        """Test BoundedArray contains with value within bounds."""
        spec = specs.BoundedArray((2,), jnp.float32, 0, 1)
        value = jnp.array([0.5, 0.5], dtype=jnp.float32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertTrue(validated)

    def test_contains_out_of_bounds(self):
        """Test BoundedArray contains with value out of bounds."""
        spec = specs.BoundedArray((2,), jnp.float32, 0, 1)
        value = jnp.array([0.5, 1.5], dtype=jnp.float32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_repr(self):
        """Test BoundedArray representation."""
        spec = specs.BoundedArray((1, 2), jnp.float32, 0, 1, "bounded_obs")
        self.assertTrue("BoundedArray(shape=(1, 2), dtype=float32" in repr(spec))
        self.assertTrue("name='bounded_obs'" in repr(spec))

    def test_jittable(self):
        """Test that BoundedArray spec methods are jittable."""
        spec = specs.BoundedArray((2, 3), jnp.float32, -1.0, 1.0)
        key = jax.random.PRNGKey(0)
        
        # Test jitting sample
        jitted_sample = jax.jit(spec.sample)
        sample = jitted_sample(key)
        self.assertEqual(sample.shape, (2, 3))
        self.assertEqual(sample.dtype, jnp.float32)
        self.assertTrue(jnp.all(sample >= -1.0))
        self.assertTrue(jnp.all(sample <= 1.0))
        
        # Test jitting contains
        jitted_contains = jax.jit(lambda x: spec.contains(x))
        result = jitted_contains(sample)
        self.assertEqual(result.dtype, jnp.bool_)
        self.assertTrue(result)


class DiscreteArraySpecTest(unittest.TestCase):
    """Tests for the DiscreteArray spec."""

    def test_init(self):
        """Test DiscreteArray initialization."""
        spec = specs.DiscreteArray(5, jnp.int32, "discrete_obs")
        self.assertEqual(spec.shape, ())
        self.assertEqual(spec.dtype, jnp.int32)
        self.assertEqual(spec.name, "discrete_obs")
        self.assertEqual(spec.num_values, 5)
        np.testing.assert_array_equal(spec.minimum, jnp.array(0, dtype=jnp.int32))
        np.testing.assert_array_equal(spec.maximum, jnp.array(4, dtype=jnp.int32))

    def test_init_invalid_num_values(self):
        """Test DiscreteArray initialization with invalid num_values."""
        with self.assertRaises(ValueError):
            specs.DiscreteArray(0)

    def test_init_invalid_dtype(self):
        """Test DiscreteArray initialization with invalid dtype."""
        with self.assertRaises(ValueError):
            specs.DiscreteArray(5, jnp.float32)

    def test_sample(self):
        """Test DiscreteArray sampling."""
        spec = specs.DiscreteArray(5)
        key = jax.random.PRNGKey(0)
        samples = jnp.array([spec.sample(jax.random.fold_in(key, i)) for i in range(100)])
        # Check that samples are within bounds
        self.assertTrue(jnp.all(samples >= 0))
        self.assertTrue(jnp.all(samples < 5))
        # Check that all possible values are sampled (with high probability)
        self.assertEqual(len(jnp.unique(samples)), 5)

    def test_contains_within_bounds(self):
        """Test DiscreteArray contains with value within bounds."""
        spec = specs.DiscreteArray(5)
        value = jnp.array(3, dtype=jnp.int32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertTrue(validated)

    def test_contains_out_of_bounds(self):
        """Test DiscreteArray contains with value out of bounds."""
        spec = specs.DiscreteArray(5)
        value = jnp.array(5, dtype=jnp.int32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_repr(self):
        """Test DiscreteArray representation."""
        spec = specs.DiscreteArray(5, name="discrete_obs")
        self.assertEqual(repr(spec), "DiscreteArray(num_values=5, dtype=int32, name='discrete_obs')")
        
    def test_jittable(self):
        """Test that DiscreteArray spec methods are jittable."""
        spec = specs.DiscreteArray(5)
        key = jax.random.PRNGKey(0)
        
        # Test jitting sample
        jitted_sample = jax.jit(spec.sample)
        sample = jitted_sample(key)
        self.assertEqual(sample.shape, ())
        self.assertEqual(sample.dtype, jnp.int32)
        self.assertTrue(sample >= 0 and sample < 5)
        
        # Test jitting contains
        jitted_contains = jax.jit(lambda x: spec.contains(x))
        value = jnp.array(3, dtype=jnp.int32)
        result = jitted_contains(value)
        self.assertEqual(result.dtype, jnp.bool_)
        self.assertTrue(result)


class MultiDiscreteArraySpecTest(unittest.TestCase):
    """Tests for the MultiDiscreteArray spec."""

    def test_init(self):
        """Test MultiDiscreteArray initialization."""
        spec = specs.MultiDiscreteArray([3, 5, 7], jnp.int32, "multi_discrete_obs")
        self.assertEqual(spec.shape, (3,))
        self.assertEqual(spec.dtype, jnp.int32)
        self.assertEqual(spec.name, "multi_discrete_obs")
        np.testing.assert_array_equal(spec.num_values, jnp.array([3, 5, 7], dtype=jnp.int32))
        np.testing.assert_array_equal(spec.minimum, jnp.zeros((3,), dtype=jnp.int32))
        np.testing.assert_array_equal(spec.maximum, jnp.array([2, 4, 6], dtype=jnp.int32))

    def test_init_invalid_num_values(self):
        """Test MultiDiscreteArray initialization with invalid num_values."""
        with self.assertRaises(ValueError):
            specs.MultiDiscreteArray([3, 0, 7])

    def test_init_invalid_dtype(self):
        """Test MultiDiscreteArray initialization with invalid dtype."""
        with self.assertRaises(ValueError):
            specs.MultiDiscreteArray([3, 5, 7], jnp.float32)

    def test_sample(self):
        """Test MultiDiscreteArray sampling."""
        spec = specs.MultiDiscreteArray([3, 5])
        key = jax.random.PRNGKey(0)
        sample = spec.sample(key)
        self.assertEqual(sample.shape, (2,))
        self.assertEqual(sample.dtype, jnp.int32)
        # Check that samples are within bounds
        self.assertTrue(jnp.all(sample >= jnp.array([0, 0])))
        self.assertTrue(jnp.all(sample <= jnp.array([2, 4])))

    def test_contains_within_bounds(self):
        """Test MultiDiscreteArray contains with value within bounds."""
        spec = specs.MultiDiscreteArray([3, 5])
        value = jnp.array([1, 3], dtype=jnp.int32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertTrue(validated)

    def test_contains_out_of_bounds(self):
        """Test MultiDiscreteArray contains with value out of bounds."""
        spec = specs.MultiDiscreteArray([3, 5])
        value = jnp.array([1, 5], dtype=jnp.int32)
        validated = spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_repr(self):
        """Test MultiDiscreteArray representation."""
        spec = specs.MultiDiscreteArray([3, 5], name="multi_discrete_obs")
        self.assertTrue("MultiDiscreteArray(num_values=" in repr(spec))
        self.assertTrue("name='multi_discrete_obs'" in repr(spec))
        
    def test_jittable(self):
        """Test that MultiDiscreteArray spec methods are jittable."""
        spec = specs.MultiDiscreteArray([3, 5, 7])
        key = jax.random.PRNGKey(0)
        
        # Test jitting sample
        jitted_sample = jax.jit(spec.sample)
        sample = jitted_sample(key)
        self.assertEqual(sample.shape, (3,))
        self.assertEqual(sample.dtype, jnp.int32)
        self.assertTrue(jnp.all(sample >= 0))
        self.assertTrue(jnp.all(sample < jnp.array([3, 5, 7])))
        
        # Test jitting contains
        jitted_contains = jax.jit(lambda x: spec.contains(x))
        value = jnp.array([2, 4, 6], dtype=jnp.int32)
        result = jitted_contains(value)
        self.assertEqual(result.dtype, jnp.bool_)
        self.assertTrue(result)


class DictSpecTest(unittest.TestCase):
    """Tests for the DictSpec spec."""

    def setUp(self):
        """Set up test specs."""
        self.array_spec = specs.Array((2, 3), jnp.float32, "array")
        self.bounded_array_spec = specs.BoundedArray((2,), jnp.float32, 0, 1, "bounded")
        self.specs_dict = {
            "array": self.array_spec,
            "bounded": self.bounded_array_spec,
        }
        self.dict_spec = specs.DictSpec(self.specs_dict, "dict_obs")

    def test_init(self):
        """Test DictSpec initialization."""
        self.assertEqual(self.dict_spec.name, "dict_obs")
        self.assertIsNone(self.dict_spec.shape)
        self.assertIsNone(self.dict_spec.dtype)

    def test_getitem(self):
        """Test DictSpec __getitem__."""
        self.assertIs(self.dict_spec["array"], self.array_spec)
        self.assertIs(self.dict_spec["bounded"], self.bounded_array_spec)

    def test_sample(self):
        """Test DictSpec sampling."""
        key = jax.random.PRNGKey(0)
        sample = self.dict_spec.sample(key)
        self.assertIsInstance(sample, dict)
        self.assertEqual(set(sample.keys()), {"array", "bounded"})
        self.assertEqual(sample["array"].shape, (2, 3))
        self.assertEqual(sample["array"].dtype, jnp.float32)
        self.assertEqual(sample["bounded"].shape, (2,))
        self.assertEqual(sample["bounded"].dtype, jnp.float32)
        self.assertTrue(jnp.all(sample["bounded"] >= 0))
        self.assertTrue(jnp.all(sample["bounded"] <= 1))

    def test_contains_valid(self):
        """Test DictSpec contains with valid dict."""
        value = {
            "array": jnp.zeros((2, 3), dtype=jnp.float32),
            "bounded": jnp.array([0.5, 0.5], dtype=jnp.float32),
        }
        validated = self.dict_spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertTrue(validated)

    def test_contains_invalid_type(self):
        """Test DictSpec contains with invalid type."""
        value = "not a dict"
        validated = self.dict_spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_contains_invalid_keys(self):
        """Test DictSpec contains with invalid keys."""
        value = {
            "array": jnp.zeros((2, 3), dtype=jnp.float32),
            "bounded": jnp.array([0.5, 0.5], dtype=jnp.float32),
            "extra": 1,
        }
        validated = self.dict_spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_contains_invalid_value(self):
        """Test DictSpec contains with invalid value."""
        value = {
            "array": jnp.zeros((2, 3), dtype=jnp.float32),
            "bounded": jnp.array([0.5, 1.5], dtype=jnp.float32),  # out of bounds
        }
        validated = self.dict_spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_repr(self):
        """Test DictSpec representation."""
        self.assertTrue("Dict(array=" in repr(self.dict_spec))
        self.assertTrue("bounded=" in repr(self.dict_spec))
        self.assertTrue("name='dict_obs'" in repr(self.dict_spec))
        
    def test_jittable(self):
        """Test that DictSpec spec methods are jittable."""
        spec = specs.DictSpec({
            "array": specs.Array((2,), jnp.float32),
            "bounded": specs.BoundedArray((2,), jnp.float32, 0.0, 1.0),
        })
        key = jax.random.PRNGKey(0)
        
        # Test jitting sample
        jitted_sample = jax.jit(spec.sample)
        sample = jitted_sample(key)
        self.assertIsInstance(sample, dict)
        self.assertEqual(set(sample.keys()), {"array", "bounded"})
        
        # Test jitting contains
        jitted_contains = jax.jit(lambda x: spec.contains(x))
        result = jitted_contains(sample)
        self.assertEqual(result.dtype, jnp.bool_)
        self.assertTrue(result)


class TupleSpecTest(unittest.TestCase):
    """Tests for the TupleSpec spec."""

    def setUp(self):
        """Set up test specs."""
        self.array_spec = specs.Array((2, 3), jnp.float32, "array")
        self.bounded_array_spec = specs.BoundedArray((2,), jnp.float32, 0, 1, "bounded")
        self.specs_tuple = (self.array_spec, self.bounded_array_spec)
        self.tuple_spec = specs.TupleSpec(self.specs_tuple, "tuple_obs")

    def test_init(self):
        """Test TupleSpec initialization."""
        self.assertEqual(self.tuple_spec.name, "tuple_obs")
        self.assertIsNone(self.tuple_spec.shape)
        self.assertIsNone(self.tuple_spec.dtype)

    def test_getitem(self):
        """Test TupleSpec __getitem__."""
        self.assertIs(self.tuple_spec[0], self.array_spec)
        self.assertIs(self.tuple_spec[1], self.bounded_array_spec)

    def test_sample(self):
        """Test TupleSpec sampling."""
        key = jax.random.PRNGKey(0)
        sample = self.tuple_spec.sample(key)
        self.assertIsInstance(sample, tuple)
        self.assertEqual(len(sample), 2)
        self.assertEqual(sample[0].shape, (2, 3))
        self.assertEqual(sample[0].dtype, jnp.float32)
        self.assertEqual(sample[1].shape, (2,))
        self.assertEqual(sample[1].dtype, jnp.float32)
        self.assertTrue(jnp.all(sample[1] >= 0))
        self.assertTrue(jnp.all(sample[1] <= 1))

    def test_contains_valid(self):
        """Test TupleSpec contains with valid tuple."""
        value = (
            jnp.zeros((2, 3), dtype=jnp.float32),
            jnp.array([0.5, 0.5], dtype=jnp.float32),
        )
        validated = self.tuple_spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertTrue(validated)

    def test_contains_invalid_type(self):
        """Test TupleSpec contains with invalid type."""
        value = "not a tuple"
        validated = self.tuple_spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_contains_invalid_length(self):
        """Test TupleSpec contains with invalid length."""
        value = (jnp.zeros((2, 3), dtype=jnp.float32),)  # Too short
        validated = self.tuple_spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_contains_invalid_value(self):
        """Test TupleSpec contains with invalid value."""
        value = (
            jnp.zeros((2, 3), dtype=jnp.float32),
            jnp.array([0.5, 1.5], dtype=jnp.float32),  # Out of bounds
        )
        validated = self.tuple_spec.contains(value)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_repr(self):
        """Test TupleSpec representation."""
        self.assertTrue("Tuple(" in repr(self.tuple_spec))
        self.assertTrue("name='tuple_obs'" in repr(self.tuple_spec))
        
    def test_jittable(self):
        """Test that TupleSpec spec methods are jittable."""
        spec = specs.TupleSpec([
            specs.Array((2,), jnp.float32),
            specs.BoundedArray((2,), jnp.float32, 0.0, 1.0),
        ])
        key = jax.random.PRNGKey(0)
        
        # Test jitting sample
        jitted_sample = jax.jit(spec.sample)
        sample = jitted_sample(key)
        self.assertIsInstance(sample, tuple)
        self.assertEqual(len(sample), 2)
        
        # Test jitting contains
        jitted_contains = jax.jit(lambda x: spec.contains(x))
        result = jitted_contains(sample)
        self.assertEqual(result.dtype, jnp.bool_)
        self.assertTrue(result)


class TimeStepTest(NamedTuple):
    """A simplified TimeStep for testing."""
    observation: Observation
    reward: Reward
    discount: Discount
    action: Action = None


class EnvironmentSpecTest(unittest.TestCase):
    """Tests for the EnvironmentSpec."""

    def setUp(self):
        """Set up test specs."""
        self.observation_spec = specs.Array((2, 3), jnp.float32, "observation")
        self.action_spec = specs.DiscreteArray(5, name="action")
        self.reward_spec = specs.Array((), jnp.float32, "reward")
        self.discount_spec = specs.BoundedArray((), jnp.float32, 0.0, 1.0, "discount")
        self.env_spec = specs.EnvironmentSpec(
            observations=self.observation_spec,
            actions=self.action_spec,
            rewards=self.reward_spec,
            discounts=self.discount_spec,
        )

    def test_replace(self):
        """Test EnvironmentSpec replace."""
        new_action_spec = specs.DiscreteArray(10, name="new_action")
        new_spec = self.env_spec.replace(actions=new_action_spec)
        self.assertIs(new_spec.observations, self.observation_spec)
        self.assertIs(new_spec.actions, new_action_spec)
        self.assertIs(new_spec.rewards, self.reward_spec)
        self.assertIs(new_spec.discounts, self.discount_spec)

    def test_contains_valid(self):
        """Test EnvironmentSpec contains with valid timestep."""
        timestep = TimeStepTest(
            observation=jnp.zeros((2, 3), dtype=jnp.float32),
            reward=jnp.array(0.5, dtype=jnp.float32),
            discount=jnp.array(0.9, dtype=jnp.float32),
        )
        validated = self.env_spec.contains(timestep)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertTrue(validated)

    def test_contains_valid_with_action(self):
        """Test EnvironmentSpec contains with valid timestep and action."""
        timestep = TimeStepTest(
            observation=jnp.zeros((2, 3), dtype=jnp.float32),
            reward=jnp.array(0.5, dtype=jnp.float32),
            discount=jnp.array(0.9, dtype=jnp.float32),
        )
        action = jnp.array(3, dtype=jnp.int32)  # Valid for DiscreteArray(5)
        validated = self.env_spec.contains(timestep, action)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertTrue(validated)

    def test_contains_invalid_observation(self):
        """Test EnvironmentSpec contains with invalid observation."""
        timestep = TimeStepTest(
            observation=jnp.zeros((3, 2), dtype=jnp.float32),  # Wrong shape
            reward=jnp.array(0.5, dtype=jnp.float32),
            discount=jnp.array(0.9, dtype=jnp.float32),
        )
        validated = self.env_spec.contains(timestep)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_contains_invalid_reward(self):
        """Test EnvironmentSpec contains with invalid reward."""
        timestep = TimeStepTest(
            observation=jnp.zeros((2, 3), dtype=jnp.float32),
            reward=jnp.array([0.5], dtype=jnp.float32),  # Wrong shape
            discount=jnp.array(0.9, dtype=jnp.float32),
        )
        validated = self.env_spec.contains(timestep)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_contains_invalid_discount(self):
        """Test EnvironmentSpec contains with invalid discount."""
        timestep = TimeStepTest(
            observation=jnp.zeros((2, 3), dtype=jnp.float32),
            reward=jnp.array(0.5, dtype=jnp.float32),
            discount=jnp.array(1.1, dtype=jnp.float32),  # Out of bounds
        )
        validated = self.env_spec.contains(timestep)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)

    def test_contains_invalid_action(self):
        """Test EnvironmentSpec contains with invalid action."""
        timestep = TimeStepTest(
            observation=jnp.zeros((2, 3), dtype=jnp.float32),
            reward=jnp.array(0.5, dtype=jnp.float32),
            discount=jnp.array(0.9, dtype=jnp.float32),
        )
        action = jnp.array(5, dtype=jnp.int32)  # Out of bounds for DiscreteArray(5)
        validated = self.env_spec.contains(timestep, action)
        self.assertEqual(validated.shape, ())
        self.assertEqual(validated.dtype, jnp.bool_)
        self.assertFalse(validated)
        
    def test_jittable(self):
        """Test that EnvironmentSpec methods are jittable."""
        # Create a valid timestep and action
        timestep = TimeStepTest(
            observation=jnp.zeros((2, 3), dtype=jnp.float32),
            reward=jnp.array(0.5, dtype=jnp.float32),
            discount=jnp.array(0.9, dtype=jnp.float32),
        )
        action = jnp.array(3, dtype=jnp.int32)
        
        # Test jitting contains
        jitted_contains = jax.jit(lambda ts, act: self.env_spec.contains(ts, act))
        result = jitted_contains(timestep, action)
        self.assertEqual(result.dtype, jnp.bool_)
        self.assertTrue(result)
        
        # Test jitting contains with just timestep
        jitted_contains_ts = jax.jit(lambda ts: self.env_spec.contains(ts))
        result_ts = jitted_contains_ts(timestep)
        self.assertEqual(result_ts.dtype, jnp.bool_)
        self.assertTrue(result_ts)


if __name__ == "__main__":
    unittest.main() 