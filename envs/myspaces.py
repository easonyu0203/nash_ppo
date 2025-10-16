""" Copy from JaxMarl spaces.py, this module contains jittable classes for action and observation spaces. """
from typing import Tuple, Sequence, Union, Dict
from collections import OrderedDict
import chex
import jax
import jax.numpy as jnp

Observation = Union[chex.Array, Dict[str, chex.Array]]

class Space(object):
    """
    Minimal jittable class for abstract jaxmarl space.
    """

    def sample(self, rng: chex.PRNGKey) -> Observation:
        raise NotImplementedError

    def contains(self, x: chex.Numeric) -> bool:
        raise NotImplementedError

class Discrete(Space):
	"""
	Minimal jittable class for discrete gymnax spaces.
	"""

	def __init__(self, num_categories: int, dtype=jnp.int32):
		assert num_categories >= 0
		self.n = num_categories
		self.shape = ()
		self.dtype = dtype

	def sample(self, rng: chex.PRNGKey) -> Observation:
		"""Sample random action uniformly from set of categorical choices."""
		return jax.random.randint(
			rng, shape=self.shape, minval=0, maxval=self.n, dtype=self.dtype
		)

	def contains(self, x: chex.Numeric) -> bool:
		"""Check whether specific object is within space."""
		# type_cond = isinstance(x, self.dtype)
		# shape_cond = (x.shape == self.shape)
		range_cond = jnp.logical_and(x >= 0, x < self.n)
		return bool(range_cond)

class MultiDiscrete(Space):
	"""
	Minimal jittable class for multi-discrete gymnax spaces.
	Each dimension can have a different number of categories.
	"""

	def __init__(self, nvec: Sequence[int], dtype=jnp.int32):
		"""
		Args:
			nvec: Vector of counts of each categorical variable.
				  For example, nvec=[3, 5, 2] means 3 dimensions with
				  categories [0,1,2], [0,1,2,3,4], and [0,1] respectively.
		"""
		self.nvec = jnp.array(nvec)
		assert jnp.all(self.nvec > 0), "All elements of nvec must be positive"
		self.shape = self.nvec.shape
		self.dtype = dtype

	def sample(self, rng: chex.PRNGKey) -> Observation:
		"""Sample random action uniformly from each categorical variable."""
		return jax.random.randint(
			rng, shape=self.shape, minval=0, maxval=self.nvec, dtype=self.dtype
		)

	def contains(self, x: chex.Numeric) -> bool:
		"""Check whether specific object is within space."""
		# type_cond = isinstance(x, self.dtype)
		# shape_cond = (x.shape == self.shape)
		range_cond = jnp.logical_and(
			jnp.all(x >= 0), jnp.all(x < self.nvec)
		)
		return bool(range_cond)

class Box(Space):
	"""
	Minimal jittable class for array-shaped gymnax spaces.
	"""
	def __init__(
		self,
		low: float,
		high: float,
		shape: Tuple[int],
		dtype: jnp.dtype = jnp.float32,
	):
		self.low = low
		self.high = high
		self.shape = shape
		self.dtype = dtype

	def sample(self, rng: chex.PRNGKey) -> Observation:
		"""Sample random action uniformly from 1D continuous range."""
		if jnp.issubdtype(self.dtype, jnp.integer):
			return jax.random.randint(
				rng, shape=self.shape, minval=self.low, maxval=self.high, dtype=self.dtype
			)
		else:
			return jax.random.uniform(
				rng, shape=self.shape, minval=self.low, maxval=self.high, dtype=self.dtype
			)

	def contains(self, x: chex.Numeric) -> bool:
		"""Check whether specific object is within space."""
		# type_cond = isinstance(x, self.dtype)
		# shape_cond = (x.shape == self.shape)
		range_cond = jnp.logical_and(
			jnp.all(x >= self.low), jnp.all(x <= self.high)
		)
		return bool(range_cond)


class Dict(Space):
	"""Minimal jittable class for dictionary of simpler jittable spaces."""
	def __init__(self, spaces: dict):
		self.spaces = spaces
		self.num_spaces = len(spaces)

	def sample(self, rng: chex.PRNGKey) -> Observation:
		"""Sample random action from all subspaces."""
		key_split = jax.random.split(rng, self.num_spaces)
		return OrderedDict(
			[
				(k, self.spaces[k].sample(key_split[i]))
				for i, k in enumerate(self.spaces)
			]
		)

	def contains(self, x: chex.Numeric) -> bool:
		"""Check whether dimensions of object are within subspace."""
		# type_cond = isinstance(x, dict)
		# num_space_cond = len(x) != len(self.spaces)
		# Check for each space individually
		out_of_space = 0
		for k, space in self.spaces.items():
			out_of_space += 1 - space.contains(getattr(x, k))
		return out_of_space == 0
