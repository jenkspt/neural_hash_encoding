from typing import Any, Sequence
from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import operator
import itertools
import functools
from jax._src.scipy.ndimage import (
    _nonempty_prod,
    _nonempty_sum,
    _INDEX_FIXERS,
    _round_half_away_from_zero,
    _nearest_indices_and_weights,
    _linear_indices_and_weights,
)

Array = Any


def bilinear_interpolate(arr, x, y, clip_to_bounds=False):
    assert len(arr.shape) == 3
    H, W, _ = arr.shape

    x = jnp.asarray(x)
    y = jnp.asarray(y)

    x0 = jnp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(int)
    y1 = y0 + 1

    if clip_to_bounds:
        x0 = jnp.clip(x0, 0, W-1)
        x1 = jnp.clip(x1, 0, W-1)
        y0 = jnp.clip(y0, 0, H-1)
        y1 = jnp.clip(y1, 0, H-1)

    Ia = arr[y0, x0, :]
    Ib = arr[y1, x0, :]
    Ic = arr[y0, x1, :]
    Id = arr[y1, x1, :]

    wa = ((x1-x) * (y1-y))[..., None]
    wb = ((x1-x) * (y-y0))[..., None]
    wc = ((x-x0) * (y1-y))[..., None]
    wd = ((x-x0) * (y-y0))[..., None]

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def map_coordinates(input, coordinates, order, mode='constant', cval=0.0):
	"""
	Adapted from jax.scipy.map_coordinates, but with a few key differences.

	1.) interpolations are always broadcasted along the last dimension of the `input`
	i.e. a 3 channel rgb image with shape [H, W, 3] will be interpolated with 2d
	coordinates and broadcasted across the channel dimension

	2.) `input` isn't required to be jax `DeviceArray` -- it can be any type that
	supports numpy fancy indexing

	Note on interpolation: `map_coordinates` indexes in the order of the axes,
	so for an image it indexes the coordinates as [y, x]
	"""

	coordinates = [jnp.asarray(c) for c in coordinates]
	cval = jnp.asarray(cval, input.dtype)

	if len(coordinates) != input.ndim-1:
		raise ValueError('coordinates must be a sequence of length input.ndim - 1, but '
                     '{} != {}'.format(len(coordinates), input.ndim - 1))

	index_fixer = _INDEX_FIXERS.get(mode)
	if index_fixer is None:
		raise NotImplementedError(
			'map_coordinates does not support mode {}. '
			'Currently supported modes are {}.'.format(mode, set(_INDEX_FIXERS)))

	if mode == 'constant':
		is_valid = lambda index, size: (0 <= index) & (index < size)
	else:
		is_valid = lambda index, size: True

	if order == 0:
		interp_fun = _nearest_indices_and_weights
	elif order == 1:
		interp_fun = _linear_indices_and_weights
	else:
		raise NotImplementedError(
			'map_coordinates currently requires order<=1')

	valid_1d_interpolations = []
	for coordinate, size in zip(coordinates, input.shape[:-1]):
		interp_nodes = interp_fun(coordinate)
		valid_interp = []
		for index, weight in interp_nodes:
			fixed_index = index_fixer(index, size)
			valid = is_valid(index, size)
			valid_interp.append((fixed_index, valid, weight))
		valid_1d_interpolations.append(valid_interp)

	outputs = []
	for items in itertools.product(*valid_1d_interpolations):
		indices, validities, weights = zip(*items)
		if all(valid is True for valid in validities):
			# fast path
			contribution = input[(*indices, Ellipsis)]
		else:
			all_valid = functools.reduce(operator.and_, validities)
			contribution = jnp.where(all_valid[..., None], input[(*indices, Ellipsis)], cval)
		outputs.append(_nonempty_prod(weights)[..., None] * contribution)

	result = _nonempty_sum(outputs)
	if jnp.issubdtype(input.dtype, jnp.integer):
		result = _round_half_away_from_zero(result)
	return result.astype(input.dtype)


@dataclass
@register_pytree_node_class
class Interpolate:
	arr: Array
	order: int
	mode: str
	cval: float = 0.0

	def __call__(self, coords, normalized=True):
		coords = [jnp.asarray(c) for c in coords]
		assert len(coords) == (self.arr.ndim - 1)
		if normalized:
			# un-normalize
			coords = [c * (s-1) for c, s in zip(coords, self.arr.shape)]
		return map_coordinates(self.arr, coords, order=self.order, mode=self.mode, cval=self.cval)

	def tree_flatten(self):
		return (self.arr, None)

	@classmethod
	def tree_unflatten(cls, aux_data, data):
		return cls(data)