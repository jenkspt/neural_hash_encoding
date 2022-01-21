from typing import Tuple, Any, Iterable
from functools import reduce
import operator
import numpy as np
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

"""
Mutli-res Grid code from NVIDIA Paper:
https://github.com/NVlabs/tiny-cuda-nn/blob/d0639158dd64b5d146d659eb881702304abaaa24/include/tiny-cuda-nn/encodings/grid.h
"""

Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any

PRIMES = (1, 73856093, 19349663, 83492791)


@register_pytree_node_class
@dataclass
class HashArray:
    """
    This is a sparse array backed by simple hash table. It minimally implements an array
    interface as to be used for (nd) linear interpolation.
    There is no collision resolution or even bounds checking.

    Attributes:
      data: The hash table represented as a 2D array.
        First dim is indexed with the hash index and second dim is the feature
      shape: The shape of the array.

    NVIDIA Implementation of multi-res hash grid:
    https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h#L66-L80
    """
    data: Array
    shape: Shape

    def __post_init__(self):
        assert self.data.ndim == 2, "Hash table data should be 2d"
        assert self.data.shape[1] == self.shape[-1]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.data.dtype

    def spatial_hash(self, coords):
        assert len(coords) <= len(PRIMES), "Add more PRIMES!"
        if len(coords) == 1:
            i = (coords[0] ^ PRIMES[1])
        else:
            i = reduce(operator.xor, (c * p for c, p in zip(coords, PRIMES)))
        return i % self.data.shape[0]

    def __getitem__(self, i):
        *spatial_i, feature_i = i if len(i) == self.ndim else (*i, Ellipsis)
        i = self.spatial_hash(spatial_i)
        return self.data[i, feature_i]
    
    def __array__(self, dtype=None):
        H, W, _ = self.shape
        y, x = jnp.mgrid[0:H:1, 0:W:1]
        arr = self[y, x, :].__array__(dtype)
        return arr

    def __repr__(self):
        return "HashArray(" + str(np.asarray(self)) + ")"

    def tree_flatten(self):
        return (self.data, self.shape)

    @classmethod
    def tree_unflatten(cls, shape, data):
        return cls(data, shape)


def growth_factor(levels: int, minres: int, maxres: int):
    return np.exp((np.log(maxres) - np.log(minres)) / (levels - 1))


def _get_level_res(levels: int, minres: int, maxres: int):
    b = growth_factor(levels, minres, maxres)
    res = [int(round(minres * (b ** l))) for l in range(0, levels)]
    return res


def _get_level_res_nd(levels: int, minres: Tuple[int, ...], maxres: Tuple[int, ...]):
    it = (_get_level_res(levels, _min, _max) \
        for _min, _max in zip(minres, maxres))
    return list(zip(*it))