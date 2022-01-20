from typing import Tuple, Any
import numpy as np
from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

"""
Mutli-res Grid code from NVIDIA Paper:
https://github.com/NVlabs/tiny-cuda-nn/blob/d0639158dd64b5d146d659eb881702304abaaa24/include/tiny-cuda-nn/encodings/grid.h
"""

Array = Any

PRIMES = (73856093, 19349663)

@register_pytree_node_class
@dataclass
class HashArray2D:
    data: Array
    shape: Tuple[int, int]

    """
    https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h#L66-L80
    """
    def spatial_hash(self, y, x):
        return (x ^ (y * PRIMES[0])) % self.data.shape[-2]

    def __getitem__(self, i):
        x, y, d = i[-3:] if len(i) == 3 else (*i[-2:], Ellipsis)
        i = self.spatial_hash(y, x)
        return self.data[i, d]
    
    def __array__(self, dtype=None):
        H, W, _ = self.shape
        y, x = jnp.mgrid[0:H:1, 0:W:1]
        arr = self[y, x, :].__array__(dtype)
        return arr

    def __repr__(self):
        return "HashArray2D(" + str(np.asarray(self)) + ")"

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