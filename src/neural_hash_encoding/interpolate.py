from typing import Any
from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

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


@dataclass
@register_pytree_node_class
class Interpolate2D:
    arr: Array

    def __call__(self, x, y, normalized=True):
        if normalized:
            # un-normalize
            y = y * (self.arr.shape[0] - 1)
            x = x * (self.arr.shape[1] - 1)
        return bilinear_interpolate(self.arr, x, y)


    def tree_flatten(self):
        return (self.arr, None)

    @classmethod
    def tree_unflatten(cls, aux_data, data):
        return cls(data)