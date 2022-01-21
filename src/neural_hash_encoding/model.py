from typing import Tuple, Callable, Any, Iterable
from dataclasses import field

import jax
import jax.numpy as jnp
from flax import linen as nn

from neural_hash_encoding.hash_array import HashArray, _get_level_res_nd
from neural_hash_encoding.interpolate import Interpolate

# Copied from flax
PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any

def uniform_init(minval=0, maxval=0.01, dtype=jnp.float64):
    def init(key, shape, dtype=dtype):
        return jax.random.uniform(key, shape, dtype, minval, maxval)
    return init


class DenseEncodingLevel(nn.Module):
    res: Shape
    features: int = 2
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    table_init: Callable[[PRNGKey, Shape, Dtype], Array] = uniform_init(-1e-4, 1e-4)

    interp: Interpolate = field(init=False)

    def setup(self):
        array = self.param('table',
                           self.table_init,
                           (*self.res, self.features),
                           self.param_dtype)
        self.interp = Interpolate(jnp.asarray(array), order=1, mode='nearest')
    
    def __call__(self, coords):
        assert len(coords) == (self.interp.arr.ndim - 1)
        return self.interp(coords, normalized=True)


class HashEncodingLevel(nn.Module):
    res: Shape
    features: int = 2
    table_size: int = 2**14
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    table_init: Callable[[PRNGKey, Shape, Dtype], Array] = uniform_init(-1e-4, 1e-4)

    interp: Interpolate = field(init=False)

    def setup(self):
        table = self.param('table',
                           self.table_init,
                           (self.table_size, self.features),
                           self.param_dtype)
        shape = (*self.res, self.features)
        array = HashArray(jnp.asarray(table), shape)
        self.interp = Interpolate(array, order=1, mode='nearest')

    def __call__(self, coords):
        assert len(coords) == (self.interp.arr.ndim - 1)
        return self.interp(coords, normalized=True)


class MultiResEncoding(nn.Module):
    levels: int=16
    table_size: int = 2**14
    features: int = 2
    minres: Shape = (16, 16)
    maxres: Shape = (512, 512)
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    param_init: Callable[[PRNGKey, Shape, Dtype], Array] = uniform_init(-1e-4, 1e-4)

    L: Tuple[nn.Module, ...] = field(init=False)

    def setup(self):
        res_levels = _get_level_res_nd(self.levels, self.minres, self.maxres)
        kwargs = dict(
            features=self.features, dtype=self.dtype,
            param_dtype=self.param_dtype, table_init=self.param_init)
        # First level is always dense
        L0 = DenseEncodingLevel(res_levels[0], **kwargs)
        # Rest are sparse hash arrays
        self.L = tuple([L0, *(HashEncodingLevel(l, table_size=self.table_size, **kwargs) for l in res_levels[1:])])

    def __call__(self, coords):
        features = [l(coords) for l in self.L]
        features = jnp.concatenate(features, -1)
        return features


class MLP(nn.Module):
    features: Tuple[int, ...] = (64, 64, 3)

    @nn.compact
    def __call__(self, x):
        assert len(self.features) >= 2
        *hidden, linear = self.features
        for h in hidden:
            x = nn.relu(nn.Dense(h)(x))
        x = nn.Dense(linear)(x)
        return x


class ImageModel(nn.Module):
    res: Shape
    channels: int=3
    levels: int=16
    table_size: int = 2**14
    features: int = 2
    minres: Shape = (16, 16)

    def setup(self):
        self.embedding = MultiResEncoding(self.levels, self.table_size,
                                            self.features, self.minres, self.res)
        self.decoder = MLP((64, 64, self.channels))
    
    def __call__(self, coords):
        features = self.embedding(coords)
        color = self.decoder(features)
        return color