import numpy as np

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn
from flax.training import train_state
import optax
from jaxopt.tree_util import tree_l2_norm

from torch.utils.data import IterableDataset, DataLoader

import matplotlib.pyplot as plt

from neural_hash_encoding.model import ImageModel


class RandomPixelData(IterableDataset):
    def __init__(self, img, batch_size):
        self.img = img
        H, W, _ = img.shape
        self.batch_size = batch_size
        num_complete_batches, leftover = divmod(H * W, batch_size)
        self._len = num_complete_batches + bool(leftover)

    def __len__(self):
        return self._len

    def __iter__(self):
        for _ in range(len(self)):
            x = np.random.randint(0, W-1, self.batch_size)
            y = np.random.randint(0, H-1, self.batch_size)
            rgb = img[y, x, :]
            xy = np.stack([x / (W-1), y / (H-1)], -1)
            yield xy, rgb / 255


class RandomPixelLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(self.__class__, self).__init__(dataset,
            *args,
            batch_size=None,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            collate_fn=lambda batch: batch,
            pin_memory=False,
            drop_last=False,
            **kwargs)


def PSNR(mse):
    return -10.0 * jnp.log(mse) / jnp.log(10.0)


def mse_loss(preds, targets):
    return jnp.mean((targets - preds)**2)


def relative_mse(preds, targets):
    return jnp.mean((targets - preds)**2 / (preds + 0.01)**2)


def l2_loss(params, l2=1e-6):
    sqnorm = tree_l2_norm(params, squared=True)
    return .5 * l2 * sqnorm


if __name__ == "__main__":
    img = np.load("tokyo.npy")
    img.flags.writeable = False # be safe!
    H, W, C = img.shape
    print(f"Image shape: {img.shape}")
    table_size = 2**22

    def create_train_state(rng, learning_rate):
        """Creates initial `TrainState`."""
        image_model = ImageModel(res=(H, W), table_size=table_size)
        x = jnp.ones((1, 2))    # Dummy data
        params = image_model.init(rng, x)['params']
        tx = optax.adamw(learning_rate, b1=.9, b2=.99, eps=1e-10)
        return train_state.TrainState.create(
            apply_fn=image_model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch):
        """Train for a single step."""
        xy, colors_targ = batch

        def loss_fn(params, weight_decay=1e-6):
            colors_pred = ImageModel((H, W), table_size=table_size).apply({'params': params}, xy)
            mlp_params = params['decoder']
            loss = mse_loss(colors_pred, colors_targ) + l2_loss(mlp_params, weight_decay)
            return loss, colors_pred

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {'loss': loss, 'psnr': PSNR(loss)}
        return state, metrics


    def write_region_plot(path, params, img, s=np.s_[10000:12048, 20000:22048, :]):
        assert len(s) == 3
        crop = img[s]

        xy = jnp.roll(jnp.mgrid[s[:2]], 1, 0).reshape(2, -1).T / jnp.array([W-1, H-1])
        rgb = ImageModel((H, W), table_size=table_size).apply({'params': params}, xy)
        crop2 = (rgb.reshape(*crop.shape) * 255).round(0).clip(0, 255).astype(np.uint8)

        fig, axs = plt.subplots(1, 2, figsize=(12, 16))
        axs[0].imshow(crop)
        axs[0].set_title('Reference')
        axs[1].imshow(crop2)
        axs[1].set_title(f'Encoding')
        fig.savefig(path)


    print("Creating train state ... ")
    rng = jax.random.PRNGKey(420)
    rng, init_rng = jax.random.split(rng)
    learning_rate = 1e-2
    state = create_train_state(init_rng, learning_rate)
    del init_rng

    print("Training ... ")
    epochs = 1
    batch_size = 2**22  # number of pixels
    ds = RandomPixelData(img, batch_size)
    loader = RandomPixelLoader(ds, num_workers=7)
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            step = epoch * len(ds) + i
            batch = (jnp.asarray(batch[0]), jnp.asarray(batch[1]))
            state, metrics = train_step(state, batch)
            loss, psnr = metrics['loss'], metrics['psnr']
            if step > 1 and (np.log10(step) == int(np.log10(step))):   # exponential logging
                path = f'viz/viz_step_{step}_psnr_{psnr:.1f}.png'
                write_region_plot(path, state.params, img)
            if step % 100 == 0:
                print(f'step: {step}, loss (mse): {loss:.4f}, psnr: {psnr:.4f}')