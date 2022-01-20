# Neural Hash Encoding

This is a work in progress reimplementation of [Instant Neural Graphics Primitives](https://github.com/NVlabs/instant-ngp)
Currently this can train an implicit representation of a gigapixel image using a multires hash encoding.

FYI: This is brand new -- most parameters in the training script are hard coded right now

Check out results in [viz](./viz)

## Setup:

Download [the Tokyo image](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837)

```bash
wget -O tokyo.jpg https://live.staticflickr.com/1859/29314390837_b39ae4876e_o_d.jpg
```

Convert to numpy binary format for faster reading (1s w/ .npz vs 14s with .jpg)

```python
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10**10

img = np.asarray(Image.open("tokyo.jpg"))   # Abount 3.5 gb
np.save("tokyo.npy", img)
```

## Train:

```bash
python src/train_image.py
```

# Implementation Notes (From the Paper)

### Architecture

> In all tasks, except for NeRF which we will
> describe later, we use an MLP with two hidden layers that have
> a width of 64 neurons and rectified linear unit (ReLU)

### 4. Initialization

- Initialize hash table entries with uniform distribution [-1e-4, 1e-4]

### 4. Training

- Optimizer
  - Adam: β1 = 0.9, β2 = 0.99, ϵ = 1e−15
  - Learning rate: 1e-2 ([ref: tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/blob/master/samples/mlp_learning_an_image.cu#L130))
- Regularization:
  - L2: 10e-6 Applied to the MLP weigths not the hash table weights

> we skip Adam steps for hash table entries whose gradient
> is exactly 0. This saves ∼10% performance when gradients are sparse
