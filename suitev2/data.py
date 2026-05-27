"""
data.py — Fashion-MNIST loader + feature extraction for E1.

The dataset is filtered to 4 classes (T-shirt, Trouser, Sneaker, Bag) and
resized to 16x16. For E3 we feed flat 256-d amplitudes to AmplitudeEmbedding;
for E1 we additionally precompute quad_means_8 (8 quadrant means) and gA4
(4 global statistics) per image.

The DataLoader returns a tuple (x_flat, quad_means_8, gA4, y) so the same
loader can serve both encodings — the model picks which fields to use.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from fashion_suite.config import BASELINE


# ============================================================
# Class remap and balanced sampling
# ============================================================

class RemapFashionMNIST(Dataset):
    """Wraps a Subset, applies CLASS_MAP, and returns the precomputed features."""

    def __init__(self, base_dataset, class_map):
        self.base_dataset = base_dataset
        self.class_map = class_map

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]            # x: (1, 16, 16) tensor
        x_flat = x.reshape(-1)                    # (256,)
        quad_means_8 = extract_quad_means_8(x)    # (8,)
        gA4 = extract_gA4(x)                       # (4,)
        return x_flat, quad_means_8, gA4, self.class_map[int(y)]


def make_balanced_indices(dataset, class_map, n_per_class, offset_per_class=0, seed=42):
    """Pick `n_per_class` indices for each class, after shuffling deterministically."""
    buckets = {c: [] for c in class_map.keys()}
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        y = int(y)
        if y in buckets:
            buckets[y].append(idx)

    selected = []
    rng = np.random.default_rng(seed)
    for c in class_map.keys():
        indices = np.array(buckets[c])
        rng.shuffle(indices)
        start = offset_per_class
        end = offset_per_class + n_per_class
        if end > len(indices):
            raise ValueError(
                f"Not enough samples for class {c}: requested {end}, available {len(indices)}."
            )
        selected.extend(indices[start:end].tolist())

    rng.shuffle(selected)
    return selected


# ============================================================
# Feature extraction for E1
# ============================================================

def extract_quad_means_8(x_img: torch.Tensor) -> torch.Tensor:
    """Divide a 16x16 image into a 2x4 grid of 8 blocks (each 8x4 px) and
    return per-block means.

    Args:
      x_img: (1, 16, 16) tensor (single image, grayscale).
    Returns:
      means: (8,) tensor.
    """
    img = x_img.squeeze(0)  # (16, 16)
    blocks = []
    # 2 rows x 4 cols of blocks, each block 8 (rows) x 4 (cols) pixels
    for row in range(2):
        for col in range(4):
            block = img[row * 8:(row + 1) * 8, col * 4:(col + 1) * 4]
            blocks.append(block.mean())
    return torch.stack(blocks)   # (8,)


def extract_gA4(x_img: torch.Tensor) -> torch.Tensor:
    """Compute 4 global statistics of the image:
       [mean, variance, gradient_energy, H_minus_V_asymmetry].

    Args:
      x_img: (1, 16, 16) tensor.
    Returns:
      gA4: (4,) tensor with values centered around 0.
    """
    img = x_img.squeeze(0)  # (16, 16)
    mean = img.mean()
    var = img.var(unbiased=False)

    # Gradient energy: mean squared first-difference
    gx = img[:, 1:] - img[:, :-1]
    gy = img[1:, :] - img[:-1, :]
    grad_energy = (gx.pow(2).mean() + gy.pow(2).mean()) / 2

    # H vs V asymmetry: difference of mean-row-variance and mean-col-variance
    h_var = img.var(dim=1, unbiased=False).mean()
    v_var = img.var(dim=0, unbiased=False).mean()
    hv_asym = h_var - v_var

    # Center mean around 0 (pixel range is [0,1], so subtract 0.5)
    return torch.stack([mean - 0.5, var, grad_energy, hv_asym])


# ============================================================
# Data loading
# ============================================================

_transform = transforms.Compose([
    transforms.Resize((BASELINE.IMG_SIZE, BASELINE.IMG_SIZE)),
    transforms.ToTensor(),
])


def load_data(batch_size=None, seed=42):
    """Return (train_loader, val_loader, test_loader) for the suite.

    Same split sizes as pulito86 (2000/400/300 per class). Uses `seed` only
    to pick balanced indices reproducibly; DataLoader shuffling uses its own RNG.
    """
    if batch_size is None:
        batch_size = BASELINE.BATCH_SIZE

    train_full = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=_transform,
    )
    test_full = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=_transform,
    )

    train_idx = make_balanced_indices(
        train_full, BASELINE.CLASS_MAP, BASELINE.TRAIN_PER_CLASS,
        offset_per_class=0, seed=seed,
    )
    val_idx = make_balanced_indices(
        train_full, BASELINE.CLASS_MAP, BASELINE.VAL_PER_CLASS,
        offset_per_class=BASELINE.TRAIN_PER_CLASS, seed=seed,
    )
    test_idx = make_balanced_indices(
        test_full, BASELINE.CLASS_MAP, BASELINE.TEST_PER_CLASS,
        offset_per_class=0, seed=seed,
    )

    train_ds = RemapFashionMNIST(Subset(train_full, train_idx), BASELINE.CLASS_MAP)
    val_ds   = RemapFashionMNIST(Subset(train_full, val_idx),   BASELINE.CLASS_MAP)
    test_ds  = RemapFashionMNIST(Subset(test_full,  test_idx),  BASELINE.CLASS_MAP)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_test_extension(offset_per_class, n_per_class, seed=42, batch_size=None):
    """Return a DataLoader on a *new, disjoint* chunk of the Fashion-MNIST test set.

    Uses the same per-class shuffle (seed-controlled) as `load_data`, so picking
    `offset_per_class=N, n_per_class=M` returns indices [N:N+M] of the shuffled
    list for each class — disjoint from `load_data`'s test set when
    `N >= base_test_per_class`.

    Caller is responsible for picking `offset_per_class` so it doesn't overlap
    with any chunk already used (typically: cumulative sum of previous chunk
    sizes per class).
    """
    if batch_size is None:
        batch_size = BASELINE.BATCH_SIZE

    test_full = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=_transform,
    )

    indices = make_balanced_indices(
        test_full, BASELINE.CLASS_MAP, n_per_class,
        offset_per_class=offset_per_class, seed=seed,
    )
    test_ds = RemapFashionMNIST(Subset(test_full, indices), BASELINE.CLASS_MAP)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)
