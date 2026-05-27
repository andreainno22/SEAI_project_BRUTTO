"""
data.py â€” Fashion-MNIST loader + feature extraction for E1.

The dataset is filtered to 4 classes (T-shirt, Trouser, Sneaker, Bag) and
resized to 16x16. For E3 we feed flat 256-d amplitudes to AmplitudeEmbedding;
for E1 we additionally precompute quad_means_8 (8 quadrant means) and gA4
(4 global statistics) per image.

The DataLoader returns a tuple (x_flat, quad_means_8, gA4, y) so the same
loader can serve both encodings â€” the model picks which fields to use.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from suitev2.config import BASELINE


# ============================================================
# Class remap and balanced sampling
# ============================================================

class RemapFashionMNIST(Dataset):
    """Wraps a Subset, applies CLASS_MAP, and returns the precomputed features.

    Args:
      base_dataset:    The primary Subset (possibly with augmentation transforms).
                       Its images are used for x_flat (the raw pixel vector sent to E3/custom).
      class_map:       Dict mapping original label â†’ remapped label.
      feature_dataset: Optional Subset with NO augmentation, sharing the same indices as
                       base_dataset.  When provided, quad_means_8 and gA4 are computed
                       from the clean (non-augmented) image rather than from the augmented
                       one.  This decouples stochastic data-augmentation (beneficial for E3)
                       from the deterministic statistical features used by E1 (which must
                       remain stable across epochs for a_embed/c_embed to learn correctly).
                       Leave None for val/test sets, where augmentation is never applied.
    """

    def __init__(self, base_dataset, class_map, feature_dataset=None):
        self.base_dataset    = base_dataset
        self.feature_dataset = feature_dataset   # clean images for E1 features (or None)
        self.class_map       = class_map

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y   = self.base_dataset[idx]          # x: (1, 16, 16), possibly augmented
        x_flat = x.reshape(-1)                    # (256,)  â€” used by E3 / custom

        # Use the clean (non-augmented) image for statistical features so that
        # E1 embedding parameters (a_embed, c_embed) see stable per-pixel statistics.
        feat_img = x
        if self.feature_dataset is not None:
            feat_img, _ = self.feature_dataset[idx]

        quad_means_8 = extract_quad_means_8(feat_img)   # (8,)
        gA4          = extract_gA4(feat_img)             # (4,)
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
       [mean-0.5, variance, gradient_energy, normalised_HV_asymmetry].

    Args:
      x_img: (1, 16, 16) tensor (grayscale, pixels in [0, 1]).
    Returns:
      gA4: (4,) tensor.
        [0] mean - 0.5          â€” centred mean, in [-0.5, 0.5]
        [1] variance            â€” in [0, 0.25]
        [2] gradient energy     â€” mean squared first-difference, in [0, 1]
        [3] normalised H-V asym â€” (h_var - v_var) / (h_var + v_var + eps),
                                  in [-1, 1]; normalisation removes the
                                  dependency on overall image intensity.
    """
    img = x_img.squeeze(0)  # (16, 16)
    mean = img.mean()
    var  = img.var(unbiased=False)

    # Gradient energy: mean squared first-difference
    gx = img[:, 1:] - img[:, :-1]
    gy = img[1:, :] - img[:-1, :]
    grad_energy = (gx.pow(2).mean() + gy.pow(2).mean()) / 2

    # Normalised H vs V asymmetry ([-1, 1]); avoids magnitude dependence.
    h_var   = img.var(dim=1, unbiased=False).mean()  # mean row-variance
    v_var   = img.var(dim=0, unbiased=False).mean()  # mean col-variance
    hv_asym = (h_var - v_var) / (h_var + v_var + 1e-8)

    return torch.stack([mean - 0.5, var, grad_energy, hv_asym])


# ============================================================
# Data loading
# ============================================================

_transform = transforms.Compose([
    transforms.Resize((BASELINE.IMG_SIZE, BASELINE.IMG_SIZE)),
    transforms.ToTensor(),
])

_train_transform = transforms.Compose([
    transforms.Resize((BASELINE.IMG_SIZE, BASELINE.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(BASELINE.IMG_SIZE, padding=2),
    transforms.ToTensor(),
])


def load_data(batch_size=None, seed=42):
    """Return (train_loader, val_loader, test_loader) for the suite.

    Same split sizes as pulito86 (2000/400/300 per class). Uses `seed` only
    to pick balanced indices reproducibly; DataLoader shuffling uses its own RNG.
    """
    if batch_size is None:
        batch_size = BASELINE.BATCH_SIZE

    train_full_aug = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=_train_transform,
    )
    train_full_val = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=_transform,
    )
    test_full = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=_transform,
    )

    train_idx = make_balanced_indices(
        train_full_val, BASELINE.CLASS_MAP, BASELINE.TRAIN_PER_CLASS,
        offset_per_class=0, seed=seed,
    )
    val_idx = make_balanced_indices(
        train_full_val, BASELINE.CLASS_MAP, BASELINE.VAL_PER_CLASS,
        offset_per_class=BASELINE.TRAIN_PER_CLASS, seed=seed,
    )
    test_idx = make_balanced_indices(
        test_full, BASELINE.CLASS_MAP, BASELINE.TEST_PER_CLASS,
        offset_per_class=0, seed=seed,
    )

    # For the training set we decouple augmentation from feature extraction:
    #   x_flat  â†’ augmented image (RandomHorizontalFlip + RandomCrop) for E3/custom
    #   E1 features (quad_means_8, gA4) â†’ clean image (no augmentation)
    # This stabilises the statistical features seen by a_embed/c_embed while
    # still providing input-space regularisation for the amplitude encoding.
    train_ds = RemapFashionMNIST(
        Subset(train_full_aug, train_idx), BASELINE.CLASS_MAP,
        feature_dataset=Subset(train_full_val, train_idx),
    )
    val_ds   = RemapFashionMNIST(Subset(train_full_val, val_idx),  BASELINE.CLASS_MAP)
    test_ds  = RemapFashionMNIST(Subset(test_full, test_idx),      BASELINE.CLASS_MAP)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_test_extension(offset_per_class, n_per_class, seed=42, batch_size=None):
    """Return a DataLoader on a *new, disjoint* chunk of the Fashion-MNIST test set.

    Uses the same per-class shuffle (seed-controlled) as `load_data`, so picking
    `offset_per_class=N, n_per_class=M` returns indices [N:N+M] of the shuffled
    list for each class â€” disjoint from `load_data`'s test set when
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

