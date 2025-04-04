# normalizer.py

import numpy as np

class RunningNormalizer:
    """
    A simple class for Welford's online mean/variance estimation plus
    normalization and optional clipping.
    """
    def __init__(self, dim, clip_value=5.0):
        self.mean = np.zeros(dim, dtype=np.float32)
        self.var = np.zeros(dim, dtype=np.float32)
        self.count = 0
        self.clip_value = clip_value

    def update(self, x: np.ndarray):
        """
        Update running mean/var with a single observation 'x'
        or a batch of observations shape [N, dim].
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]  # shape (1, dim)

        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.var += delta * (row - self.mean)

    def normalize(self, x: np.ndarray, update_stats: bool = False):
        """
        Normalize 'x' using current mean and var, optionally updating stats.
        Clip to [-clip_value, clip_value].
        """
        if update_stats:
            self.update(x)

        std = np.sqrt(self.var / max(self.count, 1) + 1e-8)
        z = (x - self.mean) / std
        z = np.clip(z, -self.clip_value, self.clip_value)
        return z
