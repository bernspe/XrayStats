"""Qualitätskontrolle (Spec Phase 1): Monotonie, robuste Ausreißer-z-Werte."""
from __future__ import annotations

import numpy as np

from . import SPINE_LABELS


def craniocaudal_monotonic(centroids_img: dict) -> bool:
    """Bildkoordinaten: y muss mit dem Label (kranial→kaudal) streng zunehmen."""
    y = [centroids_img[k][1] for k in SPINE_LABELS if k in centroids_img]
    return bool(np.all(np.diff(y) > 0))


def robust_z(x) -> np.ndarray:
    """(x − Median) / (1,4826·MAD); MAD = 0 → 0 statt ∞."""
    x = np.asarray(x, dtype=float)
    mad = 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))
    return np.zeros_like(x) if not mad else (x - np.nanmedian(x)) / mad
