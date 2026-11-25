"""Koronare Winkel aus Centroiden (Spec §4.5). Heißen `centroid_cobb`, nie nur `cobb`:
sie messen die Lage der Wirbelkörper (ähnlich Ferguson), nicht die Deckplattenneigung.

Eingaben: Patientenkoordinaten (x = Patient links +, y = kranial +), in WKD für Methode (b).
Neigung θ = Winkel der kranialwärts zeigenden Wirbelsäulenrichtung zur Vertikalen,
positiv = kranial nach Patient links geneigt.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_smoothing_spline

from . import SPINE_LABELS

# ponytail: fester Glättungsparameter λ (natürlicher kubischer Glättungsspline, y in WKD) für
# alle Visiten; λ = 0,1 glättet ~0,2° an den Rändern einer 30°-S-Kurve weg. Phase 2 kalibriert λ
# gegen die geschätzte Landmarken-SD. (UnivariateSpline schoss am Rand um ~8° über → verworfen.)
SMOOTH_LAMBDA = 0.1
MIN_CURVE_DEG = 5.0


def _ordered(centroids: dict) -> tuple[list[str], np.ndarray]:
    labels = [k for k in SPINE_LABELS if k in centroids]
    return labels, np.array([centroids[k] for k in labels], dtype=float).reshape(-1, 2)


def tilts_discrete(centroids: dict) -> dict[str, float]:
    """Methode (a): Richtung über zentrale Differenzen (Randwirbel einseitig)."""
    labels, p = _ordered(centroids)
    if len(labels) < 2:
        return {}
    t = -np.gradient(p, axis=0)                   # Reihenfolge kranial→kaudal, t zeigt kranial
    return dict(zip(labels, np.degrees(np.arctan2(t[:, 0], t[:, 1]))))


def tilts_spline(centroids_wkd: dict, lam: float = SMOOTH_LAMBDA) -> dict[str, float]:
    """Methode (b): natürlicher kubischer Glättungsspline x(y) mit festem λ, Neigung aus dx/dy."""
    labels, p = _ordered(centroids_wkd)
    if len(labels) < 5 or np.any(np.diff(p[:, 1]) >= 0):   # scipy braucht ≥ 5 Punkte; kraniokaudale Monotonie
        return {}
    y, x = p[::-1, 1], p[::-1, 0]                   # aufsteigendes y für scipy
    spl = make_smoothing_spline(y, x, lam=lam)
    return dict(zip(labels, np.degrees(np.arctan(spl.derivative()(p[:, 1])))))


def _turning_points(v: np.ndarray, idx: list[int]) -> list[int]:
    """Endpunkte + Richtungswechsel der Folge v[idx]; Plateaus (Δ = 0) werden übersprungen."""
    keep, prev = [idx[0]], 0
    for a, b in zip(idx, idx[1:]):
        d = np.sign(v[b] - v[a])
        if d and prev and d != prev:
            keep.append(a)
        prev = d or prev
    return keep + [idx[-1]] if len(idx) > 1 else keep


def end_vertebrae(tilts: dict, min_angle: float = MIN_CURVE_DEG) -> list[tuple[str, str]]:
    """Endwirbelpaare = aufeinanderfolgende Extrema der Neigung. Extrema-Paare mit
    |Δθ| < min_angle werden zusammengelegt (Rauschen soll eine Kurve nicht teilen)."""
    labels = [k for k in SPINE_LABELS if k in tilts]
    if len(labels) < 2:
        return []
    v = np.array([tilts[k] for k in labels])
    ext = _turning_points(v, list(range(len(v))))
    while len(ext) > 2:
        d = np.abs(np.diff(v[ext]))
        i = int(np.argmin(d))
        if d[i] >= min_angle:
            break
        drop = {ext[i], ext[i + 1]} - {ext[0], ext[-1]}   # Endpunkte bleiben
        ext = _turning_points(v, [e for e in ext if e not in drop])
    return [(labels[a], labels[b]) for a, b in zip(ext, ext[1:]) if abs(v[a] - v[b]) >= min_angle]


def centroid_cobb(tilts: dict, upper: str, lower: str) -> float | None:
    """Vorzeichenbehafteter Winkel θ_upper − θ_lower (fixe Endwirbel, z. B. aus der Baseline)."""
    if upper not in tilts or lower not in tilts:
        return None
    return float(tilts[upper] - tilts[lower])


def apex(centroids: dict, upper: str, lower: str) -> tuple[str, str] | None:
    """(Apex-Label, Konvexität 'links'/'rechts' aus Patientensicht): größter seitlicher
    Abstand von der Sehne Endwirbel–Endwirbel."""
    labels, p = _ordered(centroids)
    if upper not in labels or lower not in labels:
        return None
    i, j = labels.index(upper), labels.index(lower)
    if j - i < 2:
        return None
    a, b = p[i], p[j]
    n = np.array([-(b - a)[1], (b - a)[0]])        # zeigt bei a oben/b unten nach Patient links
    n = n / np.linalg.norm(n) * np.sign(a[1] - b[1] or 1)
    d = (p[i + 1:j] - a) @ n
    k = int(np.argmax(np.abs(d)))
    return labels[i + 1 + k], 'links' if d[k] > 0 else 'rechts'


def tilt_to_horizontal(line) -> float:
    """Betrag des Winkels einer Linie (2×2, Bild- oder Patientenkoordinaten) zur Horizontalen, 0–90°.
    Wie Gharbi 2024 als Betrag – die Seite geht dabei verloren."""
    d = np.asarray(line[1], dtype=float) - np.asarray(line[0], dtype=float)
    a = abs(np.degrees(np.arctan2(d[1], d[0]))) % 180
    return float(min(a, 180 - a))


def coronal_global(centroids: dict, s1_mid, wkd: float | None) -> dict[str, float]:
    """Globale koronare Ausrichtung (Gharbi 2024), Patientenkoordinaten:
    CTPA = Winkel zwischen Vertikaler und Linie S1-Mitte → T1-Centroid (Betrag),
    C7_CSVL_wkd = horizontaler Abstand C7-Centroid ↔ S1-Mitte in WKD (Betrag)."""
    out: dict[str, float] = {}
    if s1_mid is None:
        return out
    s1 = np.asarray(s1_mid, dtype=float)
    if 'T1' in centroids:
        d = np.asarray(centroids['T1'], dtype=float) - s1
        out['CTPA'] = float(abs(np.degrees(np.arctan2(d[0], d[1]))))
    if 'C7' in centroids and wkd:
        out['C7_CSVL_wkd'] = float(abs(centroids['C7'][0] - s1[0]) / wkd)
    return out


def curves(centroids: dict, tilts: dict, min_angle: float = MIN_CURVE_DEG) -> list[dict]:
    out = []
    for u, low in end_vertebrae(tilts, min_angle):
        a = apex(centroids, u, low)
        out.append(dict(upper=u, lower=low, angle=centroid_cobb(tilts, u, low),
                        apex=a[0] if a else None, convexity=a[1] if a else None))
    return out
