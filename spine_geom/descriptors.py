"""Form-Deskriptoren (Spec §4.6). Der Haltungsvektor (SS, PT, VPAs) kommt aus sagittal.py."""
from __future__ import annotations

import numpy as np

from . import SPINE_LABELS
from .frames import sacral_frame, to_frame, to_patient
from .landmarks import Landmarks
from .sagittal import order_endplate


def sagittal_shape(lm: Landmarks, facing: int, wkd: float) -> dict[str, np.ndarray]:
    """Centroide im Sakrumsystem in WKD: intrinsisch, auch auf Teilaufnahmen und liegend messbar."""
    if lm.s1_endplate is None or not wkd:
        return {}
    M, R = sacral_frame(*order_endplate(to_patient(lm.s1_endplate, facing)))
    return {k: to_frame(to_patient(v, facing), M, R) / wkd for k, v in lm.centroids.items()}


def coronal_shape(centroids_img: dict, facing: int, s1_img, direction_img, wkd: float) -> dict[str, np.ndarray]:
    """Koronares Pendant zum Sakrumsystem (a.p.), Centroide in WKD (a.p.-Kalibrierung = Wirbelbreite!).
    Ursprung = S1-Mitte, x-Achse = Beckenquerachse (S1-Deckplatte bzw. Hüftlinie bzw. Bildhorizontale,
    Wahl beim Aufrufer) Richtung Patient links, y = kranial senkrecht dazu. Rückgabe label → (links, kranial)."""
    if s1_img is None or not wkd:
        return {}
    d = to_patient(direction_img, facing)                # linear → Richtungen transformieren direkt
    e = d / np.linalg.norm(d)
    e = e if e[0] >= 0 else -e                         # Achse zeigt nach Patient links
    R = np.array([e, [-e[1], e[0]]])
    o = to_patient(s1_img, facing)
    return {k: to_frame(to_patient(v, facing), o, R) / wkd for k, v in centroids_img.items()}


LUMBAR_CHAIN = ('L1', 'L2', 'L3', 'L4', 'L5', 'S1')


def lumbar_length(centroids: dict) -> float | None:
    """Summe der Centroid-Abstände L1→S1 in der Einheit der Eingabe (px). Maßstabsreferenz, die
    unabhängig von der Kalibrierlinie ist; None, wenn ein Wirbel fehlt."""
    if not all(k in centroids for k in LUMBAR_CHAIN):
        return None
    return float(sum(np.linalg.norm(np.asarray(centroids[a]) - centroids[b])
                     for a, b in zip(LUMBAR_CHAIN, LUMBAR_CHAIN[1:])))


def segment_heights(centroids: dict, wkd: float) -> dict[str, float]:
    """Centroid-Abstand benachbarter Wirbel in WKD (Wirbel- + Bandscheibenhöhe; Abnahme =
    Kollaps). Schlüssel 'L4-L5' usw. Nur mit gültiger Kalibrierung sinnvoll."""
    labels = [k for k in SPINE_LABELS if k in centroids]
    return {f'{a}-{b}': float(np.linalg.norm(np.asarray(centroids[a]) - centroids[b]) / wkd)
            for a, b in zip(labels, labels[1:])
            if SPINE_LABELS.index(b) - SPINE_LABELS.index(a) == 1}
