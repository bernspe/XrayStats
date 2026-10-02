"""Messunsicherheit (Spec Phase 2): Landmarken-Kovarianz aus Multi-Rater-Daten, Monte-Carlo-
Fortpflanzung auf abgeleitete Größen, SEM/MDC95.

Landmarken-Arten ('kind'): Wirbel-Label (Centroid), 'H' (Hüftachse), 'M_S1'/'M_L1'
(Deckplatten-Mittelpunkt), 'ang_S1'/'ang_L1' (Deckplattenrichtung, skalar in Grad);
Suffix '_t' für Deckplatten aus dem Tangenten-Fallback (andere Fehlerquelle).
Positionen in Patientenkoordinaten und WKD, damit Bilder unterschiedlicher Auflösung poolbar sind.
"""
from __future__ import annotations

import copy

import numpy as np

from .landmarks import ENDPLATES, Landmarks

Z95 = 1.959964


def pooled_cov(groups, max_dev: float | None = None) -> tuple[np.ndarray | None, int, int, int]:
    """Gepoolte Kovarianz innerhalb von Gruppen (Gruppe = dieselbe Landmarke am selben Bild,
    Zeilen = Rater). Rückgabe (Σ, Freiheitsgrade, verworfen, gesamt); Σ = None ohne Gruppe mit n ≥ 2.

    max_dev: Abweichungen (euklidisch, vom Median; bei n = 2 vom Mittel) darüber gelten als grobe
    Fehler (Label um einen Wirbel verrutscht, falsche Deckplatte) und werden verworfen, damit Σ das
    Platzierungsrauschen beschreibt. Grobe Fehler sind eine eigene Fehlerklasse (Rate separat)."""
    ss, df, dropped, total = None, 0, 0, 0
    for g in groups:
        g = np.asarray(g, dtype=float)
        g = g.reshape(len(g), -1)
        total += len(g)
        if max_dev is not None and len(g) >= 2:
            center = np.median(g, axis=0) if len(g) > 2 else g.mean(axis=0)
            keep = np.linalg.norm(g - center, axis=1) <= max_dev
            dropped += int((~keep).sum())
            g = g[keep]
        if len(g) < 2:
            continue
        d = g - g.mean(axis=0)
        ss = d.T @ d if ss is None else ss + d.T @ d
        df += len(g) - 1
    return (ss / df if df else None), df, dropped, total


def mdc95(sem: float) -> float:
    """Minimal Detectable Change (95 %) zwischen zwei Messungen: 1,96·√2·SEM."""
    return Z95 * np.sqrt(2) * sem


def _rot(v, deg):
    a = np.radians(deg)
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])


def perturb(lm: Landmarks, facing: int, wkd: float, cov: dict, rng) -> Landmarks:
    """Kopie von lm mit Landmarken-Rauschen. cov: kind → 2×2 (WKD², Patientenkoordinaten) bzw.
    Skalar-Varianz (Grad²) für 'ang_*'. Fehlende kinds bleiben unverrauscht."""
    out = copy.copy(lm)
    to_img = np.array([facing, -1.0]) * wkd            # Patienten-WKD → Bildpixel

    def shift(kind):
        c = cov.get(kind)
        return rng.multivariate_normal([0, 0], c) * to_img if c is not None else 0.0

    out.centroids = {k: v + shift(k) for k, v in lm.centroids.items()}
    if lm.hip is not None:
        out.hip = lm.hip + shift('H')
    for _, ep in ENDPLATES:
        seg = getattr(lm, f'{ep}_endplate')
        if seg is None:
            continue
        kind = ep.upper() + ('_t' if getattr(lm, f'{ep}_source') == 'tangent' else '')
        m, half = seg.mean(axis=0), (seg[1] - seg[0]) / 2
        var = cov.get(f'ang_{kind}')
        half = _rot(half, rng.normal(0, np.sqrt(var))) if var is not None else half
        m = m + shift(f'M_{kind}')
        setattr(out, f'{ep}_endplate', np.array([m - half, m + half]))
    return out


def mc_sd(fn, lm: Landmarks, facing: int, wkd: float, cov: dict, k: int = 100, seed: int = 0) -> dict:
    """SD jeder von fn(lm) gelieferten Größe unter Landmarken-Rauschen (k Ziehungen).
    ponytail: Landmarken als unabhängig angenommen; Korrelationen zwischen Landmarken (z. B.
    derselbe Rater setzt alle Nodes zu weit anterior) fehlen – der Vergleich mit der empirischen
    Rater-SD zeigt, wie viel das ausmacht."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list] = {}
    for _ in range(k):
        for key, v in fn(perturb(lm, facing, wkd, cov, rng)).items():
            draws.setdefault(key, []).append(v)
    return {key: float(np.std(v, ddof=1)) for key, v in draws.items() if len(v) > 1}
