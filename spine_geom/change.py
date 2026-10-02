"""Veränderungsvektoren (Spec §3.7, Phase 5): multivariater Reliable Change Index.

Δx = x(t₂) − x(t₁),  Σ_Δ = Σ_mess(t₁) + Σ_mess(t₂) + 2·Σ_aufnahme (+ 2·Σ_haltung-Platzhalter)
D² = Δxᵀ Σ_Δ⁻¹ Δx ~ χ²(p) unter H₀,  RCI_k = Δx_k / √Σ_Δ,kk
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

Z95 = 1.959964


def cov_from_sd(sd, corr=None) -> np.ndarray:
    """Σ = D·R·D; ohne R diagonal."""
    sd = np.asarray(sd, dtype=float)
    return np.outer(sd, sd) * (np.eye(len(sd)) if corr is None else np.asarray(corr, dtype=float))


def nearest_psd(s, eps: float = 1e-9) -> np.ndarray:
    """Symmetrisieren und Eigenwerte ≥ eps·max: paarweise geschätzte Korrelationen ergeben nicht
    zwingend eine positiv definite Matrix."""
    s = (np.asarray(s, dtype=float) + np.asarray(s, dtype=float).T) / 2
    w, v = np.linalg.eigh(s)
    return (v * np.clip(w, eps * max(w.max(), 1e-12), None)) @ v.T


def change_test(dx, sigma_delta) -> dict:
    """D², Freiheitsgrade, p-Wert und RCI je Komponente."""
    dx = np.asarray(dx, dtype=float)
    s = np.asarray(sigma_delta, dtype=float)
    d2 = float(dx @ np.linalg.solve(s, dx))
    return dict(d2=d2, df=len(dx), p=float(chi2.sf(d2, len(dx))), rci=dx / np.sqrt(np.diag(s)))


def mean_shift(dx, sigma_delta, n_blocks: int, k: int, members, ref=None) -> dict:
    """Mittlere Verschiebung ausgewählter Wirbel (Blöcke à k Achsen in dx), optional relativ zur
    Verschiebung eines Referenzwirbels: g = A·Δx, Cov = A·Σ_Δ·Aᵀ, Test χ²(k).

    members: Blockindizes der Region; ref: Blockindex des Referenzwirbels (z. B. L1 für die BWS) oder None."""
    a = np.zeros((k, n_blocks * k))
    for i in members:
        a[:, i * k:(i + 1) * k] += np.eye(k) / len(members)
    if ref is not None:
        a[:, ref * k:(ref + 1) * k] -= np.eye(k)
    g, gs = a @ np.asarray(dx, dtype=float), a @ np.asarray(sigma_delta, dtype=float) @ a.T
    return dict(g=g, sd=np.sqrt(np.diag(gs)), test=change_test(g, gs))


def toward_target(dx, x1, target, sigma_delta) -> tuple[float, float]:
    """Komponente von Δx in Richtung Ziel (u = (x_Ziel − x₁)/|…|) mit SD √(uᵀΣu).
    Positiv = Annäherung an die Norm/das Ziel. Komponenten sollten vergleichbar skaliert sein
    (z. B. in Einheiten ihrer Norm-SD), sonst dominiert die Größe mit den größten Zahlen."""
    u = np.asarray(target, dtype=float) - np.asarray(x1, dtype=float)
    n = np.linalg.norm(u)
    if not n:
        return 0.0, float('nan')
    u = u / n
    return float(u @ np.asarray(dx, dtype=float)), float(np.sqrt(u @ np.asarray(sigma_delta) @ u))


def typology(deviant: bool, drifting: bool) -> str:
    """Zustand × Veränderung (Spec §3.7)."""
    return {(False, False): 'unauffällig', (False, True): 'beginnende Dekompensation',
            (True, False): 'kompensiert', (True, True): 'dekompensierend'}[(deviant, drifting)]


def deviation_test(x, target, ref_sd, alpha: float = 0.05) -> dict:
    """„Deutlich abweichend“: D² der Abweichung vom Ziel in Einheiten der Norm-SD ~ χ²(p).
    ponytail: Norm-Komponenten als unkorreliert behandelt – publizierte Normen liefern nur
    Randverteilungen (Spec §3.6); mit Kovarianzen (Formmodell) ersetzen."""
    z = (np.asarray(x, dtype=float) - np.asarray(target, dtype=float)) / np.asarray(ref_sd, dtype=float)
    d2 = float(z @ z)
    p = float(chi2.sf(d2, len(z)))
    return dict(d2=d2, df=len(z), p=p, deviant=p < alpha, z=z)
