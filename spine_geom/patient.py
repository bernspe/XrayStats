"""PI als Patientenkonstante (Spec §4.3, Phase 3).

Modell je Bild i eines Patienten: PI_i = PI_pat + b_i + ē_i,
  ē_i ~ N(0, σ_m²/n_i)  Messfehler (Mittel über n_i Rater), σ_m aus Rater-Streuung am selben Bild
  b_i ~ N(0, τ²)        Zusatzstreuung zwischen Bildern (Beckenrotation, Deckplattenwahl, Übergangswirbel …)
Schätzer: präzisionsgewichtetes Mittel mit w_i = 1/(σ_m²/n_i + τ²).
"""
from __future__ import annotations

import numpy as np


def pooled_sd(groups) -> float:
    """Gepoolte Standardabweichung innerhalb von Gruppen (nur Gruppen mit n ≥ 2)."""
    ss = df = 0.0
    for g in groups:
        g = np.asarray(g, dtype=float)
        if len(g) > 1:
            ss += ((g - g.mean()) ** 2).sum()
            df += len(g) - 1
    return float(np.sqrt(ss / df)) if df else np.nan


def robust_sd(groups) -> float:
    """Robuste SD innerhalb von Gruppen: 1,4826·Median(|paarweise Differenz|)/√2 über alle Paare.
    Unempfindlich gegen einzelne grobe Fehler (verrutschtes Label, falsche Hüftlinie)."""
    d = [abs(a - b) for g in groups for i, a in enumerate(g) for b in list(g)[i + 1:]]
    return float(1.4826 * np.median(d) / np.sqrt(2)) if d else np.nan


def between_image_tau2(means_by_patient, n_by_patient, sigma_m: float) -> float:
    """Momentenschätzer für τ²: gepoolte Varianz der Bildmittel je Patient minus erwarteter
    Messfehleranteil; ≥ 0."""
    num = den = 0.0
    for m, n in zip(means_by_patient, n_by_patient):
        m, n = np.asarray(m, dtype=float), np.asarray(n, dtype=float)
        if len(m) > 1:
            num += ((m - m.mean()) ** 2).sum() - (len(m) - 1) * np.mean(sigma_m ** 2 / n)
            den += len(m) - 1
    return float(max(0.0, num / den)) if den else 0.0


Z_FLAG = 3.0


def _loo_z(means, var, keep):
    """Leave-one-out-z je Bild gegen die Schätzung aus den übrigen *behaltenen* Bildern."""
    w = np.where(keep, 1 / var, 0.0)
    w_rest = w.sum() - w
    with np.errstate(divide='ignore', invalid='ignore'):
        est_rest = ((w * means).sum() - w * means) / w_rest
        return np.where(w_rest > 0, (means - est_rest) / np.sqrt(var + 1 / w_rest), 0.0)


def patient_constant(means, n, sigma_m: float, tau2: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    """(PI_pat, SE, z, flagged) je Patient.

    Sequentiell: das Bild mit dem größten Leave-one-out-|z| > 3 wird ausgeschlossen, dann neu
    gerechnet, bis keins mehr darüber liegt (mind. 2 Bilder bleiben). PI_pat/SE nur aus den
    behaltenen Bildern. Einfaches Leave-one-out reicht nicht: bei 4 Bildern mit einem starken
    Ausreißer steckt der Ausreißer in jeder LOO-Schätzung der guten Bilder und flaggt sie mit."""
    means, n = np.asarray(means, dtype=float), np.asarray(n, dtype=float)
    var = sigma_m ** 2 / n + tau2
    keep = np.ones(len(means), dtype=bool)
    while keep.sum() > 2:
        z = np.where(keep, np.abs(_loo_z(means, var, keep)), 0.0)
        if z.max() <= Z_FLAG:
            break
        keep[np.argmax(z)] = False
    w = np.where(keep, 1 / var, 0.0)
    est = float((w * means).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    z = np.where(keep, _loo_z(means, var, keep), (means - est) / np.sqrt(var + se ** 2))
    return est, se, z, ~keep
