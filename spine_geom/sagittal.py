"""Sagittale Parameter (Spec §4.2, §4.4). Alle Eingaben in Patientenkoordinaten
(x anterior +, y kranial +), Ausgaben in Grad bzw. WKD (Wirbelkörperdurchmesser)."""
from __future__ import annotations

import numpy as np

from .frames import to_patient
from .landmarks import ENDPLATES, Landmarks


# ponytail: unter 10° LL ist der Quotient L4–S1 / L1–S1 numerisch instabil (Nenner ~ Messfehler)
LDI_MIN_LL = 10.0


def _deg(r): return float(np.degrees(r))
def _unit(v): return v / np.linalg.norm(v)
def _cross(u, v): return u[0] * v[1] - u[1] * v[0]


def order_endplate(pts) -> tuple[np.ndarray, np.ndarray]:
    """(A, P): anteriorer, posteriorer Endpunkt (nach Normalisierung ist anterior = größeres x)."""
    a, b = np.asarray(pts, dtype=float)
    return (a, b) if a[0] >= b[0] else (b, a)


def endplate_tilt(A, P) -> float:
    """atan2(−e_y, e_x): positiv, wenn das anteriore Ende tiefer liegt. Für S1 = SS, für L1 = τ_L1."""
    e = _unit(np.asarray(A) - np.asarray(P))
    return _deg(np.arctan2(-e[1], e[0]))


def pelvic_tilt(M, H) -> float:
    d = np.asarray(M) - np.asarray(H)
    return _deg(np.arctan2(-d[0], d[1]))


def pelvic_incidence(A, P, H) -> float:
    A, P, H = (np.asarray(p, dtype=float) for p in (A, P, H))
    e, M = _unit(A - P), (A + P) / 2
    n = np.array([e[1], -e[0]])                               # kaudale Normale, ins Sakrum
    return _deg(np.arccos(np.clip(n @ _unit(H - M), -1, 1)))


def vpa(C, M, H) -> float:
    """Vertebral-Pelvic-Angle am Hüftachsenmittelpunkt; positiv = C anterior der Linie H→M."""
    u, v = np.asarray(M) - np.asarray(H), np.asarray(C) - np.asarray(H)
    return -_deg(np.arctan2(_cross(u, v), u @ v))


def global_tilt(C7, M, H) -> float:
    """Winkel an M zwischen Verlängerung H→M und M→C7; positiv = C7 anterior."""
    u, w = np.asarray(M) - np.asarray(H), np.asarray(C7) - np.asarray(M)
    return -_deg(np.arctan2(_cross(u, w), u @ w))


# Parameter → benötigte Landmarken. 's1'/'l1'/'l4' = Deckplatte, 'hip', 'wkd', sonst Centroid-Label.
REQUIRES: dict[str, tuple[str, ...]] = {
    'PI': ('s1', 'hip'),
    'SS': ('s1',),
    'PT': ('s1', 'hip'),
    'LL': ('s1', 'l1'),
    'LDI': ('s1', 'l1', 'l4'),
    'L1PA': ('s1', 'hip', 'L1'),
    'T4PA': ('s1', 'hip', 'T4'),
    'T1PA': ('s1', 'hip', 'T1'),
    'T4L1PA_mismatch': ('s1', 'hip', 'T4', 'L1'),
    'GT': ('s1', 'hip', 'C7'),
    'SVA_wkd': ('s1', 'C7', 'wkd'),
}
# Alle sagittalen Parameter nur aus seitlichen Röntgen-Stehaufnahmen. PI ist zwar lageunabhängig,
# lässt sich aus MRT (Hüftköpfe/Deckplatte nicht verlässlich abgebildet) und Funktionsaufnahmen
# aber nicht belastbar ableiten (DECISIONS.md, 2026-09-29).

MISSING_TEXT = {
    's1': 'keine S1-Deckplatte', 'l1': 'keine L1-Deckplatte', 'l4': 'keine L4-Deckplatte', 'hip': 'keine Hüfte',
    'wkd': 'keine gültige Kalibrierung', 'L1': 'L1 nicht annotiert', 'T4': 'T4 nicht annotiert',
    'T1': 'T1 nicht annotiert', 'C7': 'C7 nicht annotiert',
}


def available(lm: Landmarks, wkd: float | None) -> set[str]:
    have = {k for k in ('hip',) if getattr(lm, k) is not None}
    have |= {k for _, k in ENDPLATES if getattr(lm, f'{k}_endplate') is not None}
    have |= set(lm.centroids)
    if wkd:
        have.add('wkd')
    return have


def missing_reason(param: str, have: set[str]) -> str | None:
    miss = [MISSING_TEXT.get(r, r) for r in REQUIRES[param] if r not in have]
    return ', '.join(miss) or None


def sagittal_params(lm: Landmarks, facing: int, wkd: float | None) -> dict[str, float]:
    """Alle berechenbaren Parameter einer seitlichen Messung (Bildpixel → Patientenkoordinaten)."""
    have = available(lm, wkd)
    if 's1' not in have:
        return {}
    A, P = order_endplate(to_patient(lm.s1_endplate, facing))
    M = (A + P) / 2
    c = {k: to_patient(v, facing) for k, v in lm.centroids.items()}
    H = to_patient(lm.hip, facing) if lm.hip is not None else None
    out = {'SS': endplate_tilt(A, P)}
    if H is not None:
        out['PT'] = pelvic_tilt(M, H)
        out['PI'] = pelvic_incidence(A, P, H)
        for lab, key in (('L1', 'L1PA'), ('T4', 'T4PA'), ('T1', 'T1PA')):
            if lab in c:
                out[key] = vpa(c[lab], M, H)
        if 'T4PA' in out and 'L1PA' in out:
            out['T4L1PA_mismatch'] = out['T4PA'] - out['L1PA']
        if 'C7' in c:
            out['GT'] = global_tilt(c['C7'], M, H)
    if lm.l1_endplate is not None:
        out['LL'] = out['SS'] - endplate_tilt(*order_endplate(to_patient(lm.l1_endplate, facing)))
        if lm.l4_endplate is not None and out['LL'] >= LDI_MIN_LL:
            lower = out['SS'] - endplate_tilt(*order_endplate(to_patient(lm.l4_endplate, facing)))
            out['LDI'] = 100 * lower / out['LL']
    if 'C7' in c and wkd:
        out['SVA_wkd'] = float((c['C7'][0] - P[0]) / wkd)
    return out
