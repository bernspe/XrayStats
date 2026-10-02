"""Bild- → Patientenkoordinaten, Sakrumsystem, Längennormierung (Spec §4.1, §4.6; DECISIONS.md).

Patientenkoordinaten sagittal: x = anterior (+), y = kranial (+).
Koronar: x = Patient links (+), y = kranial (+).
"""
from __future__ import annotations

import numpy as np

LUMBAR_CALIB_LEVELS = ('L1', 'L2', 'L3', 'L4', 'L5')   # thorakal/S1 = ungültig (DECISIONS.md)
REFERENCE_LEVEL = 'L4'


def view_of(img_type: str | None) -> str:
    """'lat' für seitliche/sagittale Aufnahmen, sonst 'ap' (gleiche Regel wie xray-annotator)."""
    return 'lat' if any(k in (img_type or '') for k in ('lat', 'sag', 'incl', 'recl')) else 'ap'


def posture_of(img_type: str | None) -> str:
    """'standing' (Xray), 'functional' (incl/recl), 'supine' (MRI) – DECISIONS.md."""
    t = img_type or ''
    if t.startswith('MRI'):
        return 'supine'
    if 'incl' in t or 'recl' in t:
        return 'functional'
    return 'standing'


def to_patient(pts, facing: int) -> np.ndarray:
    """Bildpixel (y nach unten) → Patientenkoordinaten. facing = +1, wenn anterior (bzw.
    Patient links) im Bild nach +x zeigt, sonst −1."""
    return np.asarray(pts, dtype=float) * np.array([facing, -1.0])


def facing_lateral(s1_endplate_img=None, centroids_img: dict | None = None) -> tuple[int, set[str]]:
    """Blickrichtung einer Seitaufnahme ohne geordnete Deckplattenpunkte (Spec §4.1).

    Primär: das tiefer liegende Ende der S1-Deckplatte ist anterior (SS > 0 im Stand).
    Kontrolle/Fallback: der lumbale Scheitel liegt anterior der Sehne L1–S1 (Lordose).
    Gibt (facing, flags) zurück; das Flag 'orientation_heuristic' ist immer gesetzt.
    """
    flags = {'orientation_heuristic'}
    by_endplate = by_lordosis = 0
    if s1_endplate_img is not None:
        (x1, y1), (x2, y2) = np.asarray(s1_endplate_img, dtype=float)
        low, high = ((x1, y1), (x2, y2)) if y1 > y2 else ((x2, y2), (x1, y1))
        by_endplate = int(np.sign(low[0] - high[0]))
    c = centroids_img or {}
    if 'L1' in c and 'S1' in c and 'L3' in c:
        a, b, m = (np.asarray(c[k], dtype=float) for k in ('L1', 'S1', 'L3'))
        t = (m[1] - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.5
        by_lordosis = int(np.sign(m[0] - (a[0] + t * (b[0] - a[0]))))
    if by_endplate and by_lordosis and by_endplate != by_lordosis:
        flags.add('orientation_conflict')
    facing = by_endplate or by_lordosis
    if not facing:
        flags.add('orientation_unknown')
        facing = 1
    return facing, flags


def facing_ap() -> tuple[int, set[str]]:
    """Radiologische Konvention: Bild-links = Patient-rechts → Patient links = Bild +x."""
    return 1, {'side_assumed'}


def sacral_frame(A, P) -> tuple[np.ndarray, np.ndarray]:
    """Sakrumsystem aus S1-Deckplatte (Patientenkoordinaten, A anterior, P posterior).
    Ursprung M, x entlang e (anterior), y = kraniale Normale. Rückgabe (M, R) mit
    lokal = R @ (p − M)."""
    A, P = np.asarray(A, dtype=float), np.asarray(P, dtype=float)
    e = (A - P) / np.linalg.norm(A - P)
    return (A + P) / 2, np.array([e, [-e[1], e[0]]])


def to_frame(pts, M, R) -> np.ndarray:
    return (np.asarray(pts, dtype=float) - M) @ R.T


# ---------------------------------------------------------------------------
# Längennormierung: Wirbelkörperdurchmesser (WKD) über die calibrationLine
# ---------------------------------------------------------------------------

def calib_level_factors(rows, min_pairs: int = 20) -> dict[tuple[str, str], dict]:
    """Größenfaktor Level/L4 je Ansicht, geschätzt aus Bildern, an denen Rater
    verschiedene Kalibrierwirbel gewählt haben (DECISIONS.md).

    rows: Iterable von (image_id, view, level, length_px).
    Rückgabe {(view, level): {'factor', 'n'}}; L4 immer 1.0. Andere Level mit < min_pairs
    Bildpaaren fehlen (→ Kalibrierung ungültig).
    """
    per_img: dict = {}
    for image_id, view, level, length in rows:
        if level in LUMBAR_CALIB_LEVELS and length:
            per_img.setdefault((image_id, view), {}).setdefault(level, []).append(length)
    ratios: dict = {}
    for (_, view), levels in per_img.items():
        if REFERENCE_LEVEL not in levels:
            continue
        ref = np.mean(levels[REFERENCE_LEVEL])
        for level, lengths in levels.items():
            if level != REFERENCE_LEVEL:
                ratios.setdefault((view, level), []).append(np.mean(lengths) / ref)
    # L3 ist laut Messanweisung erlaubt → ohne genug Paare Faktor 1.0 statt ungültig (n = 0)
    out = {(v, lvl): {'factor': 1.0, 'n': 0 if lvl == 'L3' else None}
           for v in ('lat', 'ap') for lvl in (REFERENCE_LEVEL, 'L3')}
    for key, r in ratios.items():
        if len(r) >= min_pairs:
            out[key] = {'factor': float(np.median(r)), 'n': len(r)}
    return out


def wkd_px(length_px: float | None, view: str, level: str | None, factors: dict) -> float | None:
    """Pixel pro Wirbelkörperdurchmesser, umgerechnet auf L4. None = Kalibrierung ungültig."""
    f = factors.get((view, level))
    if not length_px or f is None:
        return None
    return length_px / f['factor']


# ---------------------------------------------------------------------------
# Längennormierung über Wirbelkörperhöhen (Cobb-segmental-Linien, DECISIONS.md 2026-09-30)
# ---------------------------------------------------------------------------

def vbh_factors(rows, min_n: int = 20) -> dict[tuple[str, str], dict]:
    """Wirbelkörperhöhe in WKD je Ansicht und Level: Median über Messungen mit gültiger Kalibrierlinie.
    Hält die Einheit WKD (L4-Kalibrierlinie) bei, nur die Messung pro Bild kommt aus den Höhen.

    rows: Iterable von (view, {level: höhe_px}, wkd_px aus Kalibrierlinie oder None).
    Rückgabe {(view, level): {'factor', 'n'}}; Level mit < min_n Messungen fehlen."""
    r: dict = {}
    for view, heights, wkd in rows:
        if wkd:
            for level, h in heights.items():
                r.setdefault((view, level), []).append(h / wkd)
    return {k: {'factor': float(np.median(v)), 'n': len(v)} for k, v in r.items() if len(v) >= min_n}


# ponytail: Streuungen fest aus dem Cache 30.09.2026 (Rater-CV Linie ≈ 5 %, Einzelwirbel ≈ 4 %,
# anatomische Streuung Höhe/Tiefe zwischen Patienten ≈ 12 %); bei neuem Datenstand ggf. neu schätzen.
SIGMA_LINE, SIGMA_VBH, SIGMA_ANAT = 0.05, 0.04, 0.12


def patient_scale(obs, pop: dict) -> list[float | None]:
    """Pixel pro WKD je Messung eines Patienten und einer Ansicht aus Kalibrierlinie und Wirbelkörperhöhen.

    Gewichtete Kleinste Quadrate in log: log h[m, L] = log s[m] + log a[L]; log linie[m] = log s[m].
    a[L] ist die **patienteneigene** Höhe/WKD von Level L, geschätzt über alle Messungen des Patienten
    (Prior: Populationsfaktor pop[L] mit SIGMA_ANAT). Dadurch springt der Maßstab nicht, wenn zwischen
    Visiten oder Ratern andere Wirbel markiert sind oder die Linie fehlt; die absolute Einheit bleibt der
    Mittelwert der Kalibrierlinien des Patienten.
    obs: Liste von (wkd_px aus Linie oder None, {level: höhe_px}); Level ohne pop-Faktor werden ignoriert.
    Rückgabe: wkd_px je Messung, None ohne Beobachtung."""
    levels = sorted({lvl for _, h in obs for lvl in h if lvl in pop})
    col = {lvl: len(obs) + j for j, lvl in enumerate(levels)}
    rows, rhs, w = [], [], []

    def add(idx, y, sigma):
        r = np.zeros(len(obs) + len(levels))
        r[idx] = 1.0
        rows.append(r), rhs.append(y), w.append(1 / sigma)

    for m, (line, heights) in enumerate(obs):
        if line:
            add([m], np.log(line), SIGMA_LINE)
        for lvl, h in heights.items():
            if lvl in col:
                add([m, col[lvl]], np.log(h), SIGMA_VBH)
    for lvl in levels:
        add([col[lvl]], np.log(pop[lvl]), SIGMA_ANAT)
    if not rows:
        return [None] * len(obs)
    w = np.array(w)
    x = np.linalg.lstsq(np.array(rows) * w[:, None], np.array(rhs) * w, rcond=None)[0]
    seen = np.array(rows)[:, :len(obs)].any(axis=0)
    return [float(np.exp(x[m])) if seen[m] else None for m in range(len(obs))]
