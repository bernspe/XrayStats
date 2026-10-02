"""Messung (xray_measurement_data) → Landmarken in Bildpixeln.

Nodes → Label über vertebra_range (in Reihenfolge, wie build_nodes_df / xray-annotator).
Deckplatten sind im Schema unbeschriftete cobbLines; die Zuordnung zu S1/L1 erfolgt
über die Lage relativ zu den Nachbar-Centroiden (DECISIONS.md), sonst Tangenten-Fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import SPINE_LABELS

ENDPLATE_MODES = ('Cobb-regional', 'Cobb-segmental')
# kraniale Deckplatten, die aus den Cobb-Linien gesucht werden: (Label, Attributpräfix)
ENDPLATES = (('S1', 's1'), ('L1', 'l1'), ('L4', 'l4'))


@dataclass
class Landmarks:
    centroids: dict[str, np.ndarray]              # Label → (x, y) Bildpixel
    hip: np.ndarray | None = None                 # Hüftachsenmittelpunkt H
    s1_endplate: np.ndarray | None = None         # (2, 2) Bildpixel, ungeordnet
    s1_source: str | None = None                  # 'cobb' | 'tangent'
    l1_endplate: np.ndarray | None = None
    l1_source: str | None = None
    l4_endplate: np.ndarray | None = None         # nur für LDI (L4–S1 / L1–S1)
    l4_source: str | None = None
    calib_px: float | None = None                 # Länge der calibrationLine
    calib_level: str | None = None                # geometrisch bestimmter Zielwirbel
    vb_heights: dict[str, float] = field(default_factory=dict)   # Label → Wirbelkörperhöhe (px)
    flags: set[str] = field(default_factory=set)


def _line(d) -> np.ndarray | None:
    if not isinstance(d, dict) or not all(isinstance(d.get(k), (int, float)) for k in ('x1', 'y1', 'x2', 'y2')):
        return None
    return np.array([[d['x1'], d['y1']], [d['x2'], d['y2']]], dtype=float)


def spacing(c: dict) -> float | None:
    """Mittlerer Centroid-Abstand (Wirbelhöhe + Bandscheibe) als Toleranzmaß."""
    pts = np.array([c[k] for k in SPINE_LABELS if k in c])
    return float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else None


def _neighbors(c: dict, label: str):
    i = SPINE_LABELS.index(label)
    up = c.get(SPINE_LABELS[i - 1]) if i > 0 else None
    down = c.get(SPINE_LABELS[i + 1]) if i + 1 < len(SPINE_LABELS) else None
    return up, down


def _axis(c: dict, label: str) -> np.ndarray | None:
    """Einheitsvektor der Wirbelsäulenrichtung (kranial → kaudal) am Node: zentrale bzw. einseitige Differenz."""
    up, down = _neighbors(c, label)
    d = (down if down is not None else c[label]) - (up if up is not None else c[label])
    return d / np.linalg.norm(d) if np.any(d) else None


def _tangent_endplate(c: dict, label: str, sp: float) -> np.ndarray | None:
    """Endplatte ⟂ zur Wirbelsäulenrichtung am Node, zentriert am Centroid.
    ponytail: Centroid statt Deckplattenmitte → M um ~½ Wirbelhöhe zu tief; Flag 'endplate_from_tangent'
    kennzeichnet das."""
    t = _axis(c, label)
    if t is None:
        return None
    n = np.array([-t[1], t[0]]) * sp / 2
    return np.array([c[label] - n, c[label] + n])


def _chord(c: dict, label: str):
    """Centroid-Sehne, auf der die kraniale Deckplatte von `label` liegt: kranialer Nachbar → label
    (ohne kranialen Nachbarn: label → kaudaler Nachbar, verlängert)."""
    i = SPINE_LABELS.index(label)
    above = c.get(SPINE_LABELS[i - 1]) if i > 0 else None
    below = c.get(SPINE_LABELS[i + 1]) if i + 1 < len(SPINE_LABELS) else None
    if above is not None:
        return above, c[label]
    return (c[label], below) if below is not None else None


def _crossing(ln: np.ndarray, a, b):
    """Schnittpunkt der Linie ln mit der Geraden a–b und Linienparameter t (0/1 = Endpunkte)."""
    d = ln[1] - ln[0]
    m = np.array([d, np.asarray(a) - np.asarray(b)]).T
    if abs(np.linalg.det(m)) < 1e-9:
        return None, None
    t, _ = np.linalg.solve(m, np.asarray(a) - ln[0])
    return ln[0] + t * d, t


def pick_endplate(lines, c: dict, label: str, sp: float) -> np.ndarray | None:
    """Kraniale Deckplatte von `label` aus den gezeichneten Cobb-Linien.

    Die Linien sind Cobb-Linien, keine Deckplattensegmente: im Median 2,1 WKD lang, oft einseitig
    überstehend (DECISIONS.md). Deshalb trägt die Linie nur die **Richtung**; der Mittelpunkt M ist
    ihr Schnittpunkt mit der Centroid-Sehne (analog Frontend *_cobb). Auswahl über diesen
    Schnittpunkt: zwischen der Mitte zum kranialen Nachbarn und dem Centroid (S1: bis ½ Abstand
    darunter), die Linie muss die Sehne innerhalb ihrer Länge (±10 %) kreuzen. Mehrere: S1 die
    tiefste, sonst die dem Centroid nächste. Rückgabe: Segment der Länge 1 Abstand um M."""
    chord = _chord(c, label)
    if chord is None:
        return None
    y0 = c[label][1]
    i = SPINE_LABELS.index(label)
    above = c.get(SPINE_LABELS[i - 1]) if i > 0 else None
    top = (above[1] + y0) / 2 if above is not None else y0 - sp / 2
    bottom = y0 + (sp / 2 if label == 'S1' else 0)
    cands = []
    for ln in lines:
        x, t = _crossing(ln, *chord)
        if x is not None and -0.1 <= t <= 1.1 and top < x[1] <= bottom:
            cands.append((x, ln))
    if not cands:
        return None
    key = (lambda xl: -xl[0][1]) if label == 'S1' else (lambda xl: abs(xl[0][1] - y0))
    x, ln = min(cands, key=key)
    e = (ln[1] - ln[0]) / np.linalg.norm(ln[1] - ln[0])
    return np.array([x - e * sp / 2, x + e * sp / 2])


def vb_heights(lines, c: dict, sp: float) -> dict[str, float]:
    """Wirbelkörperhöhe (px) je Label aus Cobb-segmental-Linien, als Längenmaß für die Kalibrierung.

    Höhe = Abstand der Schnittpunkte von Deck- und Grundplattenlinie mit der Wirbelsäulenachse durch den
    Centroid. Schräg liegende Linien, Keilwirbel und die Länge der gezeichneten Linie gehen damit nicht ein.
    Zuordnung über den Schnittpunkt (Achsenparameter s, kaudal +): Deckplatte −½ Abstand zum kranialen
    Nachbarn < s < 0, Grundplatte 0 < s < ½ Abstand zum kaudalen Nachbarn (ohne Nachbar: ½ sp); die Linie
    muss die Achse innerhalb ihrer Länge (±10 %) kreuzen. Genau eine Linie je Seite, sonst kein Wert
    (doppelt gezeichnet = mehrdeutig).
    ponytail: Achse = Tangente am Centroid statt Spline – Krümmungsfehler über eine Wirbelhöhe ist 2. Ordnung;
    auf Spline umstellen, falls Apex-Wirbel stark gekrümmter Kurven systematisch abweichen."""
    out = {}
    for label, p in c.items():
        t = _axis(c, label)
        if t is None:
            continue
        up, down = _neighbors(c, label)
        lim_up = np.linalg.norm(p - up) / 2 if up is not None else sp / 2
        lim_down = np.linalg.norm(down - p) / 2 if down is not None else sp / 2
        sup, inf = [], []
        for ln in lines:
            x, tl = _crossing(ln, p, p + t)
            if x is None or not -0.1 <= tl <= 1.1:
                continue
            s = float((x - p) @ t)
            if -lim_up < s < 0:
                sup.append(s)
            elif 0 < s < lim_down:
                inf.append(s)
        if len(sup) == 1 and len(inf) == 1:
            out[label] = inf[0] - sup[0]
    return out


def extract(xmd: dict, lateral: bool = True) -> Landmarks:
    """lateral=False: keine S1-/L1-Deckplatten (a.p.-Cobb-Linien sind koronar)."""
    so = (xmd or {}).get('stateObject') or {}
    nodes = so.get('nodes') or []
    vr = so.get('vertebra_range') or [0, -1]
    labels = SPINE_LABELS[vr[0]:vr[1] + 1]
    lm = Landmarks(centroids={lab: np.array([n['x'], n['y']], dtype=float) for lab, n in zip(labels, nodes)})
    if len(nodes) != len(labels):
        lm.flags.add('node_count_mismatch')
    c, sp = lm.centroids, spacing(lm.centroids)

    hip = _line(so.get('hipLine'))
    if hip is not None:
        lm.hip = hip.mean(axis=0)

    lines = [ln for d in (so.get('cobbLines') or []) if d.get('mode') in ENDPLATE_MODES
             for ln in [_line(d)] if ln is not None]
    if sp and lateral:
        for label, attr in ENDPLATES:
            if label not in c:
                continue
            ep = pick_endplate(lines, c, label, sp)
            src = 'cobb'
            if ep is None:
                ep, src = _tangent_endplate(c, label, sp), 'tangent'
                lm.flags.add('endplate_from_tangent')
            if ep is not None:
                setattr(lm, f'{attr}_endplate', ep)
                setattr(lm, f'{attr}_source', src)

    # unveränderte Modellvorschläge (src='model', identisch mit suggestedLines) sind nicht rater-geprüft
    suggested = {tuple(ln.ravel()) for d in so.get('suggestedLines') or [] for ln in [_line(d)] if ln is not None}
    seg = [ln for d in (so.get('cobbLines') or []) if d.get('mode') == 'Cobb-segmental'
           for ln in [_line(d)] if ln is not None and np.any(ln[1] - ln[0])
           and not (d.get('src') == 'model' and tuple(ln.ravel()) in suggested)]
    if sp:
        lm.vb_heights = vb_heights(seg, c, sp)

    cal = _line(so.get('calibrationLine'))
    if cal is not None and c:
        lm.calib_px = float(np.linalg.norm(cal[1] - cal[0])) or None
        mid = cal.mean(axis=0)
        lm.calib_level = min(c, key=lambda k: np.linalg.norm(c[k] - mid))
    return lm
