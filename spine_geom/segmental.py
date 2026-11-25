"""Segmentaler Therapieverlauf (Konzept: docs/degen_skoliose_verlauf_segmental_bottom_up.md).

Schritt 0: Therapiegruppe, Segmenthöhen aus Freitext, Therapie-Episoden. Rein tabellarisch, ohne I/O.
Schritt 1: Segmentwinkel aus der Frontend-Spline (mit Vorzeichen), Klassifikation der a.p.-Beckenlinie.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from . import SPINE_LABELS
from .coronal import tilt_to_horizontal

GROUPS = ('Fusion', 'Dekompression', 'konservativ')          # Rangfolge bei Konflikt am selben Tag
# Rohwert ClinicalEntryData.data.therapy (klein, getrimmt) → (Gruppe | None = keine Therapie, Revision)
THERAPY_MAP = {
    'konservativ': ('konservativ', False),
    'konservativ multimodale komplextherapie': ('konservativ', False),
    'multimodale komplextherapie': ('konservativ', False),
    'epidurale injektion': ('konservativ', False),
    'dekompression': ('Dekompression', False),
    'fusion': ('Fusion', False),
    'konversion in 4-stab': ('Fusion', True),
    'indikation': (None, False),
}
# Fallback für leere/unbekannte therapy: Schlüsselwörter in therapy + therapy_specification, erste Gruppe gewinnt
KEYWORDS = [
    ('Fusion', r'fusion|spondylodese|[tpax]lif\b|olif\b|stabilisierung|instrumentierung'),
    ('Dekompression', r'dekompression|decompression|laminektomie|laminotomie|flavektomie|foraminotomie|undercutting'),
    ('konservativ', r'konservativ|infiltration|\bprt\b|epidural|injektion|physio|krankengym|orthese|akupunktur'
                    r'|fango|gangschule|komplextherapie|schmerztherapie|psycholog|\bi\.v\.'),
]
BASELINE_MAX_DAYS = 365          # ältere Baseline → Flag baseline_old
EARLY_MAX_DAYS = 92              # erste Verlaufsvisite ≤ 3 Monate → early
PLANES = ('ap', 'lat')

_VERT = re.compile(r'\b(LWK|BWK|SWK|HWK|TH)\s*(?=\d)', re.I)
_PELVIS = re.compile(r'\s*[-–]\s*(os\s+)?(ilium|iliakal|becken|pelvis)\b', re.I)   # 'L2-Os ilium' = bis S1 + Becken
_SEG = re.compile(r'\b([CTLS])\s?(\d{1,2})\s*([/\-–])\s*([CTLS])?\s?(\d{1,2})\b')


def _text(v) -> str:
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return ' ; '.join(str(x) for x in v)
    return '' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def _seg_key(seg: str) -> int:
    """Sortierschlüssel kranial → kaudal; Nicht-Segmente (regionale Parameter) ans Ende."""
    head = str(seg).split('/')[0]
    return SPINE_LABELS.index(head) if head in SPINE_LABELS else len(SPINE_LABELS)


def parse_levels(text) -> list[str]:
    """Freitext → Segmente 'L4/L5' in kraniokaudaler Reihenfolge. 'L4/5', 'LWK 5/SWK 1', 'Th12/L1' = ein Segment,
    'L2–L5' / 'L2/L5' = alle Segmente dazwischen. Unbekannte Wirbel (L6) werden ignoriert."""
    t = _VERT.sub(lambda m: {'LWK': 'L', 'BWK': 'T', 'SWK': 'S', 'HWK': 'C', 'TH': 'T'}[m[1].upper()], _text(text))
    t = _PELVIS.sub('-S1', t)
    segs = set()
    for a, n1, _, b, n2 in _SEG.findall(t):
        v1, v2 = f'{a}{int(n1)}', f'{b or a}{int(n2)}'
        if v1 in SPINE_LABELS and v2 in SPINE_LABELS:
            i, j = sorted((SPINE_LABELS.index(v1), SPINE_LABELS.index(v2)))
            segs.update(f'{SPINE_LABELS[k]}/{SPINE_LABELS[k + 1]}' for k in range(i, j))
    return sorted(segs, key=_seg_key)


def to_pelvis(text) -> bool:
    return bool(_PELVIS.search(_text(text)))


def therapy_group(therapy, spec=None) -> tuple[str | None, bool, str]:
    """→ (Gruppe | None, Revision, Quelle 'map' | 'keyword' | 'none')."""
    raw = _text(therapy).strip().lower()
    if raw in THERAPY_MAP:
        return (*THERAPY_MAP[raw], 'map')
    hay = f'{raw} ; {_text(spec).lower()}'
    for group, pat in KEYWORDS:
        if re.search(pat, hay):
            return group, 'revision' in hay or 'konversion' in hay, 'keyword'
    return None, False, 'none'


def _as_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def therapy_events(entries: pd.DataFrame) -> pd.DataFrame:
    """ClinicalEntries eines oder mehrerer Patienten → ein Therapieereignis je (Patient, Tag).
    Einträge ohne Therapiegruppe sind keine Ereignisse. Am selben Tag: Rangfolge GROUPS, Höhen vereinigt."""
    cols = ['patient', 'date', 'group', 'revision', 'group_source', 'levels', 'fused_segments', 'fused_to_pelvis',
            'therapy_raw', 'entry_uuids']
    rows = []
    for e in entries.itertuples():
        g, rev, src = therapy_group(getattr(e, 'therapy', None), getattr(e, 'therapy_specification', None))
        if g is None or pd.isna(pd.to_datetime(e.referring_date, errors='coerce')):
            continue
        rows.append({'patient': e.patient, 'date': pd.to_datetime(e.referring_date).normalize(), 'group': g,
                     'revision': rev, 'group_source': src,
                     'levels': parse_levels(getattr(e, 'therapy_specification', None)),
                     'fused_segments': _as_int(getattr(e, 'fused_segments', None)),
                     'fused_to_pelvis': getattr(e, 'fused_to_pelvis', False) is True
                                        or (g == 'Fusion' and to_pelvis(getattr(e, 'therapy_specification', None))),
                     'therapy_raw': _text(getattr(e, 'therapy', None)), 'entry_uuids': [e.uuid]})
    if not rows:
        return pd.DataFrame(columns=cols)
    out = []
    for (p, d), g in pd.DataFrame(rows).groupby(['patient', 'date'], sort=True):
        g = g.assign(rank=g['group'].map(GROUPS.index)).sort_values('rank')
        top = g.iloc[0]
        out.append({'patient': p, 'date': d, 'group': top['group'], 'revision': bool(g['revision'].any()),
                    'group_source': top['group_source'],
                    'levels': sorted({s for lv in g['levels'] for s in lv}, key=_seg_key),
                    'fused_segments': g['fused_segments'].dropna().max() if g['fused_segments'].notna().any() else None,
                    'fused_to_pelvis': bool(g['fused_to_pelvis'].any()),
                    'therapy_raw': ' | '.join(sorted(set(g['therapy_raw']) - {''})),
                    'entry_uuids': [u for lst in g['entry_uuids'] for u in lst]})
    return pd.DataFrame(out, columns=cols)


def episodes(events: pd.DataFrame, visits: pd.DataFrame, index_date=None) -> pd.DataFrame:
    """Therapieereignisse EINES Patienten + seine Visiten (Spalten date, plane) → eine Zeile je Episode.
    Baseline = letzte Visite bis einschließlich Ereignistag; Verlauf = Visiten nach dem Ereignistag bis einschließlich
    des nächsten Ereignistags (dessen Aufnahme ist zugleich Baseline der Folge-Episode).
    Abweichung von spine_geom.course (dort zählt der Indextag als post): im Cache 10/2026 fallen Visiten am
    Ereignistag nur bei konservativen Episoden an und sind die Aufnahme beim Vorstellungstermin (11/11).
    ponytail: Tagesgenauigkeit ohne Uhrzeit; Flag same_day_visit zeigt, wie oft das vorkommt."""
    ev = events.sort_values('date').reset_index(drop=True)
    v = visits.assign(date=pd.to_datetime(visits['date']).dt.normalize())
    idx = pd.to_datetime(index_date) if index_date is not None and not pd.isna(index_date) else None
    rows = []
    for k, e in ev.iterrows():
        nxt = ev['date'].iloc[k + 1] if k + 1 < len(ev) else None
        pre = v[v['date'] <= e['date']]
        post = v[(v['date'] > e['date']) & ((v['date'] <= nxt) if nxt is not None else True)]
        base_d = pre['date'].max() if len(pre) else None
        base_planes = set(pre.loc[pre['date'] == base_d, 'plane']) if base_d is not None else set()
        post_d = sorted(post['date'].unique())
        early = post_d[0] if post_d and (post_d[0] - e['date']).days <= EARLY_MAX_DAYS else None
        late = post_d[-1] if post_d else None
        # je Ebene: Baseline UND Verlauf vorhanden? Baseline je Ebene = letzte Visite vor dem Ereignis mit dieser Ebene
        planes = [p for p in PLANES if (pre['plane'] == p).any() and (post['plane'] == p).any()]
        if base_d is None:
            status = 'keine Baseline'
        elif not post_d:
            status = 'kein Verlauf'
        elif not planes:
            status = 'keine gemeinsame Ebene'
        else:
            status = 'auswertbar'
        months = lambda d: (d - e['date']).days / 30.4375 if d is not None else np.nan
        rows.append({**e.to_dict(), 'next_event': nxt, 'baseline_date': base_d,
                     'baseline_planes': sorted(base_planes), 'early_date': early, 'late_date': late,
                     'n_post_visits': len(post_d), 'planes': planes,
                     'months_baseline': months(base_d), 'months_early': months(early), 'months_late': months(late),
                     'months_from_index': (e['date'] - idx).days / 30.4375 if idx is not None else np.nan,
                     'baseline_old': base_d is not None and (e['date'] - base_d).days > BASELINE_MAX_DAYS,
                     'same_day_visit': bool((v['date'] == e['date']).any()), 'status': status})
    return pd.DataFrame(rows)


# --- Schritt 1: Segmentwinkel -------------------------------------------------------------------------
FACING = {'lat': -1, 'ap': 1}    # Bild-x → Patient: seitlich anterior = Bild links (Nutzervorgabe), a.p. radiologisch


def frontend_spline(pts, num_steps: int = 10) -> np.ndarray:
    """Spline wie xray-annotator src/helpers/bspline.ts interpolateBSpline (≥ 2.17.6): geklemmter kubischer
    B-Spline mit den Nodes als KONTROLLpunkten (geht nur durch den ersten und letzten Node, glättet dazwischen),
    10 · n + 1 Punkte. bspline_python.interpolate_bspline (interpolierend) ist der ältere Frontend-Stand.
    ponytail: Spiegel des TS-Codes; ändert sich das Frontend, hier nachziehen (Paritätstest gegen den Cache)."""
    p = np.asarray(pts, dtype=float)
    n = len(p)
    if n < 4:
        return np.empty((0, 2))
    knots = np.r_[[0.0] * 4, np.arange(1, n - 3), [n - 3.0] * 4]
    steps = num_steps * n
    return BSpline(knots, p, 3)(np.arange(steps + 1) / steps * (n - 3))


def node_tangents(nodes) -> list[tuple[np.ndarray, np.ndarray]]:
    """(Ankerpunkt, Einheitsrichtung kaudalwärts) je Node in Bild-/Stage-Koordinaten, wie das Frontend:
    Node → Spline-Punkt mit der nächsten Höhe (floor(y)), Tangente über ±win Spline-Punkte
    (win ≈ ein Wirbel, measureCompute.splineAngles)."""
    pts = [(float(x), float(y)) for x, y in nodes]
    spl = frontend_spline(pts)
    if not len(spl):
        return []
    fy = np.floor(spl[:, 1])
    last, win = len(spl) - 2, int(np.floor(len(spl) / len(pts) + 0.5))
    out = []
    for x, y in pts:
        i = min(int(np.argmin(np.abs(fy - y))), last)
        d = spl[min(len(spl) - 1, i + win)] - spl[max(0, i - win)]
        out.append((spl[i], d / np.linalg.norm(d)))
    return out


def node_tilts(nodes, plane: str) -> list[float]:
    """θ je Node = Neigung der kranialwärts zeigenden Frontend-Tangente gegen die Vertikale in
    Patientenkoordinaten (seitlich + = anterior, a.p. + = Patient links)."""
    theta = []
    for _, d in node_tangents(nodes):
        t = -d * np.array([FACING[plane], -1.0])                # Patientenkoordinaten, kranialwärts
        theta.append(float(np.degrees(np.arctan2(t[0], t[1]))))
    return theta


def endplate_tilt(line, plane: str) -> float:
    """Gezeichnete Deckplatte (2×2 Bildkoordinaten, Endpunkte ungeordnet) → θ ihrer kranialwärts zeigenden
    Normalen gegen die Vertikale, in derselben Konvention wie node_tilts."""
    e = (np.asarray(line[1], float) - np.asarray(line[0], float)) * np.array([FACING[plane], -1.0])
    n = np.array([-e[1], e[0]])
    n = n if n[1] > 0 else -n
    return float(np.degrees(np.arctan2(n[0], n[1])))


def segment_angles(nodes, labels, plane: str, s1_endplate=None) -> dict[str, dict]:
    """Segmentwinkel wie das Frontend (`Spline-segmental`, measureCompute.ts), aber mit Vorzeichen und ungerundet.

    nodes: [(x, y)] in Stage-Koordinaten (y nach unten), kranial → kaudal; labels gleich lang; mind. 4 Nodes.
    Segment oben/unten: α = θ_unten − θ_oben (node_tilts) → seitlich + = Lordose (anterior offen), a.p. + = Öffnung
    nach Patient links. round(|α|) = Frontend-Wert.
    Randsegmente (Nutzerentscheidung 24.11.2026): der geklemmte Spline läuft an den Enden in Richtung der letzten
    Kontrollpolygon-Kante aus, die Tangente dort ist einseitig → Randwinkel strukturell zu klein (L5/S1 seitlich
    Median 3,9° statt 10°). Deshalb:
      - oberstes Segment und unterstes Segment, wenn es nicht bei S1 endet: nicht ausgegeben;
      - L5/S1 mit gezeichneter S1-Deckplatte (s1_endplate, 2×2 Stage-Koordinaten): θ_S1 = Deckplattennormale
        statt Tangente, src = 's1_endplate'; ohne Deckplatte Spline-Wert mit src = 's1_spline' und edge = True.
    Rückgabe {'L4/L5': {'angle', 'src', 'edge'}}.
    """
    theta = node_tilts(nodes, plane) if len(labels) == len(nodes) else []
    out = {}
    for k in range(1, len(theta) - 1):                             # k = 0: oberstes Segment entfällt
        up, lo = labels[k], labels[k + 1]
        if up not in SPINE_LABELS or lo not in SPINE_LABELS or SPINE_LABELS.index(lo) != SPINE_LABELS.index(up) + 1:
            continue
        bottom = k + 1 == len(theta) - 1
        if bottom and lo != 'S1':
            continue
        lo_theta, src = theta[k + 1], 'spline'
        if bottom:
            lo_theta, src = ((endplate_tilt(s1_endplate, plane), 's1_endplate') if s1_endplate is not None
                             else (lo_theta, 's1_spline'))
        out[f'{up}/{lo}'] = {'angle': float((lo_theta - theta[k] + 180) % 360 - 180), 'src': src,
                             'edge': src == 's1_spline'}
    return out


# --- Schritt 1: a.p.-Beckenlinie ----------------------------------------------------------------------
# Schwellen aus der Verteilung im Cache 10/2026 (697 a.p.-Messungen, zwei getrennte Häufungen), Einheit =
# L5–S1-Centroidabstand. ponytail: feste Schwellenregel; ~16 % bleiben 'unklar'. Bei Bedarf hier nachjustieren.
ISG_MAX_W, HIP_MIN_W, HIP_MIN_DY = 4.5, 5.5, 2.5


def pelvis_ref(hip_line, c_l5, c_s1) -> dict | None:
    """hipLine {x1, y1, x2, y2} einer a.p.-Messung → Hüftkopfzentren ('hip') oder untere ISG-Enden ('isg').
    w = Linienbreite, dy = Linienmitte unter dem S1-Centroid (Bild-y nach unten), beide in L5–S1-Abständen.
    'obliquity' = Neigung der Linie gegen die Horizontale (Beckenschiefstand), nutzbar für hip und isg."""
    if not hip_line or c_l5 is None or c_s1 is None or not all(k in hip_line for k in ('x1', 'y1', 'x2', 'y2')):
        return None
    sp = float(np.hypot(c_s1[0] - c_l5[0], c_s1[1] - c_l5[1]))
    if sp <= 0:
        return None
    w = abs(hip_line['x2'] - hip_line['x1']) / sp
    dy = ((hip_line['y1'] + hip_line['y2']) / 2 - c_s1[1]) / sp
    if dy <= 0:
        cls = 'implausibel'
    elif w < ISG_MAX_W and dy < HIP_MIN_DY:
        cls = 'isg'
    elif w >= HIP_MIN_W and dy >= HIP_MIN_DY:
        cls = 'hip'
    else:
        cls = 'unklar'
    line = [[hip_line['x1'], hip_line['y1']], [hip_line['x2'], hip_line['y2']]]
    return {'pelvis_ref': cls, 'w': w, 'dy': dy, 'obliquity': tilt_to_horizontal(line)}


# --- Schritt 2: Winkel einer Episode -----------------------------------------------------------------
def episode_angles(ep, sv: pd.DataFrame) -> pd.DataFrame:
    """Episode (Zeile aus seg_episodes) + seg_visits des Patienten → je Ebene × Segment Baseline, early, late.
    Je Ebene eigene Zeitpunkte: Baseline = letzte Visite dieser Ebene bis einschließlich Ereignistag, Verlauf =
    Visiten danach bis einschließlich nächstem Ereignis (wie episodes()). Δ = post − prä, K = |prä| − |post|."""
    e = pd.Timestamp(ep['date'])
    nxt = ep.get('next_event')
    nxt = pd.Timestamp(nxt) if nxt is not None and not pd.isna(nxt) else None
    rows = []
    for plane, g in sv.groupby('plane'):
        dates = sorted(g['date'].unique())
        pre = [d for d in dates if d <= e]
        post = [d for d in dates if d > e and (nxt is None or d <= nxt)]
        if not pre or not post:
            continue
        t = {'pre': pre[-1], 'early': post[0] if (post[0] - e).days <= EARLY_MAX_DAYS else None, 'late': post[-1]}
        piv = g.pivot_table(index='seg', columns='date', values='angle')
        epiv = g.pivot_table(index='seg', columns='date', values='edge', aggfunc='max')
        for seg in sorted(piv.index, key=_seg_key):
            r = {'plane': plane, 'seg': seg}
            for k, d in t.items():
                r[k] = piv.at[seg, d] if d is not None and d in piv.columns else np.nan
                r[f'date_{k}'] = d
            # edge = L5/S1 nur als Spline-Fallback (ohne S1-Deckplatte); Δ nie über einen Fallback-Wert,
            # sonst stünde Methode gegen Methode (Deckplatte ~13°, Spline ~4°)
            fb = {k: d is not None and d in epiv.columns and pd.notna(epiv.at[seg, d]) and bool(epiv.at[seg, d])
                  for k, d in t.items()}
            r['edge'] = any(fb.values())
            for k in ('early', 'late'):
                ok = not (fb['pre'] or fb[k])
                r[f'd_{k}'] = r[k] - r['pre'] if ok else np.nan
                r[f'k_{k}'] = abs(r['pre']) - abs(r[k]) if ok else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


# --- Schritt 2b: Slip aus Hinterkanten-Ecken ---------------------------------------------------------
def corner_slips(corners: dict, wkd: float) -> dict[str, float]:
    """Hinterkanten-Ecken {Label: (PS, PI)} (Bildkoordinaten, anterior links) → Slip je Segment in % WKD,
    > 0 = Anterolisthese. Definition wie Ground Truth und Modell (train_slip.ipynb): Abstand der PI-Ecke des
    oberen Wirbels zur Hinterkantenlinie PS–PI des unteren, auch für L5/S1 (S1-Hinterkante, nicht Meyerding)."""
    from .posterior_wall import slip
    out = {}
    if not wkd or not np.isfinite(wkd) or wkd <= 0:
        return out
    for k in range(len(SPINE_LABELS) - 1):
        up, lo = SPINE_LABELS[k], SPINE_LABELS[k + 1]
        if up in corners and lo in corners:
            ps_lo, pi_lo = (np.asarray(p, float) for p in corners[lo])
            pi_up = np.asarray(corners[up][1], float)
            if np.isfinite([*ps_lo, *pi_lo, *pi_up]).all() and np.any(pi_lo - ps_lo):
                out[f'{up}/{lo}'] = float(slip(ps_lo, pi_lo, pi_up) / wkd * 100)
    return out


# --- Schritt 3: Implantate → Höhen, Fusionsstrecke, Segmentklassen --------------------------------------
LEVEL_MAX_DIST = 0.6             # Schraube weiter als 0,6 Wirbelabstände vom nächsten Centroid (entlang Achse) → keine Höhe


def implant_levels(dets: list[dict], centroids: dict, plane: str) -> list[dict]:
    """Detektionen (Stage-Koordinaten, ends[0] = Kopf bzw. Einbringseite) + Centroide {Label: (x, y)} derselben
    Aufnahme → je Implantat 'level' und 'side'.
      Schraube: Wirbel mit dem kleinsten Abstand des Schraubenmittelpunkts entlang der lokalen Achse
                (Nachbar-Centroide); weiter als LEVEL_MAX_DIST Abstände → level None.
      Iliumschraube: level 'Becken'.  Cage: Segment, dessen Centroide den Cage-Mittelpunkt einschließen.
      side (nur a.p., Patientenseite 'L'/'R'): Schraube = Lage zum Centroid, Cage = Einbringseite (ends[0])."""
    labs = [k for k in SPINE_LABELS if k in centroids]
    c = np.array([centroids[k] for k in labs], float).reshape(-1, 2)
    out = []
    for d in dets:
        e = np.asarray(d['ends'], float)
        mid = e.mean(0)
        level = side = None
        if d['cls'] == 'iliac':
            level = 'Becken'
        elif len(c) >= 2 and d['cls'] == 'pedicle':
            ax = np.gradient(c, axis=0)
            ax /= np.linalg.norm(ax, axis=1, keepdims=True)
            dist = np.abs(np.einsum('ij,ij->i', mid - c, ax))
            i = int(np.argmin(dist))
            sp = np.median(np.linalg.norm(np.diff(c, axis=0), axis=1))
            level = labs[i] if dist[i] <= LEVEL_MAX_DIST * sp else None
            if plane == 'ap' and level:
                side = 'L' if (mid[0] - c[i][0]) * FACING['ap'] > 0 else 'R'
        elif len(c) >= 2 and d['cls'] == 'cage':
            v = np.diff(c, axis=0)
            s = np.einsum('ij,ij->i', mid - c[:-1], v) / np.einsum('ij,ij->i', v, v)
            k = int(np.argmin(np.abs(s - 0.5)))
            if -0.1 <= s[k] <= 1.1 and SPINE_LABELS.index(labs[k + 1]) == SPINE_LABELS.index(labs[k]) + 1:
                level = f'{labs[k]}/{labs[k + 1]}'
            if plane == 'ap':
                side = 'L' if (e[0][0] - mid[0]) * FACING['ap'] > 0 else 'R'
        out.append({**d, 'level': level, 'side': side})
    return out


def fusion_range(screw_levels, pelvis: bool = False) -> list[str]:
    """Wirbel mit Schrauben (+ Iliumschrauben) → fusionierte Segmente von UIV bis LIV (bzw. S1 bei Becken)."""
    idx = sorted({SPINE_LABELS.index(v) for v in screw_levels if v in SPINE_LABELS})
    if pelvis and idx:
        idx.append(SPINE_LABELS.index('S1'))
    if len(idx) < 2 and not (pelvis and idx):
        return []
    lo, hi = min(idx), max(idx)
    return [f'{SPINE_LABELS[k]}/{SPINE_LABELS[k + 1]}' for k in range(lo, hi)]


def segment_classes(group: str, levels: list[str], segs) -> dict[str, str]:
    """Segmentklasse je Segment (§3.7). Fusion: fusioniert, UIV+1, UIV+2, LIV−1, fern; Dekompression:
    dekomprimiert, benachbart, fern; konservativ: alle. levels = fusionierte bzw. dekomprimierte Segmente."""
    if group == 'konservativ' or not levels:
        return {s: 'alle' if group == 'konservativ' else 'unbekannt' for s in segs}
    top, bot = _seg_key(levels[0]), _seg_key(levels[-1])
    out = {}
    for s in segs:
        k = _seg_key(s)
        if s in levels:
            out[s] = 'fusioniert' if group == 'Fusion' else 'dekomprimiert'
        elif group == 'Fusion':
            out[s] = {top - 1: 'UIV+1', top - 2: 'UIV+2', bot + 1: 'LIV−1'}.get(k, 'fern')
        else:
            out[s] = 'benachbart' if k in (top - 1, bot + 1) else 'fern'
    return out


def episode_implants(ep, imp: pd.DataFrame) -> dict:
    """Episode + seg_implants des Patienten (eine Zeile je Implantat, cls 'NONE' für Bilder ohne Implantat;
    Spalten date, plane, cls, level, side) → Implantate an der Baseline-Visite und an der ersten Verlaufsvisite.
    Fusionsstrecke aus den Schrauben der ersten Verlaufsvisite (a.p. ∪ seitlich), Cages mit Segment und Seite."""
    e = pd.Timestamp(ep['date'])
    nxt = ep.get('next_event')
    nxt = pd.Timestamp(nxt) if nxt is not None and not pd.isna(nxt) else None
    dates = sorted(imp['date'].unique())
    pre = [d for d in dates if d <= e]
    post = [d for d in dates if d > e and (nxt is None or d <= nxt)]
    screws = lambda d: {v for v in imp.loc[(imp['date'] == d) & (imp['cls'] == 'pedicle'), 'level'] if v}
    out = {'implant_pre': sorted(screws(pre[-1]), key=SPINE_LABELS.index) if pre else None,
           'implant_post': None, 'fused_implant': [], 'pelvis_implant': False, 'cages': []}
    if post:
        p = imp[imp['date'] == post[0]]
        out['implant_post'] = sorted(screws(post[0]), key=SPINE_LABELS.index)
        out['pelvis_implant'] = bool((p['cls'] == 'iliac').any())
        out['fused_implant'] = fusion_range(out['implant_post'], out['pelvis_implant'])
        cg = p[(p['cls'] == 'cage') & p['level'].notna()]
        sides = cg[cg['plane'] == 'ap'].groupby('level')['side'].agg(lambda s: '+'.join(sorted(set(s.dropna()))))
        out['cages'] = [(lv, sides.get(lv) or None) for lv in sorted(set(cg['level']), key=_seg_key)]
    return out


def fusion_qc(text_levels: list[str], implant_levels: list[str]) -> str:
    if not implant_levels:
        return 'nur Text' if text_levels else 'keine Angabe'
    if not text_levels:
        return 'nur Implantat'
    return 'gleich' if set(text_levels) == set(implant_levels) else 'abweichend'


def choose_fusion(text: list[str], n_text, implant: list[str], implant_pre) -> tuple[list[str], list[str], str]:
    """Fusionsstrecke einer Episode → (neu fusioniert, insgesamt fusioniert, Quelle).
    Befund 24.11.2026 (QC-Liste): Die Implantat-Strecke ist an UIV/LIV um ±1 unsicher, weil die Rater die Wirbel auf
    a.p. und seitlich teils verschieden zählen. Der Freitext ist teils unvollständig (z. B. 'L1/L2, L2/L3, L5/S1' bei
    8 Segmenten). Regel: Text, wenn seine Segmentzahl zu fused_segments passt, sonst Implantat. Insgesamt fusioniert
    = diese Strecke ∪ die schon präop. instrumentierte Strecke (Revision/Verlängerung: Text nennt nur das Neue)."""
    n = int(n_text) if n_text is not None and not pd.isna(n_text) else 0
    if text and n == len(text):
        new, src = list(text), 'Text'
    elif implant:
        new, src = list(implant), 'Implantat'
    else:
        new, src = list(text), 'Text' if text else 'keine'
    total = sorted(set(new) | set(fusion_range(implant_pre or [])), key=_seg_key)
    return new, total, src


# --- Schritt 4: Verlauf je Episode ------------------------------------------------------------------
def episode_course(ep, sv: pd.DataFrame) -> pd.DataFrame:
    """Wie episode_angles, aber eine Zeile je Größe × Segment × Verlaufsvisite (für F6 Korrekturverlust).
    d = post − prä, k = |prä| − |post|; loss = Rückgang in Richtung Ausgangswert seit early:
    (α(t) − α_early) · sign(α_prä − α_early), > 0 = Korrektur geht verloren (nur ab der early-Visite).
    Werte mit Spline-Fallback (edge) gehen in keine Differenz ein."""
    e = pd.Timestamp(ep['date'])
    nxt = ep.get('next_event')
    nxt = pd.Timestamp(nxt) if nxt is not None and not pd.isna(nxt) else None
    rows = []
    for plane, g in sv.groupby('plane'):
        dates = sorted(g['date'].unique())
        pre = [d for d in dates if d <= e]
        post = [d for d in dates if d > e and (nxt is None or d <= nxt)]
        if not pre or not post:
            continue
        d0 = pre[-1]
        early = post[0] if (post[0] - e).days <= EARLY_MAX_DAYS else None
        val = g.pivot_table(index='seg', columns='date', values='angle')
        fb = g.pivot_table(index='seg', columns='date', values='edge', aggfunc='max').fillna(False).astype(bool)
        get = lambda sg, d: (val.at[sg, d] if d in val.columns else np.nan,
                             bool(fb.at[sg, d]) if d in fb.columns else False)
        for sg in sorted(val.index, key=_seg_key):
            a0, f0 = get(sg, d0)
            ae, fe = get(sg, early) if early is not None else (np.nan, True)
            for d in post:
                a, f = get(sg, d)
                if np.isnan(a):
                    continue
                ok = not (f0 or f) and not np.isnan(a0)
                ok_e = early is not None and d >= early and not (fe or f) and not np.isnan(ae)
                rows.append({'plane': plane, 'seg': sg, 'date': d, 'months': (d - e).days / 30.4375,
                             'value': a, 'pre': a0, 'date_pre': d0, 'early': ae,
                             'd': a - a0 if ok else np.nan, 'k': abs(a0) - abs(a) if ok else np.nan,
                             'loss': (a - ae) * np.sign(a0 - ae) if ok_e and ok else np.nan,
                             'is_early': d == early, 'is_late': d == post[-1]})
    return pd.DataFrame(rows)
