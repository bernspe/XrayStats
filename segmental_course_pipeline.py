"""Pipeline segmentaler Therapieverlauf (Konzept: docs/degen_skoliose_verlauf_segmental_bottom_up.md).

Schritt 0: validierte Patienten → Therapie-Episoden mit Baseline/Verlauf → data/seg_episodes.pkl.
Schritt 2: Segmentwinkel je Messung (seg_angles.pkl), Mittel je Visite (seg_visits.pkl), Messfehler je Segment
           (seg_mdc.pkl, aus ALLEN Stehaufnahmen, neu sobald degen_raw.pkl neuer ist).
Schritt 2b: Slip je Segment (seg_slip.pkl; in seg_visits/seg_mdc als plane = 'slip'): manuelle Ecken aus
           corner_annotations.csv vor Modell (POST /api/suggestcorners/, Cache seg_corner_suggestions.json).
           Fehlen Modellwerte und gibt es keinen Token, wartet der Patient (§3.8).
Schritt 3: Implantat-Detektion je Stehaufnahme (implant_infer, Cache seg_implant_dets.json) → seg_implants.pkl mit
           Höhe/Seite; Episoden bekommen Implantate prä/post, Fusionsstrecke, Cages, QC gegen den Freitext.
           Modell-Slip an Segmenten mit Schrauben in beiden Wirbeln (gleiches Bild) wird verworfen.
Schritt 4: regionale/globale Parameter aus degen_params je Visite (seg_visits, plane = 'regional', seg = Parameter);
           seg_changes.pkl = je auswertbarer Episode × Größe × Segment × Verlaufsvisite Δ, K, Korrekturverlust
           und Segmentklasse – Grundlage für F1–F6 (segmental_analysis.py).
Inkrementell: berechnet nur validierte Patienten, die noch nicht in seg_episodes.pkl stehen (§3.8).

    python segmental_course_pipeline.py              # neue Patienten aus den degen_*-Caches
    python segmental_course_pipeline.py --rebuild    # alles neu (nach Codeänderung)
    XRAY_TOKEN=… python segmental_course_pipeline.py --rebuild   # mit Abfrage des Slip-Modells
    XRAY_TOKEN=… python segmental_course_pipeline.py --check-slip  # Modell vs. manuell auf annotierten Bildern
Frische API-Daten: run(token) = degen_pipeline.run(token) + update().
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import requests
from PIL import Image

import numpy as np
import pandas as pd

import degen_pipeline as dp
from spine_geom import SPINE_LABELS
from spine_geom.frames import view_of
from spine_geom.landmarks import ENDPLATE_MODES, _line, pick_endplate, spacing
import implant_infer
from spine_geom.segmental import (choose_fusion, corner_slips, episode_course, episode_implants, episodes, fusion_qc,
                                  implant_levels, segment_angles, segment_classes, therapy_events)

PIPELINE_VERSION = 's4-2026-11-25'
VALID_STATUS = {'VALID', 'VALIDATED'}           # API-Wert ist 'VALID'
STANDING = ('Xray-LWS-ap', 'Xray-WS-ap', 'Xray-LWS-lat', 'Xray-WS-lat')
MIN_NODES = 4                                    # Frontend-Spline ist kubisch: mind. 4 Nodes
EPISODES = f'{dp.DATA_DIR}/seg_episodes.pkl'
ANGLES = f'{dp.DATA_DIR}/seg_angles.pkl'
VISITS = f'{dp.DATA_DIR}/seg_visits.pkl'
MDC = f'{dp.DATA_DIR}/seg_mdc.pkl'
RAW = f'{dp.DATA_DIR}/degen_raw.pkl'
SLIPS = f'{dp.DATA_DIR}/seg_slip.pkl'
CORNERS = f'{dp.DATA_DIR}/corner_annotations.csv'
SUGGEST = f'{dp.DATA_DIR}/seg_corner_suggestions.json'
DETS = f'{dp.DATA_DIR}/seg_implant_dets.json'
IMPLANTS = f'{dp.DATA_DIR}/seg_implants.pkl'
CHANGES = f'{dp.DATA_DIR}/seg_changes.pkl'
# regionale/globale Parameter aus degen_pipeline (Status 'ok'), Definitionen dort; nur zur Einordnung (F5a)
REGIONAL = ('cCobb_max', 'C7_CSVL_wkd', 'L4_tilt', 'L5_tilt', 'LL', 'L1PA', 'T1PA', 'SVA_wkd', 'PT', 'SS')
STAGE_H = 600
LATERAL = ('Xray-LWS-lat', 'Xray-WS-lat')
# OOF-Fehler des Slip-Modells auf seitlichem Röntgen (Xray-Model runs/slip-20261021), SEM ≈ 1,25 · MAE (Normalannahme)
MODEL_SLIP_MAE = 4.5


def patients(raw: pd.DataFrame) -> pd.DataFrame:
    """referring_patient (an jeder Messung) → eine Zeile je Patient: uuid, patient_id, index_date, valid, image_groups."""
    rows = {}
    for rp in raw['referring_patient']:
        if isinstance(rp, dict) and rp.get('uuid'):
            rows[rp['uuid']] = {'patient': rp['uuid'], 'patient_id': rp.get('patient_id'),
                                'index_date': rp.get('index_date'), 'image_groups': rp.get('image_groups') or [],
                                'valid': any((t or {}).get('status') in VALID_STATUS for t in rp.get('progresstag') or [])}
    return pd.DataFrame(rows.values(), columns=['patient', 'patient_id', 'index_date', 'image_groups', 'valid'])


def _n_nodes(xmd) -> int:
    return len(((xmd or {}).get('stateObject') or {}).get('nodes') or []) if isinstance(xmd, dict) else 0


def measurements(raw: pd.DataFrame, pats: pd.DataFrame, progress=print) -> pd.DataFrame:
    """Stehaufnahmen mit ≥ MIN_NODES Nodes, eine Zeile je Messung, mit Visitendatum. Bilder einer image_group
    = eine Visite, Datum = spätestes Bilddatum der Gruppe (alle Bildtypen)."""
    r = raw[raw['referring_patient'].map(lambda rp: isinstance(rp, dict))].copy()
    r['patient'] = r['referring_patient'].map(lambda rp: rp.get('uuid'))
    r['date'] = pd.to_datetime(r['acquisition_date'], errors='coerce').dt.normalize()
    if 'uuid' in r:
        img_date = r.drop_duplicates('uuid').set_index('uuid')['date']
        group_date = {}
        for gs in pats['image_groups']:
            for g in gs:
                ds = img_date.reindex(g.get('uuids') or []).dropna()
                group_date.update({u: ds.max() for u in ds.index})
        r['date'] = r['uuid'].map(group_date).fillna(r['date'])
    else:
        progress('Hinweis: degen_raw.pkl ohne Bild-uuid (Cache vor 24.11.2026) – image_groups werden ignoriert. '
                 'Frischer Lauf mit Token behebt das.')
    r = r[r['img_type'].isin(STANDING) & (r['xray_measurement_data'].map(_n_nodes) >= MIN_NODES) & r['date'].notna()]
    r['plane'] = r['img_type'].map(view_of)
    return r.rename(columns={'id': 'image_id'})


def visits(raw: pd.DataFrame, pats: pd.DataFrame, progress=print) -> pd.DataFrame:
    """→ (patient, date, plane, image_id) je Stehaufnahme mit Nodes."""
    return measurements(raw, pats, progress)[['patient', 'date', 'plane', 'image_id']].drop_duplicates()


def s1_endplate(xmd: dict, labels) -> np.ndarray | None:
    """Gezeichnete S1-Deckplatte, nur die Richtung zählt. Vorrang: `lowerLine` eines gespeicherten Cobb-Winkels
    (Cobb-segmental vor Cobb-regional) mit vertebra_lower = S1 – vom Frontend beschriftet, nach der
    Labelkorrektur mitverschoben. Sonst geometrische Zuordnung der unbeschrifteten cobbLines wie degen_pipeline
    (spine_geom.landmarks.pick_endplate). Kein Tangenten-Fallback – den übernimmt segment_angles (s1_spline)."""
    if 'S1' not in labels:
        return None
    for kind in ('Cobb-segmental', 'Cobb-regional'):
        for a in xmd.get('angles') or []:
            ln = _line(a.get('lowerLine') or {})
            if a.get('cobbType') == kind and (a.get('cobbAngle') or {}).get('vertebra_lower') == 'S1' \
                    and ln is not None and np.any(ln[1] - ln[0]):
                return ln
    so = xmd.get('stateObject') or {}
    c = {lab: np.array([n['x'], n['y']], dtype=float) for lab, n in zip(labels, so.get('nodes') or [])}
    sp = spacing(c)
    lines = [ln for d in (so.get('cobbLines') or []) if d.get('mode') in ENDPLATE_MODES
             for ln in [_line(d)] if ln is not None and np.any(ln[1] - ln[0])]
    return pick_endplate(lines, c, 'S1', sp) if sp and lines else None


def angles_of(meas: pd.DataFrame) -> pd.DataFrame:
    """Messungen → eine Zeile je Messung × Segment (Frontend-Spline, mit Vorzeichen; L5/S1 gegen die gezeichnete
    S1-Deckplatte, §3.3). Labels aus vertebra_range (nach k-Means-Labelkorrektur); passt die Länge nicht → übersprungen."""
    rows = []
    for m in meas.itertuples():
        so = m.xray_measurement_data.get('stateObject') or {}
        nodes, vr = so.get('nodes') or [], so.get('vertebra_range')
        if not vr or vr[1] - vr[0] + 1 != len(nodes):
            continue
        labels = SPINE_LABELS[vr[0]:vr[1] + 1]
        for seg, a in segment_angles([(n['x'], n['y']) for n in nodes], labels, m.plane,
                                     s1_endplate(m.xray_measurement_data, labels)).items():
            rows.append((m.patient, m.date, m.plane, m.img_type.split('-')[1], m.image_id, m.m_uuid, m.rater_id,
                         seg, a['angle'], a['src'], a['edge']))
    return pd.DataFrame(rows, columns=['patient', 'date', 'plane', 'view', 'image_id', 'm_uuid', 'rater_id',
                                       'seg', 'angle', 'src', 'edge'])


def visit_means(ang: pd.DataFrame) -> pd.DataFrame:
    """Rater-Mittel je Bild, dann Mittel je Visite (Patient × Datum × Ebene × Segment).
    sd_rater = Wurzel der mittleren Rater-Varianz der Bilder (NaN bei nur einem Rater).
    L5/S1-Spline-Fallbacks (edge) fallen weg, sobald in derselben Visite ein Wert mit S1-Deckplatte vorliegt –
    sonst mittelte die Visite zwei Methoden (Deckplatte ~13°, Spline ~4°)."""
    good = (~ang['edge']).groupby([ang[k] for k in ('patient', 'date', 'plane', 'seg')]).transform('any')
    ang = ang[~(ang['edge'] & good)]
    img = ang.groupby(['patient', 'date', 'plane', 'image_id', 'seg']).agg(
        angle=('angle', 'mean'), var=('angle', 'var'), n_rater=('angle', 'size'), edge=('edge', 'any'),
        src=('src', lambda x: '+'.join(sorted(set(x)))), view=('view', 'first')).reset_index()
    v = img.groupby(['patient', 'date', 'plane', 'seg']).agg(
        angle=('angle', 'mean'), sd_rater=('var', lambda x: np.sqrt(x.mean())), n_img=('angle', 'size'),
        n_rater=('n_rater', 'sum'), edge=('edge', 'any'), view=('view', lambda x: '+'.join(sorted(set(x)))),
        src=('src', lambda x: '+'.join(sorted({t for v in x for t in v.split('+')}))))
    return v.reset_index()


def mdc_table(ang: pd.DataFrame) -> pd.DataFrame:
    """SEM je Ebene × Segment = gepoolte SD zwischen Ratern desselben Bildes; MDC95 = 1,96 · √2 · SEM (§3.5).
    ponytail: nur Rater-Streuung, ohne Lagerungsfehler zwischen Visiten → MDC eher zu klein (im Konzept vermerkt)."""
    g = ang[~ang['edge']].groupby(['plane', 'seg', 'image_id'])['angle'].agg(['var', 'size']).query('size >= 2')
    g['ss'] = g['var'] * (g['size'] - 1)
    t = g.groupby(['plane', 'seg']).agg(ss=('ss', 'sum'), df=('size', lambda n: int((n - 1).sum())),
                                       n_img=('size', 'size'))
    t['sem'] = np.sqrt(t['ss'] / t['df'])
    t['mdc95'] = 1.96 * np.sqrt(2) * t['sem']
    return t.drop(columns='ss').reset_index()


# --- Slip (Schritt 2b) ------------------------------------------------------------------------------
def manual_slips(corners: pd.DataFrame) -> pd.DataFrame:
    """corner_annotations.csv → je Bild × Segment Slip in % WKD (Mittel der Durchgänge), Durchgangsdifferenz,
    instr_both. Ecken und Nodes liegen im Stage-Raum (Höhe 600). WKD = calibration.length der annotierten Messung,
    fehlt sie: Median-Verhältnis WKD/Nodeabstand × Nodeabstand (wie train_slip.ipynb)."""
    c = corners[corners['level'] != 'SKIP'].dropna(subset=['cx', 'cy'])
    sp = c.groupby(['image_id', 'pass']).apply(
        lambda g: np.median(np.hypot(np.diff(g['cx']), np.diff(g['cy']))) if len(g) > 1 else np.nan)
    wkd = c.groupby(['image_id', 'pass'])['wkd'].first()
    wkd = wkd.fillna(sp * float((wkd / sp).dropna().median()))
    rows = []
    for (img, ps), g in c.groupby(['image_id', 'pass']):
        cor = {r.level: ((r.ps_x, r.ps_y), (r.pi_x, r.pi_y)) for r in g.itertuples()}
        instr = set(g.loc[g['instrumented'].astype(str) == 'True', 'level'])
        for seg, v in corner_slips(cor, wkd.get((img, ps), np.nan)).items():
            rows.append((img, ps, seg, v, set(seg.split('/')) <= instr))
    d = pd.DataFrame(rows, columns=['image_id', 'pass', 'seg', 'slip', 'instr_both'])
    piv = d.pivot_table(index=['image_id', 'seg'], columns='pass', values='slip')
    out = d.groupby(['image_id', 'seg']).agg(slip=('slip', 'mean'), instr_both=('instr_both', 'any')).reset_index()
    out['pass_diff'] = (piv[1] - piv[2]).reindex(pd.MultiIndex.from_frame(out[['image_id', 'seg']])).values \
        if {1, 2} <= set(piv.columns) else np.nan
    return out


def _image_height(url: str) -> int:
    """Nur der Bild-Header wird gelesen (signierte GCS-URL, ohne Bearer-Token)."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        return Image.open(r.raw).size[1]


def fetch_corner_suggestions(meas: pd.DataFrame, token: str, progress=print) -> dict:
    """Fehlende Modellantworten für seitliche Messungen holen (Cache je m_uuid). Inferenzfehler (HTTP 4xx/5xx)
    werden als {'error'} gecacht, Netzfehler nicht. Speichert alle 10 Abfragen."""
    cache = json.load(open(SUGGEST)) if os.path.exists(SUGGEST) else {}
    todo = [m for m in meas.itertuples() if m.m_uuid not in cache]
    for k, m in enumerate(todo, 1):
        so = m.xray_measurement_data['stateObject']
        labels = SPINE_LABELS[so['vertebra_range'][0]:so['vertebra_range'][1] + 1]
        try:
            f = _image_height(m.img) / STAGE_H
            body = {'image': m.uuid, 'nodes': [{'label': lab, 'x': n['x'] * f, 'y': n['y'] * f}
                                               for lab, n in zip(labels, so['nodes'])]}
            cal = (m.xray_measurement_data.get('calibration') or {}).get('length')
            if cal:
                body['wkd'] = cal * f
            r = requests.post(f'{dp.API_URL}/api/suggestcorners/', json=body,
                              headers={'Authorization': f'Bearer {token}'}, timeout=120)
        except requests.RequestException as e:
            progress(f'Slip-Modell: {m.m_uuid} nicht erreichbar ({e}) – nächster Lauf versucht es erneut')
            continue
        if r.status_code in (401, 403):
            progress('Slip-Modell: Token ungültig oder abgelaufen – Abbruch.')
            break
        cache[m.m_uuid] = {'f': f, 'resp': r.json()} if r.ok else {'error': f'{r.status_code}: {r.text[:200]}'}
        if k % 10 == 0 or k == len(todo):
            json.dump(cache, open(SUGGEST, 'w'))
            progress(f'Slip-Modell: {k}/{len(todo)} Messungen')
    json.dump(cache, open(SUGGEST, 'w'))
    return cache


def slips_of(meas: pd.DataFrame, manual: pd.DataFrame, cache: dict,
             screws: dict | None = None) -> tuple[pd.DataFrame, set]:
    """Seitliche Messungen → Slip je Segment. Je Bild × Segment hat die manuelle Annotation Vorrang (eine Zeile,
    ohne Rater); sonst Modellwert je Messung (Rater-Nodes). Modellwerte an Segmenten, deren beide Wirbel im selben
    Bild Pedikelschrauben tragen (screws: {image_id: {Label}}), entfallen: dort gibt das Modell nur seinen Prior aus.
    Rückgabe (Tabelle, m_uuids ohne Modellantwort)."""
    screws = screws or {}
    lat = meas[meas['img_type'].isin(LATERAL)]
    man = manual[manual['image_id'].isin(set(lat['image_id']))]
    man_keys = set(zip(man['image_id'], man['seg']))
    img_meta = lat.drop_duplicates('image_id').set_index('image_id')[['patient', 'date', 'img_type']]
    rows = [(*img_meta.loc[r.image_id], r.image_id, None, None, r.seg, r.slip, 'manuell', np.nan, r.pass_diff,
             r.instr_both) for r in man.itertuples()]
    missing = set()
    for m in lat.itertuples():
        hit = cache.get(m.m_uuid)
        if hit is None:
            if m.image_id not in set(man['image_id']):            # manuell annotiertes Bild braucht kein Modell
                missing.add(m.m_uuid)
            continue
        for s in (hit.get('resp') or {}).get('segments') or []:
            if s.get('slip') is not None and (m.image_id, s['segment']) not in man_keys \
                    and not set(s['segment'].split('/')) <= screws.get(m.image_id, set()):
                rows.append((m.patient, m.date, m.img_type, m.image_id, m.m_uuid, m.rater_id, s['segment'],
                             float(s['slip']), 'Modell', s.get('unc', np.nan), np.nan, False))
    cols = ['patient', 'date', 'img_type', 'image_id', 'm_uuid', 'rater_id', 'seg', 'slip', 'src', 'unc',
            'pass_diff', 'instr_both']
    return pd.DataFrame(rows, columns=cols), missing


def fetch_implants(meas: pd.DataFrame, progress=print) -> dict:
    """Implantat-Detektion für alle Bilder ohne Cache-Eintrag (Schlüssel image_id). Nicht ladbare Bilder
    (signierte URL älter als 24 h, Netz) werden nicht gecacht → der Patient wartet auf einen Daten-Refresh."""
    cache = json.load(open(DETS)) if os.path.exists(DETS) else {}
    todo = meas.drop_duplicates('image_id')
    todo = todo[~todo['image_id'].astype(str).isin(cache)]
    for k, m in enumerate(todo.itertuples(), 1):
        try:
            cache[str(m.image_id)] = implant_infer.detect_url(m.img)
        except requests.RequestException as e:
            progress(f'Implantate: Bild {m.image_id} nicht ladbar ({str(e)[:80]})')
        if k % 20 == 0 or k == len(todo):
            json.dump(cache, open(DETS, 'w'))
            progress(f'Implantate: {k}/{len(todo)} Bilder')
    return cache


def implants_of(meas: pd.DataFrame, cache: dict) -> tuple[pd.DataFrame, set]:
    """Detektionen → eine Zeile je Implantat mit Höhe/Seite (Centroide = Rater-Mittel je Label desselben Bildes);
    Bild ohne Implantat → eine Zeile cls 'NONE'. Rückgabe (Tabelle, image_ids ohne Detektion im Cache)."""
    rows, missing = [], set()
    for img_id, g in meas.groupby('image_id'):
        dets = cache.get(str(img_id))
        if dets is None:
            missing.add(img_id)
            continue
        pts = {}
        for x in g['xray_measurement_data']:
            so = x['stateObject']
            for lab, n in zip(SPINE_LABELS[so['vertebra_range'][0]:so['vertebra_range'][1] + 1], so['nodes']):
                pts.setdefault(lab, []).append((n['x'], n['y']))
        m = g.iloc[0]
        base = {'patient': m['patient'], 'date': m['date'], 'plane': m['plane'], 'img_type': m['img_type'],
                'image_id': img_id}
        lv = implant_levels(dets, {k: np.mean(v, 0) for k, v in pts.items()}, m['plane'])
        rows += [{**base, **d} for d in lv] or [{**base, 'cls': 'NONE'}]
    cols = ['patient', 'date', 'plane', 'img_type', 'image_id', 'cls', 'score', 'ends', 'level', 'side']
    return pd.DataFrame(rows).reindex(columns=cols), missing


def regional_of(meas: pd.DataFrame) -> pd.DataFrame:
    """degen_params (Status 'ok') der Messungen → Format von seg_angles (plane = 'regional', seg = Parameter)."""
    par = pd.read_pickle(f'{dp.DATA_DIR}/degen_params.pkl')
    dm = pd.read_pickle(f'{dp.DATA_DIR}/degen_measure.pkl')[['m_uuid']]
    par = par[(par['status'] == 'ok') & par['param'].isin(REGIONAL)].join(dm, on='idx')
    m = meas[['m_uuid', 'patient', 'date', 'image_id', 'rater_id', 'img_type']].merge(par, on='m_uuid')
    return pd.DataFrame({'patient': m['patient'], 'date': m['date'], 'plane': 'regional',
                         'view': m['img_type'].str.split('-').str[1], 'image_id': m['image_id'], 'm_uuid': m['m_uuid'],
                         'rater_id': m['rater_id'], 'seg': m['param'], 'angle': m['value'], 'src': 'degen',
                         'edge': False})


def changes_of(ep: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Auswertbare Episoden → seg_changes (episode_course + Episodenmerkmale + Segmentklasse)."""
    parts = []
    for _, e in ep[ep['status'] == 'auswertbar'].iterrows():
        c = episode_course(e, visits[visits['patient'] == e['patient']])
        if c.empty:
            continue
        lv = e['fused_total'] if e['group'] == 'Fusion' and isinstance(e.get('fused_total'), list) else e['levels']
        seg = c['plane'] != 'regional'
        cls = segment_classes(e['group'], list(lv or []), c.loc[seg, 'seg'].unique())
        c['cls'] = c['seg'].map(cls).where(seg)
        parts.append(c.assign(patient=e['patient'], patient_id=e['patient_id'], ep_date=e['date'], group=e['group'],
                              revision=e['revision'],
                              fused_n=len(e['fused_new']) if isinstance(e.get('fused_new'), list) else np.nan,
                              pelvis=bool(e.get('pelvis_implant')) or bool(e.get('fused_to_pelvis'))))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def slip_as_angles(sl: pd.DataFrame) -> pd.DataFrame:
    """seg_slip → Format von seg_angles (plane = 'slip', angle = Slip in % WKD) für visit_means/episode_angles."""
    return pd.DataFrame({'patient': sl['patient'], 'date': sl['date'], 'plane': 'slip',
                         'view': sl['img_type'].str.split('-').str[1], 'image_id': sl['image_id'],
                         'm_uuid': sl['m_uuid'].fillna('manuell'), 'rater_id': sl['rater_id'], 'seg': sl['seg'],
                         'angle': sl['slip'], 'src': sl['src'], 'edge': False})


def slip_mdc(manual: pd.DataFrame) -> pd.DataFrame:
    """MDC95 für Slip je Segment. ponytail: konservativ eine Zahl je Segment = größere der beiden SEM
    (manuell aus Durchgang 1 vs. 2: SD(d)/√2; Modell: 1,25 · OOF-MAE), weil Episoden Quellen mischen können."""
    d = manual.dropna(subset=['pass_diff'])
    sem_man = float(d['pass_diff'].std() / np.sqrt(2)) if len(d) > 2 else np.nan
    sem = np.nanmax([sem_man, 1.25 * MODEL_SLIP_MAE])
    segs = [f'{a}/{b}' for a, b in zip(SPINE_LABELS[19:-1], SPINE_LABELS[20:])]          # T12/L1 … L5/S1
    return pd.DataFrame({'plane': 'slip', 'seg': segs, 'df': len(d), 'n_img': d['image_id'].nunique(),
                         'sem': sem, 'mdc95': 1.96 * np.sqrt(2) * sem, 'sem_manual': sem_man})


def check_slip(token: str, n: int = 40, progress=print) -> pd.DataFrame:
    """Abnahme 2b: Modell (über denselben Abfrageweg wie die Pipeline) gegen manuelle Ecken auf seitlichen
    Röntgenbildern mit beiden Quellen, je Bild eine Messung, höchstens n Bilder (alle Patienten, nicht nur Kohorte).
    Erwartung: MAE ≈ OOF-Wert (≈ 4,5 % WKD), keine systematische Vorzeichenumkehr."""
    raw = pd.read_pickle(RAW)
    manual = manual_slips(pd.read_csv(CORNERS))
    meas = measurements(raw, patients(raw), progress)
    meas = meas[meas['img_type'].isin(LATERAL) & meas['image_id'].isin(set(manual['image_id']))]
    meas = meas.drop_duplicates('image_id').head(n)
    cache = fetch_corner_suggestions(meas, token, progress)
    rows = [(m.image_id, s['segment'], float(s['slip'])) for m in meas.itertuples()
            for s in ((cache.get(m.m_uuid) or {}).get('resp') or {}).get('segments') or [] if s.get('slip') is not None]
    j = pd.DataFrame(rows, columns=['image_id', 'seg', 'model']).merge(manual, on=['image_id', 'seg'])
    d = j['model'] - j['slip']
    progress(f'{j["image_id"].nunique()} Bilder, {len(j)} Segmente: MAE {d.abs().mean():.1f} % WKD, Bias {d.mean():+.1f}, '
             f'r = {np.corrcoef(j["model"], j["slip"])[0, 1]:.2f}, Vorzeichen gleich bei |GT| ≥ 10 %: '
             f'{(np.sign(j["model"]) == np.sign(j["slip"]))[j["slip"].abs() >= 10].mean():.0%}')
    return j


def _load(path, rebuild):
    return pd.read_pickle(path) if os.path.exists(path) and not rebuild else None


def _append(path, old, new):
    pd.concat([old, new], ignore_index=True).to_pickle(path) if old is not None and len(old) else new.to_pickle(path)


def update(rebuild: bool = False, progress=print, token: str | None = None) -> pd.DataFrame:
    """Neue validierte Patienten an die seg_*-Tabellen anhängen; seg_mdc neu, wenn degen_raw oder die Ecken neuer
    sind. Mit token werden fehlende Slip-Modellwerte abgefragt; ohne bleiben solche Patienten „wartend“.
    Gibt seg_episodes zurück."""
    raw = pd.read_pickle(RAW)
    entries = pd.read_pickle(f'{dp.DATA_DIR}/degen_entries.pkl')
    old = _load(EPISODES, rebuild)
    old = old if old is not None else pd.DataFrame(columns=['patient'])
    pats = patients(raw)
    meas = None
    manual = manual_slips(pd.read_csv(CORNERS)) if os.path.exists(CORNERS) else \
        pd.DataFrame(columns=['image_id', 'seg', 'slip', 'instr_both', 'pass_diff'])
    if rebuild or not os.path.exists(MDC) or os.path.getmtime(MDC) < max(
            os.path.getmtime(RAW), os.path.getmtime(CORNERS) if os.path.exists(CORNERS) else 0):
        progress('Messfehler je Segment aus allen Stehaufnahmen …')
        meas = measurements(raw, pats, progress)
        pd.concat([mdc_table(angles_of(meas)), slip_mdc(manual)], ignore_index=True).to_pickle(MDC)
    new = pats[pats['valid'] & ~pats['patient'].isin(set(old['patient']))]
    progress(f'{int(pats["valid"].sum())} validierte Patienten, davon {len(new)} neu.')
    if new.empty:
        return old
    meas = meas if meas is not None else measurements(raw, pats, progress)
    meas = meas[meas['patient'].isin(set(new['patient']))]
    progress(f'Implantate: Detektion für {meas["image_id"].nunique()} Bilder (Cache: nur neue) …')
    imp, img_missing = implants_of(meas, fetch_implants(meas, progress))
    screws = imp[imp['cls'] == 'pedicle'].groupby('image_id')['level'].agg(lambda s: set(s.dropna())).to_dict()
    cache = json.load(open(SUGGEST)) if os.path.exists(SUGGEST) else {}
    sl, missing = slips_of(meas, manual, cache, screws)
    if missing and token:
        lat = meas[meas['m_uuid'].isin(missing)]
        progress(f'Slip-Modell: {len(lat)} Messungen ohne Antwort, frage ab …')
        cache = fetch_corner_suggestions(lat, token, progress)
        sl, missing = slips_of(meas, manual, cache, screws)
    waiting = set(meas.loc[meas['m_uuid'].isin(missing) | meas['image_id'].isin(img_missing), 'patient'])
    if waiting:                                  # unvollständig → nicht anlegen, nächster Lauf (§3.8)
        progress(f'{len(waiting)} Patienten warten: Slip-Modell fehlt für {len(missing)} Messungen (Token), '
                 f'Implantate für {len(img_missing)} Bilder (Bild-URLs > 24 h → Daten neu laden).')
        new = new[~new['patient'].isin(waiting)]
        meas, sl = meas[~meas['patient'].isin(waiting)], sl[~sl['patient'].isin(waiting)]
        imp = imp[~imp['patient'].isin(waiting)]
        if new.empty:
            return old
    vis = meas[['patient', 'date', 'plane', 'image_id']].drop_duplicates()
    ang = angles_of(meas).assign(pipeline_version=PIPELINE_VERSION)
    _append(ANGLES, _load(ANGLES, rebuild), ang)
    _append(SLIPS, _load(SLIPS, rebuild), sl.assign(pipeline_version=PIPELINE_VERSION))
    _append(IMPLANTS, _load(IMPLANTS, rebuild), imp.assign(pipeline_version=PIPELINE_VERSION))
    vm = pd.concat([visit_means(ang), visit_means(slip_as_angles(sl)), visit_means(regional_of(meas))])
    _append(VISITS, _load(VISITS, rebuild), vm.assign(pipeline_version=PIPELINE_VERSION))
    stamp = datetime.now().isoformat(timespec='seconds')
    parts = []
    for p in new.itertuples():
        ev = therapy_events(entries[entries['patient'] == p.patient])
        ep = (episodes(ev, vis[vis['patient'] == p.patient], p.index_date) if len(ev)
              else pd.DataFrame([{'patient': p.patient, 'status': 'keine Therapie'}]))
        if len(ev):
            ip = imp[imp['patient'] == p.patient]
            extra = pd.DataFrame([episode_implants(r, ip) for _, r in ep.iterrows()], index=ep.index)
            ep = pd.concat([ep, extra], axis=1)
            ep['fusion_qc'] = [fusion_qc(r['levels'], r['fused_implant']) if r['group'] == 'Fusion' else None
                               for _, r in ep.iterrows()]
            ch = [choose_fusion(r['levels'], r['fused_segments'], r['fused_implant'], r['implant_pre'])
                  if r['group'] == 'Fusion' else (None, None, None) for _, r in ep.iterrows()]
            ep['fused_new'], ep['fused_total'], ep['fused_src'] = (list(x) for x in zip(*ch))
        parts.append(ep.assign(patient_id=p.patient_id, index_date=pd.to_datetime(p.index_date),
                               added_at=stamp, pipeline_version=PIPELINE_VERSION))
    out = pd.concat([old, *parts], ignore_index=True) if len(old) else pd.concat(parts, ignore_index=True)
    out.to_pickle(EPISODES)
    _append(CHANGES, _load(CHANGES, rebuild),
            changes_of(pd.concat(parts, ignore_index=True), vm).assign(pipeline_version=PIPELINE_VERSION))
    progress(f'{len(new)} Patienten berechnet: {sum(len(x) for x in parts)} Episoden-Zeilen, '
             f'{len(ang)} Segmentwinkel aus {ang["m_uuid"].nunique()} Messungen, {len(sl)} Slip-Werte '
             f'({int((sl["src"] == "manuell").sum())} manuell), {int((imp["cls"] != "NONE").sum())} Implantate.')
    return out


def run(token: str, progress=print) -> pd.DataFrame:
    dp.run(token, progress)
    return update(progress=progress, token=token)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true', help='seg_*-Tabellen verwerfen und alles neu rechnen')
    ap.add_argument('--check-slip', action='store_true', help='Slip-Modell vs. manuelle Ecken (braucht XRAY_TOKEN)')
    args = ap.parse_args()
    if args.check_slip:
        check_slip(os.environ['XRAY_TOKEN'])
    else:
        update(args.rebuild, token=os.environ.get('XRAY_TOKEN'))
