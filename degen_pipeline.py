"""Pipeline API → Tabellen für degen_scoliosis_app.py (Plan: docs/PLAN_degen_skoliose_app.md).

Ausgaben (data/):
  degen_raw.pkl      label-korrigierte Messungen (df4d) inkl. Rohannotation – Basis für Neuberechnung
  degen_base.pkl     /api/clinicalbasedata/ (Geschlecht, Geburtsdatum)
  degen_entries.pkl  /api/clinicalentrydata/ (Therapie, Fusion, Komplikationen je Ereignis)
  degen_measure.pkl  eine Zeile pro Messung (Meta, Kalibrierung, Flags, gespeicherte globalParams)
  degen_params.pkl   long: Messung × Parameter (value, status, reason)
  degen_curves.pkl   koronare centroid_cobb-Kurven je a.p.-Messung und Methode
  degen_shape.pkl    sagittaler Formvektor: Centroide im Sakrumsystem (WKD) + Segmenthöhen
  degen_visits.pkl   Patient × Datum (Geschlecht, Alter)
  degen_calib.pkl    Level-Faktoren {(view, level): {factor, n}}
  degen_vbh.pkl      Wirbelkörperhöhe in WKD {(view, level): {factor, n}}
  degen_uncert.pkl   Phase 2: Landmarken-SD, SEM/MDC95, centroid_cobb-vs-Cobb-Paare
  reports/messunsicherheit.md  Bericht Phase 2

Neuberechnung ohne Token (nach Codeänderung): python degen_pipeline.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

import annotation_analysis as aa
from prepare_outlier_data import API_URL, fetch_data
from spine_geom.coronal import (centroid_cobb, coronal_global, curves, tilt_to_horizontal, tilts_discrete,
                                tilts_spline)
from spine_geom.descriptors import coronal_shape, lumbar_length, sagittal_shape, segment_heights
from spine_geom.frames import (calib_level_factors, facing_ap, facing_lateral, posture_of,
                               patient_scale, to_patient, vbh_factors, view_of, wkd_px)
from spine_geom.landmarks import ENDPLATE_MODES, ENDPLATES, extract, pick_endplate, spacing
from spine_geom.patient import pooled_sd, robust_sd
from spine_geom.qc import craniocaudal_monotonic
from spine_geom.sagittal import (REQUIRES, available, missing_reason,
                                 sagittal_params)
from spine_geom.uncertainty import Z95, mc_sd, mdc95, pooled_cov

DATA_DIR = 'data'
# a.p.: größte centroid_cobb-Kurve (Methode a) + koronare Normparameter nach Gharbi 2024
CORONAL = ['cCobb_max', 'FHA', 'L4_tilt', 'L5_tilt', 'CTPA', 'C7_CSVL_wkd']
PARAMS = list(REQUIRES) + CORONAL
PROTOCOL_LEVELS = ('L3', 'L4')          # Messanweisung: L3 oder L4, je nach Sichtbarkeit
# ponytail: grobe QC-Schwelle, fängt hipLine auf S1-Höhe ab (4/414 im Cache 08/2026);
# feinere Ausreißerlogik (robuste z-Werte je Patient) steckt im PI-Tab (Phase 3).
PT_PLAUSIBLE = (-30.0, 60.0)
MC_DRAWS = 100
MIN_DF_LANDMARK = 20                    # weniger Freiheitsgrade → gepoolte Centroid-Kovarianz der Ansicht
# ponytail: feste Grenzen für grobe Fehler (≈ halber Wirbelabstand bzw. 15° Deckplattenrichtung);
# darüber = Label-/Deckplatten-Verwechslung statt Platzierungsrauschen.
GROSS_DEV_WKD, GROSS_DEV_DEG = 0.5, 15.0
# Kalibrier-Konsistenz (2026-09-30, revidiert): Lendenlänge L1–S1 in WKD weicht > 15 % vom Patientenmedian
# derselben Ansicht ab → nur QC-Hinweis calib_inconsistent, keine Korrektur (eine echte Höhenänderung sieht
# genauso aus). Maßstabsfehler σ_s nur aus Aufnahmen desselben Tages.
CALIB_MAX_DEV = 0.15
MIN_SCALE_PAIRS = 10
RAW_COLS = ['id', 'uuid', 'm_uuid', 'img', 'patient_id', 'referring_patient', 'img_type', 'acquisition_date',
            'rater_id', 'rater_name', 'xray_measurement_data', 'label_shift_corrected']


def fetch_clinical_base(token: str) -> pd.DataFrame:
    resp = requests.get(f'{API_URL}/api/clinicalbasedata/',
                        headers={'Authorization': f'Bearer {token}'}, timeout=60)
    resp.raise_for_status()
    rows = [{**{k: v for k, v in r.items() if k != 'data'}, **(r.get('data') or {})} for r in resp.json()]
    return pd.DataFrame(rows)


def fetch_clinical_entries(token: str) -> pd.DataFrame:
    resp = requests.get(f'{API_URL}/api/clinicalentrydata/',
                        headers={'Authorization': f'Bearer {token}'}, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame([{**{k: v for k, v in r.items() if k != 'data'}, **(r.get('data') or {})} for r in resp.json()])


def _calib_status(lm, view, factors) -> str:
    if not lm.calib_px:
        return 'fehlt'
    if (view, lm.calib_level) not in factors:
        return 'ungültig'
    return 'ok' if lm.calib_level in PROTOCOL_LEVELS else 'off_protocol'


def _row(i, p, value=np.nan, status='fehlt', reason=None):
    return dict(idx=i, param=p, value=value, status=status, reason=reason)


def build(raw: pd.DataFrame) -> dict:
    raw = raw[raw['xray_measurement_data'].map(lambda x: isinstance(x, dict))]
    views = raw['img_type'].map(view_of)
    lms = {i: extract(x, lateral=views[i] == 'lat') for i, x in raw['xray_measurement_data'].items()}
    factors = calib_level_factors((raw.at[i, 'id'], views[i], lm.calib_level, lm.calib_px) for i, lm in lms.items())
    # Maßstab primär aus Wirbelkörperhöhen, Kalibrierlinie nur als Fallback (DECISIONS.md 2026-09-30)
    line_wkd = {i: wkd_px(lm.calib_px, views[i], lm.calib_level, factors) for i, lm in lms.items()}
    vbh = vbh_factors((views[i], lm.vb_heights, line_wkd[i]) for i, lm in lms.items())

    meta = []
    for i, lm in lms.items():
        if not lm.centroids:
            continue
        r, view = raw.loc[i], views[i]
        facing, flags = facing_lateral(lm.s1_endplate, lm.centroids) if view == 'lat' else facing_ap()
        flags |= lm.flags
        if r.get('label_shift_corrected'):
            flags.add('label_shift_corrected')
        if not craniocaudal_monotonic(lm.centroids):
            flags.add('non_monotonic')
        rp = r.get('referring_patient')
        meta.append(dict(
            idx=i, m_uuid=r['m_uuid'], image_id=r['id'], img=r.get('img'), patient_id=r['patient_id'],
            patient_uuid=rp.get('uuid') if isinstance(rp, dict) else None,
            date=pd.to_datetime(r['acquisition_date'], errors='coerce'), img_type=r['img_type'],
            view=view, posture=posture_of(r['img_type']), rater_id=r['rater_id'], rater_name=r['rater_name'],
            calib_px=lm.calib_px, calib_level=lm.calib_level, calib_status=_calib_status(lm, view, factors),
            n_vbh=sum((view, lvl) in vbh for lvl in lm.vb_heights), wkd_line_px=line_wkd[i],
            node_spacing_px=spacing(lm.centroids),
            facing=facing, qc_flags=flags, s1_source=lm.s1_source, l1_source=lm.l1_source, l4_source=lm.l4_source,
            stored_gp=r['xray_measurement_data'].get('globalParams') or {},
        ))
    meas = pd.DataFrame(meta).set_index('idx')
    meas['wkd_px'], meas['calib_source'] = np.nan, None
    for (_, view), g in meas.groupby(['patient_id', 'view'], dropna=False):
        pop = {lvl: f['factor'] for (v, lvl), f in vbh.items() if v == view}
        heights = [{k: h for k, h in lms[i].vb_heights.items() if k in pop} for i in g.index]
        s = patient_scale([(line_wkd[i], h) for i, h in zip(g.index, heights)], pop)
        meas.loc[g.index, 'wkd_px'] = [np.nan if x is None else x for x in s]
        meas.loc[g.index, 'calib_source'] = ['vbh' if h else ('line' if line_wkd[i] else None)
                                             for i, h in zip(g.index, heights)]

    # Fallback Kalibrierung: patientenspezifisches Verhältnis Node-Abstand / WKD (DECISIONS.md).
    # ponytail: Node-Abstand enthält Bandscheibenhöhe → driftet bei Degeneration; Flag calib_imputed.
    ratio = (meas['node_spacing_px'] / meas['wkd_px']).groupby([meas['patient_id'], meas['view']]).transform('median')
    imp = meas['wkd_px'].isna() & ratio.notna() & meas['node_spacing_px'].notna()
    meas.loc[imp, 'wkd_px'] = meas.loc[imp, 'node_spacing_px'] / ratio[imp]
    meas.loc[imp, 'calib_source'] = 'imputed'
    for i in meas.index[imp]:
        meas.at[i, 'qc_flags'] = meas.at[i, 'qc_flags'] | {'calib_imputed'}
    meas['lum_len_px'] = [lumbar_length(lms[i].centroids) for i in meas.index]
    scale = calib_consistency(meas)

    rows, curve_rows, shape_rows, cobb_pairs, main_ends, s1_ap = [], [], [], [], {}, {}
    for i, m in meas.iterrows():
        lm = lms[i]
        wkd = m['wkd_px'] if pd.notna(m['wkd_px']) else None
        if m['view'] == 'lat':
            vals = sagittal_params(lm, m['facing'], wkd)
            hip_bad = 'PT' in vals and not PT_PLAUSIBLE[0] <= vals['PT'] <= PT_PLAUSIBLE[1]
            if hip_bad:                                   # hipLine nicht an den Hüftköpfen → Hüfte verwerfen
                lm.hip = None
                meas.at[i, 'qc_flags'] = m['qc_flags'] | {'hip_implausible'}
                vals = sagittal_params(lm, m['facing'], wkd)
            have = available(lm, wkd)
            # Aufnahmetyp zuerst: MRT/Funktionsaufnahme liefern keinen sagittalen Parameter, egal was annotiert ist
            posture_reason = {'functional': 'Funktionsaufnahme', 'supine': 'keine Stehaufnahme'}.get(m['posture'])
            for p in REQUIRES:
                reason = posture_reason or missing_reason(p, have)
                if reason and hip_bad:
                    reason = reason.replace('keine Hüfte', 'Hüfte unplausibel')
                if reason:
                    rows.append(_row(i, p, reason=reason))
                    continue
                if p not in vals:                     # Landmarken da, aber nicht berechenbar (LDI bei LL < 10°)
                    rows.append(_row(i, p, reason='nicht berechenbar'))
                    continue
                tangent = any(getattr(lm, f'{ep}_source') == 'tangent' for ep in REQUIRES[p] if ep in ('s1', 'l1', 'l4'))
                rows.append(_row(i, p, vals[p], 'tangente' if tangent else 'ok'))
            heights = segment_heights(lm.centroids, wkd) if wkd else {}
            for lab, xy in sagittal_shape(lm, m['facing'], wkd).items():
                nxt = next((h for k, h in heights.items() if k.startswith(f'{lab}-')), np.nan)
                shape_rows.append(dict(idx=i, view='lat', label=lab, xs=xy[0], ys=xy[1], seg_h=nxt,
                                       frame_ref='S1-Deckplatte' if lm.s1_source == 'cobb' else 'S1-Tangente'))
        else:
            c = {k: to_patient(v, m['facing']) for k, v in lm.centroids.items()}
            methods = {'a': tilts_discrete(c)}
            if wkd:
                methods['b'] = tilts_spline({k: v / wkd for k, v in c.items()})
            for meth, t in methods.items():
                for cv in curves(c, t):
                    curve_rows.append(dict(idx=i, method=meth, **cv))
            if m['posture'] != 'standing':
                rows += [_row(i, p, reason='keine Stehaufnahme') for p in CORONAL]
                continue
            so = raw.at[i, 'xray_measurement_data'].get('stateObject') or {}
            sp = spacing(lm.centroids)
            lines = [np.array([[d['x1'], d['y1']], [d['x2'], d['y2']]], dtype=float)
                     for d in so.get('cobbLines') or [] if d.get('mode') in ENDPLATE_MODES]
            ep = {lab: pick_endplate(lines, lm.centroids, lab, sp) if sp and lab in lm.centroids else None
                  for lab in ('L4', 'L5', 'S1')}
            # S1-Mitte: Schnitt der S1-Deckplattenlinie mit der L5–S1-Sehne, sonst S1-Centroid (≈ Ersatz)
            s1_img = ep['S1'].mean(axis=0) if ep['S1'] is not None else lm.centroids.get('S1')
            s1_ap[i] = ep['S1'].mean(axis=0) if ep['S1'] is not None else None
            s1_status = 'ok' if ep['S1'] is not None else 'tangente'
            glob = coronal_global(c, to_patient(s1_img, m['facing']) if s1_img is not None else None, wkd)
            hl = so.get('hipLine') or {}
            # koronarer Formvektor: Beckenquerachse = S1-Deckplatte a.p. > Hüftlinie > Bildhorizontale
            if ep['S1'] is not None:
                axis, ref = ep['S1'][1] - ep['S1'][0], 'S1-Deckplatte'
            elif len(hl) >= 4:
                axis, ref = np.array([hl['x2'] - hl['x1'], hl['y2'] - hl['y1']], dtype=float), 'Hüftlinie'
            else:
                axis, ref = np.array([1.0, 0.0]), 'Bildhorizontale'
            for lab, zc in coronal_shape(lm.centroids, m['facing'], s1_img, axis, wkd).items():
                shape_rows.append(dict(idx=i, view='ap', label=lab, xs=zc[0], ys=zc[1], seg_h=np.nan, frame_ref=ref))
            vals = {'FHA': (tilt_to_horizontal([[hl['x1'], hl['y1']], [hl['x2'], hl['y2']]]), 'ok')
                    if len(hl) >= 4 else 'keine Hüfte'}
            for lab in ('L4', 'L5'):
                vals[f'{lab}_tilt'] = (tilt_to_horizontal(ep[lab]), 'ok') if ep[lab] is not None \
                    else f'keine {lab}-Deckplatte'
            vals['CTPA'] = (glob['CTPA'], s1_status) if 'CTPA' in glob else \
                ('T1 nicht annotiert' if 'T1' not in c else 'S1 nicht annotiert')
            vals['C7_CSVL_wkd'] = (glob['C7_CSVL_wkd'], s1_status) if 'C7_CSVL_wkd' in glob else \
                ('C7 nicht annotiert' if 'C7' not in c else 'S1 nicht annotiert' if s1_img is None
                 else 'keine gültige Kalibrierung')
            if len(methods['a']) < 3:
                vals['cCobb_max'] = 'zu wenige Nodes'
            else:
                cv = max(curves(c, methods['a']), key=lambda x: abs(x['angle']), default=None)
                labs = list(methods['a'])
                main_ends[i] = (cv['upper'], cv['lower']) if cv else (labs[0], labs[-1])
                vals['cCobb_max'] = (abs(centroid_cobb(methods['a'], *main_ends[i])), 'ok')
            for p in CORONAL:
                v = vals[p]
                rows.append(_row(i, p, reason=v) if isinstance(v, str) else _row(i, p, v[0], v[1]))
            if i not in main_ends:
                continue
            # manueller Cobb-regional desselben Raters bei gleichen Endwirbeln (Phase 2: Bland-Altman)
            for a in raw.at[i, 'xray_measurement_data'].get('angles') or []:
                ca = a.get('cobbAngle') or {}
                u, low = ca.get('vertebra_upper'), ca.get('vertebra_lower')
                if a.get('cobbType') == 'Cobb-regional' and u != low and u in methods['a'] and low in methods['a']:
                    cobb_pairs.append(dict(idx=i, image_id=m['image_id'], upper=u, lower=low, manual=ca.get('angle'),
                                           cc_a=abs(centroid_cobb(methods['a'], u, low)),
                                           cc_b=abs(centroid_cobb(methods['b'], u, low))
                                           if 'b' in methods and u in methods['b'] and low in methods['b'] else np.nan))
    params = pd.DataFrame(rows)
    params['sd'], uncert = uncertainty(meas, lms, params, main_ends, s1_ap, pd.DataFrame(cobb_pairs))
    shape = pd.DataFrame(shape_rows)
    uncert['change'] = change_model(meas, params, shape)
    uncert['change'].update(scale)
    return dict(measure=meas, params=params, curves=pd.DataFrame(curve_rows),
                shape=shape, calib=factors, vbh=vbh, uncert=uncert)


# ---------------------------------------------------------------------------
# Kalibrier-Konsistenz über Aufnahmen (Maßstab)
# ---------------------------------------------------------------------------

def calib_consistency(meas: pd.DataFrame) -> dict:
    """Maßstab (Kalibrierlinie) über Aufnahmen: nur QC-Hinweis und Fehlerschätzung, **keine Korrektur**.

    Referenz ist die Lendenlänge L1–S1 in px (unabhängig von der Kalibrierlinie), r = Lendenlänge / WKD.
    - Flag calib_inconsistent: r weicht > CALIB_MAX_DEV vom Patientenmedian derselben Ansicht ab (≥ 2 Bilder).
      Nur Hinweis – eine echte Höhenänderung (OP-Distraktion, Kollaps) sieht genauso aus; eine Korrektur
      hätte sie entfernt (2PVD: reale +3 cm auf 45 cm laut cm-Lineal, korrigiert blieben < 1 cm übrig).
    - scale_sd je Ansicht: relative Maßstabsstreuung zwischen verschiedenen Aufnahmen **desselben Tages**
      (dort ist keine biologische Änderung möglich), paarweise Log-Differenzen, robust (MAD), abzüglich des
      Rater-Anteils (steckt schon in den Rater-Fehlern der Koordinaten)."""
    ok = (meas['posture'] == 'standing') & meas['wkd_px'].notna() & meas['lum_len_px'].notna()
    m = meas[ok]
    ratio = m['lum_len_px'] / m['wkd_px']
    img = ratio.groupby([m['patient_id'], m['view'], m['date'], m['image_id']]).median()
    n_rater = m.groupby('image_id').size().mean()
    rater = ratio.groupby(m['image_id']).apply(lambda s: s.std() / s.mean() if len(s) > 1 else np.nan).median()

    for (pid, view), g in img.groupby(level=[0, 1]):
        if len(g) < 2:
            continue
        med = float(np.median(g))
        sel = (m['patient_id'] == pid) & (m['view'] == view)
        for i in m.index[sel]:
            if abs(ratio[i] / med - 1) > CALIB_MAX_DEV:
                meas.at[i, 'qc_flags'] = meas.at[i, 'qc_flags'] | {'calib_inconsistent'}

    out, n_pairs = {}, {}
    for view in ('lat', 'ap'):
        d = []
        for _, g in img[img.index.get_level_values(1) == view].groupby(level=[0, 2]):   # Patient × Tag
            v = np.log(g.to_numpy())
            d += [abs(a - b) for i, a in enumerate(v) for b in v[i + 1:]]
        n_pairs[view] = len(d)
        sd_img = 1.4826 * np.median(d) / np.sqrt(2) if len(d) >= MIN_SCALE_PAIRS else np.nan
        out[view] = float(np.sqrt(max(0.0, sd_img ** 2 - (rater ** 2 if pd.notna(rater) else 0) / n_rater))) \
            if pd.notna(sd_img) else np.nan
    return dict(scale_sd=out, scale_pairs=n_pairs, scale_rater_cv=float(rater) if pd.notna(rater) else np.nan,
                n_calib_flagged=int(meas['qc_flags'].map(lambda f: 'calib_inconsistent' in f).sum()))


# ---------------------------------------------------------------------------
# Phase 2: Messunsicherheit
# ---------------------------------------------------------------------------

def _landmark_groups(meas: pd.DataFrame, lms: dict) -> dict:
    """(view, kind) → Liste von Rater-Arrays je Bild, Positionen in Patienten-WKD.
    Nur Stehaufnahmen, nur Bilder mit ≥ 2 Ratern; Blickrichtung = Mehrheit am Bild."""
    groups: dict = {}
    sel = meas[(meas['posture'] == 'standing') & meas['wkd_px'].notna()]
    for (image_id, view), g in sel.groupby(['image_id', 'view']):
        if len(g) < 2:
            continue
        f = 1 if g['facing'].sum() >= 0 else -1
        # ein WKD pro Bild: Absolutpositionen durch den WKD je Rater zu teilen, erzeugt bei ~4 %
        # Kalibrier-Streuung scheinbare Abweichungen von ~0,5 WKD (Absolutwerte ~10 WKD)
        w = g['wkd_px'].mean()
        per_kind: dict = {}
        for i, m in g.iterrows():
            lm = lms[i]
            pts = dict(lm.centroids)
            if lm.hip is not None:
                pts['H'] = lm.hip
            for _, ep in ENDPLATES:
                seg = getattr(lm, f'{ep}_endplate')
                if seg is not None:
                    suf = ep.upper() + ('_t' if getattr(lm, f'{ep}_source') == 'tangent' else '')
                    pts[f'M_{suf}'] = seg.mean(axis=0)
                    d = seg[1] - seg[0]
                    per_kind.setdefault(f'ang_{suf}', []).append(np.degrees(np.arctan2(d[1], d[0])))
            for k, v in pts.items():
                per_kind.setdefault(k, []).append(to_patient(v, f) / w)
        for k, vals in per_kind.items():
            if len(vals) < 2:
                continue
            if k.startswith('ang_'):                      # Richtung mod 180°, um den ersten Wert zentriert
                vals = [vals[0] + ((v - vals[0] + 90) % 180) - 90 for v in vals]
            groups.setdefault((view, k), []).append(np.array(vals))
    return groups


def _coronal_mc_fn(m, w, ends, s1_fixed):
    """a.p.-Größen aus (verrauschten) Centroiden. ponytail: Hüftlinie und a.p.-Deckplattenlinien werden
    nicht verrauscht (Richtung aus Linien, keine eigene Kovarianz) → FHA/L4-/L5-Tilt ohne MC-SD, nur
    empirische Rater-SEM; eine gezeichnete S1-Mitte bleibt fix, sonst wandert der S1-Centroid mit."""
    def fn(x):
        c = {k: to_patient(v, m['facing']) for k, v in x.centroids.items()}
        s1 = s1_fixed if s1_fixed is not None else x.centroids.get('S1')
        out = coronal_global(c, to_patient(s1, m['facing']) if s1 is not None else None, w)
        if ends:
            out['cCobb_max'] = abs(centroid_cobb(tilts_discrete(c), *ends))
        return out
    return fn


def uncertainty(meas, lms, params, main_ends, s1_ap, cobb_pairs) -> tuple[pd.Series, dict]:
    groups = _landmark_groups(meas, lms)
    cov, lm_rows = {'lat': {}, 'ap': {}}, []
    for (view, kind), g in groups.items():
        c, df, dropped, total = pooled_cov(g, GROSS_DEV_DEG if kind.startswith('ang_') else GROSS_DEV_WKD)
        if c is None:
            continue
        cov[view][kind] = c if not kind.startswith('ang_') else float(c[0, 0])
        sds = np.sqrt(np.diag(np.atleast_2d(c)))
        lm_rows.append(dict(view=view, kind=kind, sd_x=sds[0], sd_y=sds[1] if len(sds) > 1 else np.nan, df=df,
                            grobe_fehler=dropped / total))
    lm_sd = pd.DataFrame(lm_rows)
    for view in cov:                                  # dünn besetzte Centroid-Labels → gepoolte Centroid-Kovarianz
        cen = [(k, g) for (v, k), g in groups.items() if v == view and not k[0] in 'HMa']
        pooled, *_ = pooled_cov([a for _, g in cen for a in g], GROSS_DEV_WKD)
        for k, g in cen:
            if pooled is not None and (k not in cov[view] or sum(len(a) - 1 for a in g) < MIN_DF_LANDMARK):
                cov[view][k] = pooled

    sd = pd.Series(np.nan, index=params.index)
    key = params.set_index(['idx', 'param']).index
    pos = pd.Series(range(len(params)), index=key)
    have = params.loc[params['status'] != 'fehlt', 'idx'].unique()
    for i in have:
        m, lm = meas.loc[i], lms[i]
        w = m['wkd_px'] if pd.notna(m['wkd_px']) else None
        if not w:
            continue
        if m['view'] == 'lat':
            fn = lambda x, m=m, w=w: sagittal_params(x, m['facing'], w)  # noqa: E731
        else:
            fn = _coronal_mc_fn(m, w, main_ends.get(i), s1_ap.get(i))
        for p, v in mc_sd(fn, lm, m['facing'], w, cov[m['view']], k=MC_DRAWS, seed=int(i)).items():
            if (i, p) in pos.index:
                sd.iloc[pos[(i, p)]] = v
    sd[params['status'] == 'fehlt'] = np.nan

    # SEM je Parameter: empirisch (Rater am selben Bild) vs. Monte-Carlo (Median je Messung)
    p = params.merge(meas[['image_id', 'posture']], left_on='idx', right_index=True).assign(sd=sd.values)
    sem_rows = []
    for name, g in p[p['status'] != 'fehlt'].groupby('param'):
        ok = g[g['status'] == 'ok']
        e_ok = pooled_sd(x.to_numpy() for _, x in ok.groupby('image_id')['value'])
        e_all = pooled_sd(x.to_numpy() for _, x in g.groupby('image_id')['value'])
        n_img = int((ok.groupby('image_id').size() > 1).sum())
        mc = ok['sd'].median()
        e_rob = robust_sd(x.to_numpy() for _, x in ok.groupby('image_id')['value'])
        sem_rows.append(dict(param=name, n_bilder_multirater=n_img, sem_emp=e_ok, sem_emp_inkl_tangente=e_all,
                             sem_robust=e_rob, sem_mc=mc, mdc95_emp=mdc95(e_ok), mdc95_robust=mdc95(e_rob),
                             mdc95_mc=mdc95(mc)))
    cobb_sem = []
    if len(cobb_pairs):
        grp = cobb_pairs.groupby(['image_id', 'upper', 'lower'])
        for col in ('manual', 'cc_a', 'cc_b'):
            s_ = pooled_sd(x.dropna().to_numpy() for _, x in grp[col])
            cobb_sem.append(dict(methode=col, sem_emp=s_, mdc95=mdc95(s_),
                                 n_gruppen=int((grp[col].count() > 1).sum())))
    return sd, dict(landmark_sd=lm_sd, sem=pd.DataFrame(sem_rows), cobb_pairs=cobb_pairs,
                    cobb_sem=pd.DataFrame(cobb_sem), cov=cov)


# ---------------------------------------------------------------------------
# Phase 5: Fehlermodell für Veränderungen
# ---------------------------------------------------------------------------

MIN_TAU_PAIRS = 10          # weniger Aufnahmepaare am selben Tag → τ unbekannt (Platzhalter 0 + Warnung)


def change_model(meas: pd.DataFrame, params: pd.DataFrame, shape: pd.DataFrame) -> dict:
    """Fehlermodell für Σ_Δ = Σ_mess(t₁) + Σ_mess(t₂) + 2·diag(τ²).

    Komponenten: alle Skalarparameter und die Formkoordinaten, nur Stehaufnahmen:
    seitlich '<Label>.x' (anterior) / '<Label>.y' (kranial) im Sakrumsystem, WKD (Wirbeltiefe);
    a.p. '<Label>.z' (links) / '<Label>.c' (kranial) im koronaren Beckensystem, WKD (Wirbelbreite).
    - corr: Korrelation der Messfehler aus Rater-Abweichungen am selben Bild (paarweise, ≥ 30 Paare).
    - sem: robuste Rater-SEM je Komponente (Fallback, wo keine MC-SD vorliegt).
    - tau: Aufnahmefehler aus Paaren verschiedener Bilder desselben Patienten am selben Tag –
      Var(Δ) = var₁ + var₂ + 2τ², robust über MAD. Enthält Neupositionierung, Projektion, Haltung
      innerhalb eines Tages; Haltungsschwankung *zwischen* Tagen bleibt als Platzhalter (Slider).
    """
    meta = meas[['image_id', 'patient_id', 'date', 'posture']]
    sc = params[params['status'] != 'fehlt'][['idx', 'param', 'value', 'sd']]
    sh = shape.merge(meta[['posture']], left_on='idx', right_index=True)
    sh = sh[sh['posture'] == 'standing']
    sfx = {('lat', 'xs'): '.x', ('lat', 'ys'): '.y', ('ap', 'xs'): '.z', ('ap', 'ys'): '.c'}
    long = pd.concat([sc] + [pd.DataFrame({'idx': g['idx'], 'param': g['label'] + sfx[(view, col)], 'value': g[col],
                                           'sd': np.nan})
                             for view, g in sh.groupby('view') for col in ('xs', 'ys')], ignore_index=True)
    long = long.merge(meta, left_on='idx', right_index=True)

    sem = long.groupby('param').apply(
        lambda g: robust_sd(x.to_numpy() for _, x in g.groupby('image_id')['value']), include_groups=False)
    long['var'] = long['sd'].fillna(long['param'].map(sem)) ** 2

    wide = long.pivot_table(index=['image_id', 'idx'], columns='param', values='value')
    dev = wide - wide.groupby(level='image_id').transform('mean')
    dev = dev.where(wide.groupby(level='image_id').transform('count') > 1)
    corr = dev.corr(min_periods=30)

    img = (long.groupby(['patient_id', 'date', 'image_id', 'param'])
           .agg(value=('value', 'mean'), var=('var', 'mean'), n=('value', 'size')).reset_index())
    img['var'] /= img['n']
    tau_rows = []
    for p, g in img.groupby('param'):
        d, v = [], []
        for _, day in g.groupby(['patient_id', 'date']):
            vals, var = day['value'].to_numpy(), day['var'].to_numpy()
            for a in range(len(vals)):
                for b in range(a + 1, len(vals)):
                    d.append(vals[a] - vals[b])
                    v.append(var[a] + var[b])
        tau = np.nan
        if len(d) >= MIN_TAU_PAIRS:
            s_diff = 1.4826 * np.median(np.abs(d))
            tau = float(np.sqrt(max(0.0, (s_diff ** 2 - np.mean(v)) / 2)))
        tau_rows.append(dict(param=p, tau=tau, n_pairs=len(d)))
    tau = pd.DataFrame(tau_rows).set_index('param')

    # Tag-zu-Tag-Fehler aus der PI (biologisch konstant): Paare verschiedener Tage desselben Patienten,
    # Überschuss über Mess- und Gleichtagesfehler = Positionierung/Projektion zwischen Visiten.
    pi = img[img['param'] == 'PI']
    d, v = [], []
    for _, g in pi.groupby('patient_id'):
        g = g.groupby('date').agg(value=('value', 'mean'), var=('var', 'mean')).reset_index()
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                d.append(g.at[a, 'value'] - g.at[b, 'value'])
                v.append(g.at[a, 'var'] + g.at[b, 'var'])
    tau_same = tau.at['PI', 'tau'] if 'PI' in tau.index and pd.notna(tau.at['PI', 'tau']) else 0.0
    tau_day = float(np.sqrt(max(0.0, ((1.4826 * np.median(np.abs(d))) ** 2 - np.mean(v)) / 2 - tau_same ** 2))) \
        if len(d) >= MIN_TAU_PAIRS else np.nan
    return dict(corr=corr, sem=sem, tau=tau, tau_day_pi=tau_day, n_day_pairs_pi=len(d))


# ---------------------------------------------------------------------------
# Bericht Phase 2
# ---------------------------------------------------------------------------

PARAM_TEXT = {'PI': 'PI (°)', 'SS': 'SS (°)', 'PT': 'PT (°)', 'LL': 'LL L1–S1 (°)', 'LDI': 'LDI (%)', 'L1PA': 'L1PA (°)',
              'T4PA': 'T4PA (°)', 'T1PA': 'T1PA (°)', 'T4L1PA_mismatch': 'T4-L1PA-Mismatch (°)',
              'GT': 'Global Tilt (°)', 'SVA_wkd': 'SVA (WKD)', 'cCobb_max': 'centroid_cobb max (°)',
              'FHA': 'Hüftkopfwinkel a.p. (°)', 'L4_tilt': 'L4-Tilt a.p. (°)', 'L5_tilt': 'L5-Tilt a.p. (°)',
              'CTPA': 'koronarer T1PA (°)', 'C7_CSVL_wkd': 'C7-CSVL (WKD)'}


def bland_altman(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    return dict(n=len(d), bias=d.mean(), loa=Z95 * d.std()) if len(d) > 2 else dict(n=len(d), bias=np.nan, loa=np.nan)


def cobb_agreement(pairs: pd.DataFrame) -> pd.DataFrame:
    """Bland-Altman centroid_cobb (a/b) vs. manueller Cobb-regional, gesamt und stratifiziert."""
    rows = []
    strata = {'alle': pairs, 'untere Endwirbel ≠ S1': pairs[pairs['lower'] != 'S1'],
              'untere Endwirbel = S1': pairs[pairs['lower'] == 'S1']}
    for name, g in strata.items():
        for col, meth in (('cc_a', 'a (zentrale Differenzen)'), ('cc_b', 'b (Glättungsspline)')):
            rows.append(dict(stratum=name, methode=meth, **bland_altman(g[col], g['manual'].astype(float))))
    return pd.DataFrame(rows)


def write_uncertainty_report(u: dict, meas: pd.DataFrame, path: str = 'reports/messunsicherheit.md') -> None:
    sem = u['sem'].assign(Parameter=lambda d: d['param'].map(PARAM_TEXT)).set_index('param') \
        .reindex([p for p in PARAM_TEXT if p in set(u['sem']['param'])])
    fmt = lambda x: '–' if pd.isna(x) else f'{x:.2f}'  # noqa: E731
    lines = [
        '# Messunsicherheit (Phase 2)', '',
        f'Automatisch erzeugt von `degen_pipeline.py` am {pd.Timestamp.now():%d.%m.%Y %H:%M}. '
        f'Datenbasis: {len(meas)} Messungen, {meas["image_id"].nunique()} Bilder, '
        f'{meas["patient_id"].nunique()} Patienten.', '',
        '## Methode', '',
        '- **Empirische SEM:** gepoolte SD zwischen Ratern am selben Bild (nur Deckplatte aus Cobb-Linie; '
        'zweite Spalte inkl. Tangenten-Fallback). Enthält Platzierungsrauschen *und* grobe Fehler.',
        '- **Monte-Carlo-SEM:** Landmarken-Kovarianz je Ansicht und Landmarke aus Multi-Rater-Daten '
        f'(robust: Abweichungen > {GROSS_DEV_WKD} WKD bzw. > {GROSS_DEV_DEG:.0f}° als grobe Fehler verworfen), '
        f'{MC_DRAWS} Ziehungen je Messung, Landmarken unabhängig; Median über Messungen. Beschreibt reines '
        'Platzierungsrauschen.',
        '- **Robuste SEM:** 1,4826·Median(|paarweise Rater-Differenz|)/√2 – unempfindlich gegen einzelne grobe '
        'Fehler; typische Messpräzision, wenn die Landmarken richtig identifiziert sind.',
        '- **MDC95** = 1,96·√2·SEM: kleinste Veränderung zwischen zwei Messungen, die mit 95 % kein Messrauschen ist.',
        '- **Nicht enthalten:** Haltungsvariabilität zwischen Visiten (Σ_haltung) – ohne Test-Retest-Aufnahmen '
        'nicht schätzbar. Für Visitenvergleiche ist die MDC95 daher eine **Untergrenze**.', '',
        '## SEM und MDC95 je Parameter', '',
        '| Parameter | Bilder ≥ 2 Rater | SEM emp. | SEM emp. inkl. Tangente | SEM robust | SEM MC | MDC95 emp. | '
        'MDC95 robust | MDC95 MC |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for p, r in sem.iterrows():
        lines.append(f"| {r['Parameter']} | {r['n_bilder_multirater']} | {fmt(r['sem_emp'])} | "
                     f"{fmt(r['sem_emp_inkl_tangente'])} | {fmt(r['sem_robust'])} | {fmt(r['sem_mc'])} | "
                     f"{fmt(r['mdc95_emp'])} | {fmt(r['mdc95_robust'])} | {fmt(r['mdc95_mc'])} |")
    lm = u['landmark_sd']
    key = lm[lm['kind'].isin(['C7', 'T1', 'T4', 'L1', 'L3', 'L5', 'S1', 'H', 'M_S1', 'M_L1', 'ang_S1', 'ang_L1'])]
    lines += ['', '## Landmarken-Streuung (Platzierungsrauschen, robust)', '',
              'Positionen in WKD (Patientenkoordinaten: seitlich x = anterior, a.p. x = Patient links; y = kranial), '
              'Richtungen in Grad. „Grobe Fehler“ = Anteil verworfener Rater-Werte (Label-/Deckplattenverwechslung).', '',
              '| Ansicht | Landmarke | SD x | SD y | FG | grobe Fehler |', '|---|---|---|---|---|---|']
    for _, r in key.sort_values(['view', 'kind']).iterrows():
        lines.append(f"| {r['view']} | {r['kind']} | {fmt(r['sd_x'])} | {fmt(r['sd_y'])} | {r['df']} | "
                     f"{100 * r['grobe_fehler']:.1f} % |")
    ag = cobb_agreement(u['cobb_pairs']) if len(u['cobb_pairs']) else pd.DataFrame()
    lines += ['', '## centroid_cobb vs. manueller Cobb (a.p., gleiche Endwirbel, gleicher Rater)', '',
              '| Stratum | Methode | n | Bias (°) | LoA ± (°) |', '|---|---|---|---|---|']
    for _, r in ag.iterrows():
        lines.append(f"| {r['stratum']} | {r['methode']} | {r['n']} | {fmt(r['bias'])} | {fmt(r['loa'])} |")
    lines += ['', 'Rater-SEM bei festen Endwirbeln (Gruppen = Bild × Endwirbelpaar, ≥ 2 Rater):', '',
              '| Methode | SEM (°) | MDC95 (°) | Gruppen |', '|---|---|---|---|']
    for _, r in u['cobb_sem'].iterrows():
        lines.append(f"| {r['methode']} | {fmt(r['sem_emp'])} | {fmt(r['mdc95'])} | {r['n_gruppen']} |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def build_visits(meas: pd.DataFrame, base: pd.DataFrame | None) -> pd.DataFrame:
    v = (meas.dropna(subset=['date'])
         .groupby(['patient_id', 'date'])
         .agg(patient_uuid=('patient_uuid', 'first'), img_types=('img_type', lambda s: sorted(set(s))))
         .reset_index())
    v['visit_no'] = v.groupby('patient_id')['date'].rank(method='dense').astype(int)
    if base is not None and {'patient', 'sex', 'birthdate'} <= set(base.columns):
        b = base[['patient', 'sex', 'birthdate']].drop_duplicates('patient')
        v = v.merge(b, left_on='patient_uuid', right_on='patient', how='left').drop(columns='patient')
        v['age'] = (v['date'] - pd.to_datetime(v['birthdate'], errors='coerce')).dt.days / 365.25
        v.loc[~v['age'].between(18, 110), 'age'] = np.nan      # Testeinträge/Tippfehler im Geburtsdatum
    return v


def recompute(progress=print) -> None:
    """Alle abgeleiteten Tabellen aus degen_raw.pkl / degen_base.pkl neu berechnen."""
    progress('Berechne Landmarken, Parameter und Monte-Carlo-Unsicherheit (~2 min) …')
    raw = pd.read_pickle(f'{DATA_DIR}/degen_raw.pkl')
    try:
        base = pd.read_pickle(f'{DATA_DIR}/degen_base.pkl')
    except FileNotFoundError:
        base = None
    t = build(raw)
    t['visits'] = build_visits(t['measure'], base)
    for name, obj in t.items():
        pd.to_pickle(obj, f'{DATA_DIR}/degen_{name}.pkl')
    write_uncertainty_report(t['uncert'], t['measure'])


def run(token: str, progress=print) -> None:
    progress('Lade /api/imagedata/ …')
    df4d = aa.build_df4d(fetch_data(token))
    progress('Erkenne Label-Shifts (k-Means) …')
    errs = aa.detect_label_errors(df4d)
    raw = aa.apply_label_corrections(df4d, errs)
    shifted = set(errs.loc[errs['error'] == 'shift', 'm_uuid']) if len(errs) else set()
    raw['label_shift_corrected'] = raw['m_uuid'].isin(shifted)
    raw[RAW_COLS].to_pickle(f'{DATA_DIR}/degen_raw.pkl')
    progress('Lade /api/clinicalbasedata/ …')
    try:
        fetch_clinical_base(token).to_pickle(f'{DATA_DIR}/degen_base.pkl')
    except requests.RequestException as e:
        progress(f'clinicalbasedata nicht geladen ({e}) – ohne Alter/Geschlecht weiter')
    progress('Lade /api/clinicalentrydata/ …')
    try:
        fetch_clinical_entries(token).to_pickle(f'{DATA_DIR}/degen_entries.pkl')
    except requests.RequestException as e:
        progress(f'clinicalentrydata nicht geladen ({e}) – Verlauf ohne klinische Filter')
    recompute(progress)


if __name__ == '__main__':
    recompute()
