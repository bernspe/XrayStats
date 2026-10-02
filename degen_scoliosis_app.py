"""
Degenerative Skoliose – Verlaufsmodell (Streamlit)

Plan: docs/PLAN_degen_skoliose_app.md · Entscheidungen: DECISIONS.md
Stand: Schritte 1a–1c, Phasen 2–5 – Abdeckung, Visite (Overlay), PI als Patientenkonstante,
Messunsicherheit, Normabweichung, Verlauf (Veränderungsvektoren), Kalibrierung & QC.

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run degen_scoliosis_app.py
"""
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

import annotation_analysis as aa
import degen_pipeline as dp
from spine_geom.frames import to_patient
from spine_geom.landmarks import ENDPLATES, extract
from spine_geom import SPINE_LABELS as SPINE_ORDER
from spine_geom import course as cs
from spine_geom.change import (change_test, cov_from_sd, deviation_test, mean_shift, nearest_psd, toward_target,
                               typology)
from spine_geom.norms import Z95, deviation_state, load_norms
from spine_geom.patient import Z_FLAG, between_image_tau2, patient_constant, pooled_sd
from spine_geom.sagittal import order_endplate

LABELS = {'PI': 'PI', 'SS': 'SS', 'PT': 'PT', 'LL': 'LL (L1–S1)', 'LDI': 'LDI (%)', 'L1PA': 'L1PA', 'T4PA': 'T4PA',
          'T1PA': 'T1PA', 'T4L1PA_mismatch': 'T4-L1PA-Mismatch', 'GT': 'Global Tilt', 'SVA_wkd': 'SVA (WKD)',
          'cCobb_max': 'centroid_cobb max (a.p.)', 'FHA': 'Hüftkopfwinkel (a.p.)', 'L4_tilt': 'L4-Tilt (a.p.)',
          'L5_tilt': 'L5-Tilt (a.p.)', 'CTPA': 'koronarer T1PA', 'C7_CSVL_wkd': 'C7-CSVL (WKD)'}
GLOBAL = ['T4PA', 'T1PA', 'T4L1PA_mismatch', 'GT', 'SVA_wkd']
RANK = {'ok': 2, 'tangente': 1, 'fehlt': 0}
NO_LAT = 'keine seitl. Aufnahme'
NO_AP = 'keine a.p.-Aufnahme'
# Hauptgrund: fehlender Bildausschnitt/Aufnahmetyp schlägt fehlende Annotation – dann hilft auch eine Hüfte nicht.
PRIORITY = [NO_LAT, NO_AP, 'Funktionsaufnahme', 'keine Stehaufnahme', 'zu wenige Nodes', 'C7 nicht annotiert', 'T1 nicht annotiert', 'T4 nicht annotiert',
            'L1 nicht annotiert', 'S1 nicht annotiert', 'keine Hüfte', 'Hüfte unplausibel', 'keine S1-Deckplatte',
            'keine L1-Deckplatte', 'keine L4-Deckplatte', 'keine L5-Deckplatte', 'keine gültige Kalibrierung',
            'nicht berechenbar']


def primary_reason(reason: str | None) -> str | None:
    return min(reason.split(', '), key=lambda r: PRIORITY.index(r) if r in PRIORITY else len(PRIORITY)) \
        if reason else None


CACHE = f'{dp.DATA_DIR}/degen_measure.pkl'
NORMS = load_norms()
KONVA_STAGE_HEIGHT = 600        # Nodes/Linien liegen im Konva-Stage-Raum (Bild auf Höhe 600 skaliert)
COLORS = px.colors.qualitative.Plotly
# Frontend-globalParams (xray-annotator ≥ 2.17.6): ohne Suffix Spline-Konstruktion, _cobb über Cobb-Linie
STORED = {'pelvic_incidence': 'PI', 'pelvic_tilt': 'PT', 'sacral_slope': 'SS', 't1_pelvic_angle': 'T1PA',
          'global_tilt': 'GT'}


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------

@st.cache_data
def load(mtime: float, params: tuple = tuple(dp.PARAMS)):
    """Argumente nur als Cache-Schlüssel: neuer Pipeline-Lauf oder geänderte Parameterliste → neu laden.
    Kein führender Unterstrich – den ignoriert st.cache_data beim Hashen (das war der KeyError 'LDI':
    eine laufende Instanz hielt Daten von vor der Neuberechnung)."""
    d = {}
    for n in ('measure', 'params', 'visits', 'calib', 'vbh', 'curves', 'shape', 'raw', 'uncert'):
        try:
            d[n] = pd.read_pickle(f'{dp.DATA_DIR}/degen_{n}.pkl')
        except FileNotFoundError:           # Datenstand vor Schritt 1c → nach Neuladen vorhanden
            d[n] = None
    d['cov'] = visit_coverage(d['measure'], d['params'], d['visits'])
    return d


def visit_coverage(meas: pd.DataFrame, params: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Patient × Visite × Parameter: bester Status über alle seitlichen Messungen der Visite,
    bei 'fehlt' der häufigste Grund."""
    p = params.merge(meas[['patient_id', 'date']], left_on='idx', right_index=True).dropna(subset=['date'])
    p['rank'] = p['status'].map(RANK)
    best = (p.sort_values('rank', ascending=False)
            .groupby(['patient_id', 'date', 'param'])
            .agg(status=('status', 'first'), n_meas=('idx', 'nunique'),
                 reason=('reason', lambda s: s.mode().iat[0] if s.notna().any() else None))
            .reset_index())
    grid = visits[['patient_id', 'date', 'visit_no']].merge(pd.DataFrame({'param': dp.PARAMS}), how='cross')
    cov = grid.merge(best, on=['patient_id', 'date', 'param'], how='left')
    cov['reason'] = cov['reason'].where(cov['status'].notna(),
                                        np.where(cov['param'].isin(dp.CORONAL), NO_AP, NO_LAT))
    cov['status'] = cov['status'].fillna('fehlt')
    cov.loc[cov['status'] != 'fehlt', 'reason'] = None
    cov['main_reason'] = cov['reason'].map(primary_reason)
    return cov


def _pct(n, total):
    return f'{n} / {total} ({100 * n / total:.0f} %)' if total else '–'


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def sidebar():
    st.sidebar.header('Daten')
    try:
        st.sidebar.caption(f'Stand: {datetime.fromtimestamp(os.path.getmtime(CACHE)):%d.%m.%Y %H:%M}')
    except OSError:
        st.sidebar.caption('Noch kein Datenstand vorhanden.')
    token = st.sidebar.text_input('Bearer Token', type='password',
                                  help='xray-annotator.renecol.org → Account info → Copy token')
    if st.sidebar.button('Von API laden', width='stretch'):
        if not token.strip():
            st.sidebar.error('Bitte Token eingeben.')
            return
        with st.sidebar.status('Lade und berechne …') as status:
            try:
                dp.run(token.strip(), progress=status.write)
            except Exception as e:  # Token abgelaufen, Netzwerk …
                status.update(label='Fehler', state='error')
                st.sidebar.error(f'{type(e).__name__}: {e}')
                return
            status.update(label='Fertig', state='complete')
        st.cache_data.clear()
        st.rerun()
    st.sidebar.divider()
    return st.sidebar.toggle('Tangenten-Fallback zählt als abgebildet', value=True,
                             help='S1-/L1-Deckplatte nicht gezeichnet → Spline-Richtung am Node (ungenauer, '
                                  'M liegt am Centroid statt an der Deckplattenmitte).')


def tab_coverage(meas, cov, visits, count_tangent):
    ok_status = {'ok', 'tangente'} if count_tangent else {'ok'}
    cov = cov.assign(measurable=cov['status'].isin(ok_status))
    n_pat = visits['patient_id'].nunique()
    per_pat = cov.groupby(['param', 'patient_id'])['measurable'].agg(['any', 'sum'])

    lat_pat = meas.loc[(meas['view'] == 'lat') & (meas['posture'] == 'standing'), 'patient_id'].nunique()
    glob_any = cov[cov['param'].isin(GLOBAL) & cov['measurable']]['patient_id'].nunique()
    pi_any = per_pat.loc['PI', 'any'].sum()
    short = lambda n: f'{n} ({100 * n / n_pat:.0f} %)'  # noqa: E731
    c = st.columns(4)
    c[0].metric('Patienten', n_pat, help=f'{len(visits)} Visiten (Patient × Aufnahmedatum)')
    c[1].metric('seitl. Stehaufnahme', short(lat_pat))
    c[2].metric('PI mind. 1×', short(pi_any))
    c[3].metric('Global mind. 1×', short(glob_any), help='T4PA, T1PA, T4-L1PA-Mismatch, Global Tilt oder SVA')

    st.subheader('Abdeckung je Parameter')
    base = cov[cov['visit_no'] == 1].groupby('param')['measurable'].sum()
    never = per_pat[~per_pat['any']].reset_index().merge(
        cov[~cov['measurable']].groupby(['param', 'patient_id'])['main_reason']
        .agg(lambda s: s.mode().iat[0]).reset_index(), on=['param', 'patient_id'])
    top_reason = never.groupby('param')['main_reason'].agg(lambda s: f'{s.mode().iat[0]} ({(s == s.mode().iat[0]).sum()}×)')
    summary = pd.DataFrame({
        'Parameter': [LABELS[p] for p in dp.PARAMS],
        'mind. 1×': [_pct(per_pat.loc[p, 'any'].sum(), n_pat) for p in dp.PARAMS],
        'an Baseline (1. Visite)': [_pct(base.get(p, 0), n_pat) for p in dp.PARAMS],
        '≥ 2 Visiten': [_pct((per_pat.loc[p, 'sum'] >= 2).sum(), n_pat) for p in dp.PARAMS],
        'Häufigster Grund, wenn nie abgebildet': [top_reason.get(p, '–') for p in dp.PARAMS],
    })
    st.dataframe(summary, hide_index=True)

    st.subheader('Warum fehlt ein Parameter? (Patienten, bei denen er nie abgebildet ist)')
    fig = px.bar(never.assign(Parameter=never['param'].map(LABELS)).groupby(['Parameter', 'main_reason'])
                 .size().reset_index(name='Patienten'),
                 y='Parameter', x='Patienten', color='main_reason', orientation='h',
                 category_orders={'Parameter': [LABELS[p] for p in dp.PARAMS], 'main_reason': PRIORITY},
                 labels={'main_reason': 'Hauptgrund'})
    fig.update_layout(height=380, margin=dict(t=10, b=10), legend=dict(orientation='h', y=-0.2))
    st.plotly_chart(fig)

    st.subheader('Patient × Parameter (Anzahl Visiten mit Parameter)')
    mat = per_pat['sum'].unstack('param')[dp.PARAMS].rename(columns=LABELS)
    mat.insert(0, 'Visiten', visits.groupby('patient_id').size())
    if st.toggle('Nur Patienten ohne globale Parameter'):
        mat = mat[mat[[LABELS[p] for p in GLOBAL]].sum(axis=1) == 0]
    st.caption(f'{len(mat)} Patienten')
    st.dataframe(mat.style.map(lambda v: f'background-color: {"#f4c7c3" if v == 0 else "#c9e7cf"}',
                               subset=[LABELS[p] for p in dp.PARAMS]), height=420)
    st.download_button('Visitenmatrix als CSV', cov.to_csv(index=False).encode(),
                       file_name='degen_abdeckung_visiten.csv', mime='text/csv')

    st.subheader('Detail je Patient')
    pid = st.selectbox('Patient', sorted(visits['patient_id'].unique()))
    v = visits[visits['patient_id'] == pid]
    if {'sex', 'age'} <= set(v.columns) and v['sex'].notna().any():
        st.caption(f"{v['sex'].iat[0]}, Alter bei 1. Visite {v['age'].iat[0]:.0f} J.")
    sym = cov[cov['patient_id'] == pid].assign(
        cell=lambda d: d['status'].map({'ok': '✓', 'tangente': '≈ Tangente'}).fillna(d['reason']))
    detail = sym.pivot(index='date', columns='param', values='cell')[dp.PARAMS].rename(columns=LABELS)
    detail.insert(0, 'Aufnahmen', v.set_index('date')['img_types'].map(', '.join))
    detail.index = detail.index.strftime('%Y-%m-%d')
    st.dataframe(detail)


def tab_calibration(meas, factors, vbh=None):
    st.subheader('Level-Faktoren (Größe relativ zu L4)')
    st.caption('Geschätzt aus Bildern, an denen Rater verschiedene Kalibrierwirbel gewählt haben. '
               'Level ohne Faktor → Kalibrierung ungültig → Fallback über Node-Abstand.')
    n_txt = lambda n: 'Referenz' if n is None else ('Default (zu wenig Paare)' if n == 0 else str(n))  # noqa: E731
    st.dataframe(pd.DataFrame([{'Ansicht': v, 'Level': lvl, 'Faktor': f['factor'], 'Bildpaare': n_txt(f['n'])}
                               for (v, lvl), f in sorted(factors.items())]),
                 hide_index=True, column_config={'Faktor': st.column_config.NumberColumn(format='%.3f')})

    if vbh:
        st.subheader('Wirbelkörperhöhe in WKD (Populationsfaktoren)')
        st.caption('Maßstab primär aus Wirbelkörperhöhen der Cobb-segmental-Linien, je Patient mit eigener '
                   'Anatomie geschätzt (DECISIONS.md 2026-09-30); Kalibrierlinie nur als zusätzliche Beobachtung.')
        st.dataframe(pd.DataFrame([{'Ansicht': v, 'Level': lvl, 'Höhe/WKD': f['factor'], 'Messungen': f['n']}
                                   for (v, lvl), f in sorted(vbh.items())]),
                     hide_index=True, column_config={'Höhe/WKD': st.column_config.NumberColumn(format='%.3f')})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Kalibrierstatus')
        st.dataframe(pd.crosstab(meas['calib_status'], meas['view'], margins=True, margins_name='Σ'))
        if 'calib_source' in meas:
            st.subheader('Maßstabsquelle')
            st.dataframe(pd.crosstab(meas['calib_source'].fillna('–'), meas['view'], margins=True, margins_name='Σ'))
    with c2:
        st.subheader('QC-Flags (Messungen)')
        st.dataframe(meas['qc_flags'].explode().value_counts().rename('Messungen'))

    st.subheader('Kalibrierung außerhalb der Messanweisung (L3/L4) je Rater')
    off = meas[meas['calib_status'].isin(['off_protocol', 'ungültig'])]
    st.dataframe(pd.crosstab(off['rater_name'], off['calib_level']))


# ---------------------------------------------------------------------------
# Visite: Bild mit Overlay
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_image(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None


def _seg(fig, pts, color, name, dash='solid', width=2, show=False):
    pts = np.asarray(pts)
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='lines', name=name, legendgroup=name,
                             showlegend=show, line=dict(color=color, dash=dash, width=width), hoverinfo='name'))


def overlay_figure(img_bytes, items, curves_df, height: int = 750):
    """items: Liste (Rater, xmd, Messungszeile). Koordinaten im Konva-Stage-Raum (Höhe 600)."""
    fig = go.Figure()
    width = None                                           # ohne Bild: x-Achse nach den Punkten
    if img_bytes:
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img.thumbnail((1200, 1200))                       # Payload klein halten, Seitenverhältnis bleibt
        width = img.width * KONVA_STAGE_HEIGHT / img.height
        fig.add_layout_image(source=img, x=0, y=0, sizex=width, sizey=KONVA_STAGE_HEIGHT, xref='x', yref='y',
                             sizing='stretch', layer='below')
    xs, ys = [], []
    for k, (rater, xmd, m) in enumerate(items):
        col = COLORS[k % len(COLORS)]
        lm = extract(xmd, lateral=m['view'] == 'lat')
        labs = list(lm.centroids)
        c = np.array([lm.centroids[x] for x in labs])
        xs += list(c[:, 0]) + ([lm.hip[0]] if lm.hip is not None else [])
        ys += list(c[:, 1]) + ([lm.hip[1]] if lm.hip is not None else [])
        fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode='markers+text' if k == 0 else 'markers',
                                 text=labs, textposition='middle left', textfont=dict(color=col, size=10),
                                 marker=dict(color=col, size=7), name=rater, legendgroup=rater,
                                 hovertext=labs, hoverinfo='text'))
        cal = (xmd.get('stateObject') or {}).get('calibrationLine') or {}
        if len(cal) >= 4:
            _seg(fig, [[cal['x1'], cal['y1']], [cal['x2'], cal['y2']]], 'lightgray', 'Kalibrierlinie', 'dot')
        if m['view'] == 'lat':
            hl = (xmd.get('stateObject') or {}).get('hipLine') or {}
            if len(hl) >= 4:
                heads = np.array([[hl['x1'], hl['y1']], [hl['x2'], hl['y2']]])
                fig.add_trace(go.Scatter(x=heads[:, 0], y=heads[:, 1], mode='markers', showlegend=False,
                                         marker=dict(symbol='circle-open', size=12, color=col), hoverinfo='skip'))
            for name, ep_attr in ENDPLATES:
                ep, src = getattr(lm, f'{ep_attr}_endplate'), getattr(lm, f'{ep_attr}_source')
                if ep is not None:
                    _seg(fig, ep, col, f'{name}-Deckplatte ({src})', 'dash' if src == 'tangent' else 'solid', 3)
            if lm.hip is not None and lm.s1_endplate is not None and 'hip_implausible' not in m['qc_flags']:
                H, M = lm.hip, lm.s1_endplate.mean(axis=0)
                fig.add_trace(go.Scatter(x=[H[0]], y=[H[1]], mode='markers', showlegend=False, hoverinfo='text',
                                         hovertext='Hüftachse H', marker=dict(symbol='star', size=12, color=col)))
                _seg(fig, [H, M], col, 'H → S1', width=1)
                for lab in ('L1', 'T4', 'T1'):
                    if lab in lm.centroids:
                        _seg(fig, [H, lm.centroids[lab]], col, f'H → {lab} (VPA)', 'dot', 1)
        elif curves_df is not None:
            for _, cv in curves_df[(curves_df['idx'] == m.name) & (curves_df['method'] == 'a')].iterrows():
                ends = np.array([lm.centroids[cv['upper']], lm.centroids[cv['lower']]])
                fig.add_trace(go.Scatter(x=ends[:, 0], y=ends[:, 1], mode='markers', showlegend=False,
                                         marker=dict(symbol='square-open', size=14, color=col), hoverinfo='text',
                                         hovertext=f"{cv['upper']}–{cv['lower']}: {abs(cv['angle']):.1f}°"))
                if cv['apex']:
                    a = lm.centroids[cv['apex']]
                    fig.add_trace(go.Scatter(x=[a[0]], y=[a[1]], mode='markers', showlegend=False,
                                             marker=dict(symbol='diamond', size=11, color=col), hoverinfo='text',
                                             hovertext=f"Apex {cv['apex']}, konvex {cv['convexity']}"))
    # Ausschnitt um die annotierten Punkte (Aufnahmen haben oft breite schwarze Ränder)
    pad = 80
    x_rng = [max(0, min(xs) - pad), min(width, max(xs) + pad)] if xs and width else ([0, width] if width else None)
    y_rng = [min(KONVA_STAGE_HEIGHT, max(ys) + pad), max(0, min(ys) - pad)] if ys else [KONVA_STAGE_HEIGHT, 0]
    fig.update_xaxes(range=x_rng, visible=False, constrain='domain')
    fig.update_yaxes(range=y_rng, visible=False, scaleanchor='x', constrain='domain')
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation='h', y=-0.02),
                      plot_bgcolor='black')
    return fig


def _image_panel(d, pid, date, view: str, key: str) -> None:
    """Röntgenbild einer Visite mit Overlay aller Rater; Stehaufnahmen zuerst (die gehen in den Verlauf ein)."""
    meas = d['measure']
    mv = meas[(meas['patient_id'] == pid) & (meas['date'] == date) & (meas['view'] == view)]
    if mv.empty:
        st.info(f"Keine {'seitliche' if view == 'lat' else 'a.p.'}-Aufnahme an dieser Visite.")
        return
    imgs = (mv.assign(_o=mv['posture'].ne('standing')).sort_values('_o')
            .drop_duplicates('image_id').set_index('image_id')['img_type'])
    image_id = st.selectbox('Aufnahme', imgs.index, key=key, format_func=lambda i: f'{imgs[i]} (#{i})') \
        if len(imgs) > 1 else imgs.index[0]
    if len(imgs) == 1:
        st.caption(f'{imgs.iat[0]} (#{image_id})')
    mi = mv[mv['image_id'] == image_id]
    items = [(m['rater_name'], d['raw'].loc[i, 'xray_measurement_data'], m) for i, m in mi.iterrows()]
    img = fetch_image(mi['img'].iat[0]) if mi['img'].notna().any() else None
    if img is None:
        st.caption('Bild nicht ladbar (signierte URL > 24 h alt?) – Overlay ohne Bild.')
    st.plotly_chart(overlay_figure(img, items, d['curves'], height=620), key=f'{key}_fig')


def tab_visit(d):
    meas, params, raw = d['measure'], d['params'], d['raw']
    if raw is None:
        st.info('Für das Overlay fehlt data/degen_raw.pkl – bitte einmal „Von API laden“.')
        return
    c1, c2, c3 = st.columns(3)
    pid = c1.selectbox('Patient', sorted(meas['patient_id'].dropna().unique()), key='visit_pid')
    mp = meas[meas['patient_id'] == pid]
    dates = sorted(mp['date'].dropna().unique())
    if not dates:
        st.warning('Keine datierten Aufnahmen.')
        return
    date = c2.selectbox('Visite', dates, format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'),
                        index=len(dates) - 1)
    mv = mp[mp['date'] == date]
    imgs = mv.drop_duplicates('image_id').set_index('image_id')['img_type']
    image_id = c3.selectbox('Aufnahme', imgs.index, format_func=lambda i: f'{imgs[i]} (#{i})')
    mi = mv[mv['image_id'] == image_id]
    raters = st.multiselect('Rater', list(mi['rater_name']), default=list(mi['rater_name']))
    mi = mi[mi['rater_name'].isin(raters)]
    items = [(m['rater_name'], raw.loc[i, 'xray_measurement_data'], m) for i, m in mi.iterrows()]

    left, right = st.columns([3, 2])
    with left:
        url = mi['img'].iat[0] if len(mi) else None
        img = fetch_image(url) if url else None
        if url and img is None:
            st.caption('Bild nicht ladbar (signierte URL > 24 h alt?) – Overlay ohne Bild.')
        st.plotly_chart(overlay_figure(img, items, d['curves']))
    with right:
        if mi['view'].eq('lat').any():
            st.markdown('**Sagittale Parameter** (eigene Berechnung)')
            p = params[params['idx'].isin(mi.index)].merge(mi[['rater_name']], left_on='idx', right_index=True)
            sd = p['sd'] if 'sd' in p else pd.Series(np.nan, index=p.index)
            cell = p['value'].map('{:.1f}'.format) + sd.map(lambda x: '' if pd.isna(x) else f' ± {x:.1f}')
            cell = cell.where(p['status'] == 'ok', cell + ' ≈').where(p['status'] != 'fehlt', '–')
            tab = p.assign(cell=cell).pivot(index='param', columns='rater_name', values='cell')
            st.dataframe(tab.reindex([x for x in dp.PARAMS if x in tab.index]).rename(index=LABELS))
            st.caption('± = Monte-Carlo-SD (Platzierungsrauschen), ≈ = Ersatzkonstruktion (Deckplatte aus Tangente '
                       'bzw. S1-Centroid statt S1-Deckplatte)')
            st.markdown('**Frontend-globalParams** (gespeichert)')
            gp = pd.DataFrame({m['rater_name']: m['stored_gp'] for _, m in mi.iterrows()})
            st.dataframe(gp.round(1) if len(gp) else gp)
        elif d['curves'] is not None:
            st.markdown('**centroid_cobb-Kurven** (a = zentrale Differenzen, b = Glättungsspline)')
            cv = d['curves'][d['curves']['idx'].isin(mi.index)].merge(mi[['rater_name']], left_on='idx',
                                                                      right_index=True)
            st.dataframe(cv.assign(angle=cv['angle'].abs().round(1))[
                ['rater_name', 'method', 'upper', 'lower', 'angle', 'apex', 'convexity']], hide_index=True)
        st.markdown('**QC**')
        st.dataframe(pd.DataFrame({'Rater': mi['rater_name'], 'Kalibrierung': mi['calib_status'],
                                   'Level': mi['calib_level'],
                                   'Flags': mi['qc_flags'].map(lambda f: ', '.join(sorted(f)))}), hide_index=True)


# ---------------------------------------------------------------------------
# Phase 3: PI als Patientenkonstante
# ---------------------------------------------------------------------------

@st.cache_data
def pi_analysis(mtime: float, include_tangent: bool):
    d = load(mtime)
    meas, params = d['measure'], d['params']
    ok = {'ok', 'tangente'} if include_tangent else {'ok'}
    p = params[(params['param'] == 'PI') & params['status'].isin(ok)].merge(
        meas[['patient_id', 'image_id', 'date', 'img_type', 'rater_name']], left_on='idx', right_index=True)
    sigma = pooled_sd(g.to_numpy() for _, g in p.groupby('image_id')['value'])
    img = (p.groupby(['patient_id', 'image_id'])
           .agg(date=('date', 'first'), img_type=('img_type', 'first'), pi=('value', 'mean'), n=('value', 'size'),
                rater_sd=('value', 'std'))
           .reset_index())
    groups = [g for _, g in img.groupby('patient_id')]
    tau2 = between_image_tau2([g['pi'] for g in groups], [g['n'] for g in groups], sigma)
    pats, parts = [], []
    for g in groups:
        est, se, z, flagged = patient_constant(g['pi'], g['n'], sigma, tau2)
        parts.append(g.assign(z=z, excluded=flagged, sd_img=np.sqrt(sigma ** 2 / g['n'] + tau2)))
        pats.append(dict(patient_id=g['patient_id'].iat[0], PI=est, SE=se, images=len(g), ratings=int(g['n'].sum()),
                         sd_between=g['pi'].std() if len(g) > 1 else np.nan,
                         conspicuous=int((np.abs(z) > Z_FLAG).sum()), excluded=int(flagged.sum())))
    return p, pd.concat(parts), pd.DataFrame(pats), sigma, tau2


def _bland_altman(df, a, b, title):
    x = df[[a, b]].dropna()
    if len(x) < 3:
        st.caption(f'{title}: zu wenige Paare.')
        return
    diff, mean = x[a] - x[b], (x[a] + x[b]) / 2
    bias, sd = diff.mean(), diff.std()
    fig = px.scatter(x=mean, y=diff, labels={'x': 'Mittelwert (°)', 'y': f'{a} − {b} (°)'}, opacity=0.6)
    for yv, dash in ((bias, 'solid'), (bias + 1.96 * sd, 'dash'), (bias - 1.96 * sd, 'dash')):
        fig.add_hline(y=yv, line_dash=dash, line_color='firebrick')
    fig.update_layout(height=320, margin=dict(t=30, b=10), title=f'{title}: Bias {bias:+.1f}°, LoA ±{1.96 * sd:.1f}° (n={len(x)})')
    st.plotly_chart(fig)


def tab_pi(d, mtime, count_tangent):
    p, img, pat, sigma, tau2 = pi_analysis(mtime, count_tangent)
    if pat.empty:
        st.info('Keine PI-Messungen.')
        return
    st.caption('PI ist morphologisch konstant → einmal pro Patient geschätzt (präzisionsgewichtet über Bilder). '
               f'Bild auffällig bei Leave-one-out-|z| > {Z_FLAG:.0f}, schlimmstes auffälliges Bild wird sequentiell '
               'ausgeschlossen (≥ 2 Bilder bleiben). Tangenten-Fallback folgt dem Schalter in der Sidebar.')
    c = st.columns(5)
    c[0].metric('Patienten mit PI', len(pat))
    c[1].metric('σ Rater (gleiches Bild)', f'{sigma:.1f}°', help='gepoolte SD zwischen Ratern am selben Bild')
    c[2].metric('τ zwischen Bildern', f'{np.sqrt(tau2):.1f}°',
                help='Zusatzstreuung zwischen Aufnahmen desselben Patienten über den Messfehler hinaus')
    c[3].metric('Median SE Patient', f'{pat["SE"].median():.1f}°')
    c[4].metric('Bilder auffällig / ausgeschl.', f'{int((np.abs(img["z"]) > Z_FLAG).sum())} / {int(img["excluded"].sum())}')

    left, right = st.columns(2)
    with left:
        st.markdown('**Streuung innerhalb Patient vs. Messfehler** (Patienten mit ≥ 2 Bildern)')
        multi = pat.dropna(subset=['sd_between'])
        fig = px.histogram(multi, x='sd_between', nbins=30, labels={'sd_between': 'SD der Bild-PI je Patient (°)'})
        fig.add_vline(x=sigma, line_dash='dash', annotation_text='σ Rater')
        fig.add_vline(x=np.sqrt(sigma ** 2 / img['n'].mean() + tau2), line_color='firebrick',
                      annotation_text='erwartet (σ²/n̄+τ²)')
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig)
    with right:
        st.markdown('**Patienten**')
        st.dataframe(pat.sort_values('conspicuous', ascending=False).round(1), hide_index=True, height=320,
                     column_config={'conspicuous': 'auffällig', 'excluded': 'ausgeschl.', 'sd_between': 'SD Bilder'})

    st.markdown('**Verlauf je Patient**')
    pid = st.selectbox('Patient', pat.sort_values('conspicuous', ascending=False)['patient_id'], key='pi_pid')
    g, row = img[img['patient_id'] == pid], pat.set_index('patient_id').loc[pid]
    fig = go.Figure()
    fig.add_hrect(y0=row['PI'] - 1.96 * row['SE'], y1=row['PI'] + 1.96 * row['SE'], fillcolor='lightgray',
                  opacity=0.4, line_width=0)
    fig.add_hline(y=row['PI'], annotation_text=f"PI {row['PI']:.1f}° ± {1.96 * row['SE']:.1f}")
    fig.add_trace(go.Scatter(x=g['date'], y=g['pi'], mode='markers', name='Bild (Rater-Mittel)',
                             error_y=dict(array=1.96 * g['sd_img']),
                             marker=dict(size=11, color=np.where(g['excluded'], 'firebrick', 'steelblue')),
                             hovertext=g['img_type'] + ', z=' + g['z'].round(1).astype(str)))
    pp = p[p['patient_id'] == pid]
    fig.add_trace(go.Scatter(x=pp['date'], y=pp['value'], mode='markers', name='einzelne Rater',
                             marker=dict(size=6, symbol='x', color='gray'), hovertext=pp['rater_name']))
    fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title='PI (°)')
    st.plotly_chart(fig)

    st.subheader('Validierung gegen das bisherige Frontend-PI')
    meas = d['measure']
    own = p.set_index('idx')[['value']].rename(columns={'value': 'PI_eigen'})
    gp = meas.loc[own.index, 'stored_gp'] if 'stored_gp' in meas else pd.Series(dtype=object)
    if gp.empty:
        st.info('Gespeicherte globalParams fehlen im Datenstand – bitte neu laden.')
        return
    own['PI_frontend_spline'] = gp.map(lambda g: g.get('pelvic_incidence'))
    own['PI_frontend_cobb'] = gp.map(lambda g: g.get('pelvic_incidence_cobb'))
    c1, c2 = st.columns(2)
    with c1:
        _bland_altman(own.astype(float), 'PI_eigen', 'PI_frontend_spline', 'vs. Spline-Surrogat')
    with c2:
        _bland_altman(own.astype(float), 'PI_eigen', 'PI_frontend_cobb', 'vs. Frontend-Cobb')
    shape = d['shape']
    if shape is not None:
        h = shape[shape['label'] == 'L5'].set_index('idx')['seg_h']
        x = own.join(h).dropna(subset=['PI_frontend_spline', 'seg_h']).astype(float)
        x['Δ Spline − eigen'] = x['PI_frontend_spline'] - x['PI_eigen']
        r = x[['seg_h', 'Δ Spline − eigen']].corr().iat[0, 1] if len(x) > 2 else np.nan
        st.markdown(f'**Abhängigkeit von der L5/S1-Höhe** (Centroid-Abstand L5–S1 in WKD; r = {r:.2f})')
        fig = px.scatter(x, x='seg_h', y='Δ Spline − eigen', trendline='ols' if len(x) > 2 else None,
                         labels={'seg_h': 'L5–S1 (WKD)'}, opacity=0.6)
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig)
    both = own.join(meas[['patient_id', 'image_id']]).groupby(['patient_id', 'image_id'])[
        ['PI_eigen', 'PI_frontend_spline']].mean()
    sds = both.groupby('patient_id').agg(['std', 'count'])
    sds = sds[sds[('PI_eigen', 'count')] > 1]
    st.markdown(f"**Zeitliche Instabilität** (SD der Bild-PI je Patient, n = {len(sds)} Patienten mit ≥ 2 Bildern): "
                f"eigen Median {sds[('PI_eigen', 'std')].median():.1f}°, "
                f"Spline-Surrogat Median {sds[('PI_frontend_spline', 'std')].median():.1f}°")


# ---------------------------------------------------------------------------
# Phase 2: Messunsicherheit
# ---------------------------------------------------------------------------

def tab_uncertainty(d):
    u = d['uncert']
    if u is None:
        st.info('Noch keine Unsicherheitsauswertung – bitte „Von API laden“ oder `python degen_pipeline.py`.')
        return
    st.caption('Empirisch = SD zwischen Ratern am selben Bild (inkl. grober Fehler). Monte-Carlo = robuste '
               'Landmarken-Kovarianz fortgepflanzt (reines Platzierungsrauschen). MDC95 = 1,96·√2·SEM. '
               'Haltungsvariabilität zwischen Visiten ist nicht enthalten → MDC95 ist für Visitenvergleiche eine '
               'Untergrenze. Bericht: reports/messunsicherheit.md')
    sem = u['sem'].assign(param=lambda x: x['param'].map(LABELS))
    st.subheader('SEM und MDC95 je Parameter')
    st.dataframe(sem, hide_index=True, column_config={
        'param': 'Parameter', 'n_bilder_multirater': 'Bilder ≥ 2 Rater',
        **{c: st.column_config.NumberColumn(c.replace('_', ' '), format='%.2f')
           for c in ('sem_emp', 'sem_emp_inkl_tangente', 'sem_robust', 'sem_mc', 'mdc95_emp', 'mdc95_robust',
                     'mdc95_mc')}})
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Landmarken-Streuung')
        st.caption('WKD bzw. Grad; grobe Fehler = Label-/Deckplattenverwechslung')
        st.dataframe(u['landmark_sd'].sort_values(['view', 'kind']), hide_index=True, height=420,
                     column_config={'sd_x': st.column_config.NumberColumn(format='%.3f'),
                                    'sd_y': st.column_config.NumberColumn(format='%.3f'),
                                    'grobe_fehler': st.column_config.NumberColumn(format='percent')})
    with c2:
        st.subheader('Rater-SEM bei festen Endwirbeln')
        st.dataframe(u['cobb_sem'].round(2), hide_index=True)
        st.caption('manual = manueller Cobb-regional, cc_a/cc_b = centroid_cobb (zentrale Differenzen / '
                   'Glättungsspline) mit denselben Endwirbeln')
    st.subheader('centroid_cobb vs. manueller Cobb (a.p.)')
    pairs = u['cobb_pairs'].astype({'manual': float})
    stratum = st.segmented_control('Stratum', ['alle', 'untere Endwirbel ≠ S1', 'untere Endwirbel = S1'],
                                   default='alle')
    if stratum == 'untere Endwirbel ≠ S1':
        pairs = pairs[pairs['lower'] != 'S1']
    elif stratum == 'untere Endwirbel = S1':
        pairs = pairs[pairs['lower'] == 'S1']
    a, b = st.columns(2)
    with a:
        _bland_altman(pairs.rename(columns={'cc_a': 'centroid_a', 'manual': 'Cobb'}), 'centroid_a', 'Cobb',
                      'Methode a')
    with b:
        _bland_altman(pairs.rename(columns={'cc_b': 'centroid_b', 'manual': 'Cobb'}), 'centroid_b', 'Cobb',
                      'Methode b')


# ---------------------------------------------------------------------------
# Phase 4: Normabweichung
# ---------------------------------------------------------------------------

@st.cache_data
def visit_values(mtime: float, include_tangent: bool) -> pd.DataFrame:
    """Patient × Visite × Parameter: Mittel über alle Messungen (Rater, Aufnahmen) der Visite,
    SD = √(mittlere MC-Varianz / n), ohne MC-SD die robuste Rater-SEM. ponytail: Messungen als unabhängig behandelt; Rater am
    selben Bild teilen Bildfehler → SD eher zu klein. Phase 5 ersetzt das durch Σ_mess + Aufnahmeanteil."""
    d = load(mtime)
    ok = {'ok', 'tangente'} if include_tangent else {'ok'}
    p = d['params'][d['params']['status'].isin(ok)].merge(d['measure'][['patient_id', 'date']], left_on='idx',
                                                          right_index=True).dropna(subset=['date'])
    # ohne MC-SD (Hüftkopfwinkel, L4-/L5-Tilt: Linien werden nicht verrauscht) → robuste Rater-SEM des Parameters
    if d['uncert'] is not None:
        sem = d['uncert']['sem'].set_index('param')['sem_robust']
        p['sd'] = p['sd'].fillna(p['param'].map(sem))
    g = p.groupby(['patient_id', 'date', 'param'])
    return pd.DataFrame({'value': g['value'].mean(), 'n': g['value'].size(),
                         'sd': np.sqrt((g['sd'].apply(lambda s: (s ** 2).mean())) / g['value'].size())}).reset_index()


def _state(vv, pi_row, visit_row, pid, date):
    x = vv[(vv['patient_id'] == pid) & (vv['date'] == date)]
    values = {r['param']: (r['value'], r['sd'] if pd.notna(r['sd']) else None) for _, r in x.iterrows()
              if r['param'] != 'PI'}
    pi = (pi_row['PI'], pi_row['SE']) if pi_row is not None else None
    age = visit_row['age'] if 'age' in visit_row and pd.notna(visit_row.get('age')) else None
    sex = visit_row.get('sex') if pd.notna(visit_row.get('sex')) else None
    return pd.DataFrame(deviation_state(values, pi, age, sex, NORMS))


def tab_norm(d, mtime, count_tangent):
    vv = visit_values(mtime, count_tangent)
    _, _, pat, _, _ = pi_analysis(mtime, count_tangent)
    pat = pat.set_index('patient_id')
    visits = d['visits'].set_index(['patient_id', 'date'])
    missing_strata = [e['source'] for p_ in NORMS['population'].values() for e in p_
                      if 'strata' in e and not e['strata']]
    st.caption('Standard-Referenz MEANS (multiethnisch); passende Alters-/Geschlechtsstrata haben Vorrang. '
               'PI-abhängige Ziele verwenden die Patienten-PI aus Phase 3 (Schätzer ± SE). Werte: Mittel aller '
               'Messungen der Visite ± SD. Quellen und Werte: spine_geom/norms.yaml.')
    if missing_strata:
        st.warning(f"Noch keine Alters-/Geschlechtsstrata für {', '.join(sorted(set(missing_strata)))} hinterlegt "
                   '(Tabelle im Volltext, Paywall) → vorerst Gesamtnorm MEANS.')

    c1, c2 = st.columns(2)
    pids = sorted(vv['patient_id'].unique(), key=lambda p: (p not in pat.index, p))
    pid = c1.selectbox('Patient', pids, key='norm_pid')
    dates = sorted(vv.loc[vv['patient_id'] == pid, 'date'].unique())
    date = c2.selectbox('Visite', dates, index=len(dates) - 1, format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'),
                        key='norm_date')
    vrow = visits.loc[(pid, date)] if (pid, date) in visits.index else pd.Series(dtype=object)
    prow = pat.loc[pid] if pid in pat.index else None
    m = st.columns(3)
    m[0].metric('PI (Patient)', f"{prow['PI']:.1f}° ± {Z95 * prow['SE']:.1f}" if prow is not None else '–')
    m[1].metric('Alter', f"{vrow['age']:.0f} J." if pd.notna(vrow.get('age')) else '–')
    m[2].metric('Geschlecht', vrow.get('sex') if pd.notna(vrow.get('sex')) else '–')
    rows = _state(vv, prow, vrow, pid, date)
    if rows.empty:
        st.info('Keine normierbaren Parameter an dieser Visite.')
    else:
        rows['param'] = rows['param'].map(lambda x: LABELS.get(x, {'PI_LL': 'PI − LL'}.get(x, x)))
        num = {c: st.column_config.NumberColumn(format='%.1f') for c in
               ('wert', 'wert_sd', 'referenz', 'referenz_sd', 'abweichung', 'abweichung_sd')}
        for art in ('Population', 'PI-Ziel', 'Zielbereich', 'Klassifikation'):
            part = rows[rows['art'] == art].drop(columns='art').dropna(axis=1, how='all')
            part = part.apply(lambda c: pd.to_numeric(c) if c.map(lambda x: x is None or isinstance(x, (int, float))).all()
                              else c).infer_objects()
            if len(part):
                st.markdown(f'**{art}**')
                cfg = {**num, 'z': st.column_config.NumberColumn(format='%.2f')}
                st.dataframe(part, hide_index=True, column_config={k: v for k, v in cfg.items()
                                                                   if k in part and pd.api.types.is_numeric_dtype(part[k])})

    st.subheader('Kohorte: letzte Visite mit LL je Patient')
    last = vv[vv['param'] == 'LL'].sort_values('date').groupby('patient_id').tail(1)
    coh = last.merge(pat[['PI', 'SE']], left_on='patient_id', right_index=True)
    if coh.empty:
        return
    coh = coh.assign(PI_LL=coh['PI'] - coh['value'])
    fig = px.scatter(coh, x='PI', y='value', hover_name='patient_id', error_x=Z95 * coh['SE'],
                     labels={'value': 'LL L1–S1 (°)', 'PI': 'PI Patient (°)'}, opacity=0.7)
    gap = NORMS['pi_targets']['LL'][0]
    xs = np.array([coh['PI'].min(), coh['PI'].max()])
    fig.add_trace(go.Scatter(x=xs, y=gap['a'] * xs + gap['b'], mode='lines', name='GAP ideale LL'))
    for thr, dash in zip(NORMS['classifications']['schwab_PI_LL']['thresholds'], ('dash', 'dot')):
        fig.add_trace(go.Scatter(x=xs, y=xs - thr, mode='lines', line=dict(dash=dash, color='firebrick'),
                                 name=f'PI − LL = {thr:.0f}° (Schwab)'))
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    cls = NORMS['classifications']['schwab_PI_LL']
    counts = coh['PI_LL'].map(lambda v: cls['labels'][sum(v >= t for t in cls['thresholds'])]).value_counts()
    st.caption('SRS-Schwab PI−LL (letzte Visite): ' + ', '.join(f'{k}: {counts.get(k, 0)}' for k in cls['labels']))

    with st.expander('Normquellen'):
        st.dataframe(pd.DataFrame([{'Quelle': k, **{f: v.get(f) for f in ('citation', 'population', 'n', 'verified')}}
                                   for k, v in NORMS['sources'].items()]), hide_index=True)


# ---------------------------------------------------------------------------
# Phase 5: Verlauf / Veränderungsvektoren
# ---------------------------------------------------------------------------

WKD_PARAMS = {'SVA_wkd', 'C7_CSVL_wkd'}
PCT_PARAMS = {'LDI'}
CORE = ['SS', 'PT', 'LL', 'L1PA']       # Kohortenvergleich: für alle Patienten derselbe Parametersatz


SHAPE_SFX = {'lat': ('.x', '.y'), 'ap': ('.z', '.c'), '3d': ('.x', '.y', '.z')}
SHAPE_MODE = {'lat': '2D seitlich', 'ap': '2D a.p.', '3d': '3D (näherungsweise)'}


@st.cache_data
def visit_shapes(mtime: float) -> pd.DataFrame:
    """Patient × Visite × Formkoordinate: seitlich 'L3.x'/'L3.y' (Sakrumsystem, WKD Tiefe), a.p. 'L3.z'/'L3.c'
    (koronares Beckensystem, WKD Breite); Mittel über alle Stehaufnahme-Messungen der Visite."""
    d = load(mtime)
    sh = d['shape'].merge(d['measure'][['patient_id', 'date', 'posture']], left_on='idx', right_index=True)
    sh = sh[sh['posture'] == 'standing'].dropna(subset=['date'])
    if 'view' not in sh:                                # Datenstand vor 3D-Erweiterung
        sh = sh.assign(view='lat')
    sfx = {('lat', 'xs'): '.x', ('lat', 'ys'): '.y', ('ap', 'xs'): '.z', ('ap', 'ys'): '.c'}
    long = pd.concat([g.assign(param=g['label'] + sfx[(view, col)], value=g[col])
                      for view, g in sh.groupby('view') for col in ('xs', 'ys')])
    g = long.groupby(['patient_id', 'date', 'param'])['value']
    return pd.DataFrame({'value': g.mean(), 'n': g.size()}).reset_index()


def _station(vs_p: pd.DataFrame, date, mode: str) -> pd.DataFrame:
    """Wirbel × Achsen einer Visite (nur Wirbel mit allen Achsen des Modus)."""
    v = vs_p[vs_p['date'] == date].set_index('param')['value']
    labs = [l_ for l_ in SPINE_ORDER if all(f'{l_}{s}' in v.index for s in SHAPE_SFX[mode])]
    return pd.DataFrame({s: [v[f'{l_}{s}'] for l_ in labs] for s in SHAPE_SFX[mode]}, index=labs)


def _ap_scale(vs_p: pd.DataFrame, date) -> float:
    """Nur für die 3D-Darstellung: a.p.-WKD (Breite) → seitliche WKD (Tiefe) über das Verhältnis der
    Segmentlängen gemeinsamer Nachbarwirbel. ponytail: Segmente ≈ kraniokaudal, in beiden Ansichten gleich lang
    angenommen; für den Test irrelevant (jede Komponente trägt ihre eigene Varianz)."""
    lat, ap = _station(vs_p, date, 'lat'), _station(vs_p, date, 'ap')
    ratios = []
    for a, b in zip(SPINE_ORDER, SPINE_ORDER[1:]):
        if a in lat.index and b in lat.index and a in ap.index and b in ap.index:
            dl = np.hypot(*(lat.loc[a] - lat.loc[b]).to_numpy())
            da = np.hypot(*(ap.loc[a] - ap.loc[b]).to_numpy())
            if da > 0:
                ratios.append(dl / da)
    return float(np.median(ratios)) if ratios else 1.0


def height_invariant(vs_p: pd.DataFrame, stations) -> pd.DataFrame:
    """Jede Station je Ansicht auf die mittlere Wirbelsäulenlänge aller Stationen skalieren: Länge = Polygonzug
    durch die in *allen* Stationen vorhandenen Wirbel. Entfernt Höhenänderungen (OP-Distraktion, Kollaps) und
    Maßstabsfehler, Einheit bleibt ≈ WKD. Ansichten mit < 2 gemeinsamen Wirbeln bleiben unverändert."""
    out = vs_p.copy()
    for sfx in (SHAPE_SFX['lat'], SHAPE_SFX['ap']):
        st_ = {t: _station(vs_p, t, 'lat' if sfx == SHAPE_SFX['lat'] else 'ap') for t in stations}
        common = [l_ for l_ in SPINE_ORDER if all(l_ in s_.index for s_ in st_.values())]
        if len(common) < 2:
            continue
        length = {t: np.linalg.norm(np.diff(s_.loc[common].to_numpy(), axis=0), axis=1).sum() for t, s_ in st_.items()}
        ref = np.mean(list(length.values()))
        for t, L in length.items():
            sel = (out['date'] == t) & out['param'].str.endswith(sfx)
            out.loc[sel, 'value'] *= ref / L
    return out


def shape_step(vs_p, ta, tb, mode, cm, deg, wkd) -> dict | None:
    """Test eines Stationsschritts über die in beiden Visiten vorhandenen Wirbel."""
    a, b = _station(vs_p, ta, mode), _station(vs_p, tb, mode)
    labs = [l_ for l_ in a.index if l_ in b.index]
    if len(labs) < 2:
        return None
    coords = [f'{l_}{s}' for l_ in labs for s in SHAPE_SFX[mode]]
    va = vs_p[vs_p['date'] == ta].set_index('param')
    vb = vs_p[vs_p['date'] == tb].set_index('param')
    sem = cm['sem'].reindex(coords).fillna(cm['sem'].median()).to_numpy()
    s = sigma_delta(coords, sem / np.sqrt(va.loc[coords, 'n'].to_numpy()), sem / np.sqrt(vb.loc[coords, 'n'].to_numpy()),
                    cm, deg, wkd, va.loc[coords, 'value'].to_numpy(), vb.loc[coords, 'value'].to_numpy())
    dx = vb.loc[coords, 'value'].to_numpy() - va.loc[coords, 'value'].to_numpy()
    k = len(SHAPE_SFX[mode])
    per = {l_: np.sqrt(change_test(dx[i * k:(i + 1) * k], s[i * k:(i + 1) * k, i * k:(i + 1) * k])['d2'])
           for i, l_ in enumerate(labs)}
    # Gesamtvektor = Verschiebung des Schwerpunkts aller gemeinsamen Wirbel: g = A·Δx, Cov = A·Σ_Δ·Aᵀ
    gv = mean_shift(dx, s, len(labs), k, range(len(labs)))
    out = dict(res=change_test(dx, s), labs=labs, per=per,
               disp={l_: float(np.linalg.norm(dx[i * k:(i + 1) * k])) for i, l_ in enumerate(labs)},
               gv=gv['g'], gv_sd=gv['sd'], gv_test=gv['test'],
               c_a=a.loc[labs].mean().to_numpy(), c_b=b.loc[labs].mean().to_numpy(), regions={})
    # Regionen: lumbal (L1–L5) im Sakrumsystem; thorakal relativ zu L1 (Eigenbeitrag der BWS, ohne die
    # über die Gliederkette mitgetragene LWS-Verschiebung – nur Translation von L1 herausgerechnet)
    lum = [l_ for l_ in labs if l_.startswith('L')]
    tho = [l_ for l_ in labs if l_.startswith('T')]
    for name, sel, ref in (('lumbal', lum, None), ('thorakal', tho, 'L1')):
        if len(sel) >= MIN_REGION_VERTEBRAE and (ref is None or ref in labs):
            r = mean_shift(dx, s, len(labs), k, [labs.index(l_) for l_ in sel],
                           labs.index(ref) if ref else None)
            out['regions'][name] = dict(r, labs=sel, anchor=a.loc[sel].mean().to_numpy())
    return out


GV_COLOR = 'magenta'
REGION_COLOR = {'lumbal': 'crimson', 'thorakal': 'deepskyblue'}
REGION_TEXT = {'lumbal': 'Lumbalvektor (L1–L5, Sakrumsystem)', 'thorakal': 'Thorakalvektor (relativ zu L1)'}
MIN_REGION_VERTEBRAE = 3


def _shape_figure_2d(vs_p, stations, mode, steps, overall=None):
    xs, ys = SHAPE_SFX[mode]
    cols = px.colors.sample_colorscale('Viridis', np.linspace(0, 0.9, len(stations)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-0.5, 0.5], y=[0, 0], mode='lines', line=dict(color='gray'),
                             name='S1 / Beckenachse (Ursprung)'))
    for t, col in zip(stations, cols):
        st_ = _station(vs_p, t, mode)
        fig.add_trace(go.Scatter(x=st_[xs], y=st_[ys], mode='lines+markers+text', text=st_.index,
                                 textposition='middle left', textfont=dict(size=9, color=col), line=dict(color=col),
                                 marker=dict(color=col, size=7), name=pd.Timestamp(t).strftime('%Y-%m-%d')))
    for (ta, tb), r in steps.items():
        a, b = _station(vs_p, ta, mode), _station(vs_p, tb, mode)
        for l_ in r['labs']:
            big = r['per'][l_] > Z95
            fig.add_annotation(x=b.at[l_, xs], y=b.at[l_, ys], ax=a.at[l_, xs], ay=a.at[l_, ys], xref='x', yref='y',
                               axref='x', ayref='y', showarrow=True, arrowhead=2, arrowwidth=3 if big else 1.5,
                               arrowcolor='darkorange' if big else '#444')
    if overall:                                         # Gesamtvektor erste → letzte Station (Schwerpunkt)
        i, j = SHAPE_SFX[mode].index(xs), SHAPE_SFX[mode].index(ys)
        ca, cb = overall['c_a'], overall['c_b']
        fig.add_trace(go.Scatter(x=[ca[i], cb[i]], y=[ca[j], cb[j]], mode='markers', name='Gesamtvektor (Schwerpunkt)',
                                 marker=dict(color=GV_COLOR, size=9, symbol=['circle-open', 'circle'])))
        fig.add_annotation(x=cb[i], y=cb[j], ax=ca[i], ay=ca[j], xref='x', yref='y', axref='x', ayref='y',
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=4, arrowcolor=GV_COLOR)
        for name, r in overall['regions'].items():     # an den Regionsschwerpunkt bei t₀ angesetzt
            p0, p1 = r['anchor'], r['anchor'] + r['g']
            fig.add_trace(go.Scatter(x=[p0[i]], y=[p0[j]], mode='markers', name=REGION_TEXT[name],
                                     marker=dict(color=REGION_COLOR[name], size=9, symbol='circle-open')))
            fig.add_annotation(x=p1[i], y=p1[j], ax=p0[i], ay=p0[j], xref='x', yref='y', axref='x', ayref='y',
                               showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=4, arrowcolor=REGION_COLOR[name])
    fig.update_yaxes(scaleanchor='x', title='kranial (WKD)')
    if mode == 'lat':                                   # Konvention seitlicher Bilder: anterior links, posterior rechts
        fig.update_xaxes(title='← anterior · posterior → (WKD)', autorange='reversed')
    else:
        fig.update_xaxes(title='Patient links (WKD Breite)')
    fig.update_layout(height=600, margin=dict(t=10, b=10), legend=dict(orientation='h', y=-0.12))
    return fig


def _cone(fig, tips, dirs, color, size):
    """Pfeilspitzen in 3D: Kegel an den Endpunkten, Richtung = Verschiebung, einheitliche Größe."""
    d = np.asarray(dirs, dtype=float)
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-9
    if not keep.any():
        return
    t, u = np.asarray(tips, dtype=float)[keep], d[keep] / n[keep, None]
    fig.add_trace(go.Cone(x=t[:, 0], y=t[:, 1], z=t[:, 2], u=u[:, 0], v=u[:, 1], w=u[:, 2], anchor='tip',
                          sizemode='absolute', sizeref=size, colorscale=[[0, color], [1, color]], showscale=False,
                          showlegend=False, hoverinfo='skip'))


def _shape_figure_3d(vs_p, stations, steps, overall=None):
    cols = px.colors.sample_colorscale('Viridis', np.linspace(0, 0.9, len(stations)))
    fig = go.Figure()
    pos = {}
    for t, col in zip(stations, cols):
        st_ = _station(vs_p, t, '3d')
        k = _ap_scale(vs_p, t)
        pos[t] = pd.DataFrame({'a': st_['.x'], 'l': st_['.z'] * k, 'c': st_['.y']})
        fig.add_trace(go.Scatter3d(x=pos[t]['a'], y=pos[t]['l'], z=pos[t]['c'], mode='lines+markers+text',
                                   text=st_.index, textfont=dict(size=9, color=col), line=dict(color=col, width=5),
                                   marker=dict(color=col, size=4), name=pd.Timestamp(t).strftime('%Y-%m-%d')))
    for big, col, w in ((False, '#444', 3), (True, 'darkorange', 7)):
        x, y, z, tips, dirs = [], [], [], [], []
        for (ta, tb), r in steps.items():
            for l_ in r['labs']:
                if (r['per'][l_] > Z95) == big:
                    a, b = pos[ta].loc[l_].to_numpy(), pos[tb].loc[l_].to_numpy()
                    x += [a[0], b[0], None]; y += [a[1], b[1], None]; z += [a[2], b[2], None]  # noqa: E702
                    tips.append(b)
                    dirs.append(b - a)
        if x:
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=col, width=w),
                                       name='Verschiebung > Fehler' if big else 'Verschiebung im Fehlerbereich'))
            _cone(fig, tips, dirs, col, 0.25 if not big else 0.35)
    if overall:                                         # Gesamtvektor: Schwerpunkt der gemeinsamen Wirbel
        ta, tb = stations[0], stations[-1]
        ca, cb = pos[ta].loc[overall['labs']].mean().to_numpy(), pos[tb].loc[overall['labs']].mean().to_numpy()
        fig.add_trace(go.Scatter3d(x=[ca[0], cb[0]], y=[ca[1], cb[1]], z=[ca[2], cb[2]], mode='lines+markers',
                                   line=dict(color=GV_COLOR, width=10), marker=dict(color=GV_COLOR, size=4),
                                   name='Gesamtvektor (Schwerpunkt)'))
        _cone(fig, [cb], [cb - ca], GV_COLOR, 0.5)
        k = _ap_scale(vs_p, ta)                         # a.p.-Komponente für die Darstellung skalieren
        for name, r in overall['regions'].items():
            p0 = pos[ta].loc[r['labs']].mean().to_numpy()
            g = np.array([r['g'][0], r['g'][2] * k, r['g'][1]])   # (ant, kran, links) → (a, l, c)
            p1 = p0 + g
            fig.add_trace(go.Scatter3d(x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]], mode='lines+markers',
                                       line=dict(color=REGION_COLOR[name], width=10),
                                       marker=dict(color=REGION_COLOR[name], size=4), name=REGION_TEXT[name]))
            _cone(fig, [p1], [g], REGION_COLOR[name], 0.5)
    fig.update_layout(height=650, margin=dict(t=10, b=10, l=0, r=0), legend=dict(orientation='h'),
                      # Blick von Patient links (+y): anterior (+x) erscheint links wie im 2D-Seitenbild.
                      # Kein autorange='reversed' – in 3D kehrt das die Seite je nach Kamera um.
                      scene=dict(xaxis_title='anterior (WKD)', yaxis_title='Patient links', zaxis_title='kranial',
                                 aspectmode='data', camera=dict(eye=dict(x=0.4, y=2.0, z=0.5), up=dict(x=0, y=0, z=1))))
    return fig


def shape_section(vs, pid, cm, deg, wkd, fmt):
    st.subheader('Formvektor')
    mode = st.segmented_control('Darstellung', list(SHAPE_MODE), default='lat', format_func=SHAPE_MODE.get,
                                key='shape_mode') or 'lat'
    vs_p = vs[vs['patient_id'] == pid]
    avail = [t for t in sorted(vs_p['date'].unique()) if len(_station(vs_p, t, mode)) >= 2]
    # Schlüssel je Patient: sonst behält das Widget die Daten des vorigen Patienten (→ leere Auswahl)
    stations = sorted(st.multiselect('Stationen', avail, default=avail, format_func=fmt, key=f'shape_st_{mode}_{pid}'))
    if st.toggle('Höheninvariant', key='shape_hinv',
                 help='Jede Station auf die mittlere Länge der Wirbelsäule (gemeinsame Wirbel) skaliert: zeigt nur die '
                      'Formänderung, ohne Höhenänderung durch OP/Kollaps und ohne Maßstabsfehler. Tests laufen auf den '
                      'skalierten Werten; der Maßstabsterm in Σ_Δ bleibt enthalten (konservativ).'):
        vs_p = height_invariant(vs_p, stations)
    need = {'lat': 'seitliche Stehaufnahme mit S1-Deckplatte und Kalibrierung',
            'ap': 'a.p.-Stehaufnahme mit S1 und Kalibrierung',
            '3d': 'seitliche und a.p.-Stehaufnahme am selben Tag'}[mode]
    if len(stations) < 2:
        st.info(f'Weniger als zwei Stationen mit Formdaten (nötig je Visite: {need}).')
        return
    pairs = list(zip(stations, stations[1:])) + ([(stations[0], stations[-1])] if len(stations) > 2 else [])
    res = {p: shape_step(vs_p, *p, mode, cm, deg, wkd) for p in pairs}
    consecutive = {p: r for p, r in res.items() if r and p in list(zip(stations, stations[1:]))}
    overall = res.get((stations[0], stations[-1]))
    fig = _shape_figure_3d(vs_p, stations, consecutive, overall) if mode == '3d' else \
        _shape_figure_2d(vs_p, stations, mode, consecutive, overall)
    st.plotly_chart(fig, key=f'shape_fig_{mode}')
    axis = {'.x': 'ant', '.y': 'kran', '.z': 'links', '.c': 'kran'}
    rows = []
    for (ta, tb), r in res.items():
        if r is None:
            rows.append({'Schritt': f'{fmt(ta)} → {fmt(tb)}', 'Wirbel': 0})
            continue
        rows.append({'Schritt': ('Gesamt ' if (ta, tb) == (stations[0], stations[-1]) and len(stations) > 2 else '')
                     + f'{fmt(ta)} → {fmt(tb)}',
                     'Jahre': (pd.Timestamp(tb) - pd.Timestamp(ta)).days / 365.25,
                     'Wirbel': f"{r['labs'][0]}–{r['labs'][-1]} ({len(r['labs'])})",
                     'D² (FG)': f"{r['res']['d2']:.1f} ({r['res']['df']})", 'p': r['res']['p'],
                     'auffällig': ', '.join(l_ for l_ in r['labs'] if r['per'][l_] > Z95),
                     'Gesamtvektor (WKD, ±95 %)': ', '.join(
                         f'{axis[sfx]} {v:+.2f} ± {Z95 * sd:.2f}'
                         for sfx, v, sd in zip(SHAPE_SFX[mode], r['gv'], r['gv_sd'])),
                     'p Gesamtvektor': r['gv_test']['p'],
                     **{f'{name.capitalize()} (WKD, ±95 %)': ', '.join(
                         f'{axis[sfx]} {v:+.2f} ± {Z95 * sd:.2f}'
                         for sfx, v, sd in zip(SHAPE_SFX[mode], rg['g'], rg['sd'])) + f" [{rg['labs'][0]}–{rg['labs'][-1]}]"
                        for name, rg in r['regions'].items()},
                     **{f'p {name}': rg['test']['p'] for name, rg in r['regions'].items()}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        'Jahre': st.column_config.NumberColumn(format='%.1f'), 'p': st.column_config.NumberColumn(format='%.3f'),
        'p Gesamtvektor': st.column_config.NumberColumn(format='%.3f'),
        'p lumbal': st.column_config.NumberColumn(format='%.3f'),
        'p thorakal': st.column_config.NumberColumn(format='%.3f')})
    st.caption('Jeder Schritt testet die Wirbel, die in *beiden* Stationen vorhanden sind – eine Ganzaufnahme '
               'zwischendurch geht so nicht verloren. Pfeile: aufeinanderfolgende Stationen; orange = Verschiebung '
               'dieses Wirbels jenseits des Fehlers (√D² > 1,96). Stationen unabhängig von t₁/t₂ oben. '
               '**Gesamtvektor** (magenta, erste → letzte Station): Verschiebung des Schwerpunkts der gemeinsamen '
               'Wirbel relativ zum Sakrum – z. B. anterior bei Lordoseverlust, kaudal bei Höhenverlust; eigener Test '
               'mit Σ = A·Σ_Δ·Aᵀ. Die Schrittvektoren addieren sich nur bei gleichem Wirbelsatz exakt zum Gesamtvektor. '
               '**Lumbalvektor** (rot): mittlere Verschiebung L1–L5 relativ zum Sakrum. **Thorakalvektor** (blau): '
               'mittlere Verschiebung der BWS-Wirbel *minus* Verschiebung von L1 – der Eigenbeitrag der BWS ohne die '
               'über die Kette mitgetragene LWS-Veränderung (nur die Translation von L1 wird herausgerechnet, eine '
               'Kippung am thorakolumbalen Übergang wirkt weiter über den Hebel). Beide angesetzt am Regionsschwerpunkt '
               f'bei t₀; je Region ≥ {MIN_REGION_VERTEBRAE} gemeinsame Wirbel, thorakal zusätzlich L1.')
    st.caption({'lat': 'Sakrumsystem: Ursprung S1-Deckplattenmitte, x entlang der S1-Deckplatte (anterior), '
                       'y senkrecht (kranial); Einheit Wirbeltiefe. Haltung (Beckenkippung) ist herausgerechnet. '
                       'Darstellung wie seitliche Bilder: anterior links, posterior rechts.',
                'ap': 'Koronares Beckensystem: Ursprung S1-Mitte, x entlang der Beckenquerachse (S1-Deckplatte a.p., '
                      'sonst Hüftlinie, sonst Bildhorizontale) Richtung Patient links; Einheit Wirbelbreite. '
                      'Darstellung wie im a.p.-Bild (Patient links = rechts).',
                '3d': 'Blick von Patient links, anterior links wie im Seitenbild. '
                      '3D näherungsweise (Spec §4.7): a.p. und seitliche Aufnahme sind nicht simultan; Zuordnung über '
                      'das Wirbel-Label, kraniale Achse aus der seitlichen Aufnahme. Die a.p.-Achse ist für die '
                      'Darstellung über die Segmentlängen auf Wirbeltiefe skaliert; der Test nutzt die '
                      'Originaleinheiten. Axiale Rotation ist aus Centroiden nicht bestimmbar.'}[mode])


def _day_sd(p: str, deg: float, wkd: float) -> float:
    """Tag-zu-Tag-Platzhalter je Komponente in ihrer Einheit (Grad bzw. WKD; LDI in % → 0)."""
    if p in WKD_PARAMS or '.' in p:
        return wkd
    return 0.0 if p in PCT_PARAMS else deg


def _scale_view(p: str) -> str | None:
    """Ansicht, deren Kalibrierung eine Längenkomponente skaliert (Winkel: None)."""
    if p.endswith(('.x', '.y')) or p == 'SVA_wkd':
        return 'lat'
    if p.endswith(('.z', '.c')) or p == 'C7_CSVL_wkd':
        return 'ap'
    return None


def sigma_delta(params, sd1, sd2, cm, deg: float, wkd: float, x1=None, x2=None) -> np.ndarray:
    """Σ_Δ = D₁RD₁ + D₂RD₂ + 2·diag(τ_gleicherTag² + τ_Tag²) + Maßstabsterm.

    Maßstabsterm: Jede Aufnahme hat einen relativen Maßstabsfehler ε (Kalibrierlinie), der alle Längen
    einer Ansicht gemeinsam und proportional zum Abstand von S1 skaliert: Δ ≈ (ε₂ − ε₁)·x →
    2·σ_s²·x·xᵀ je Ansicht (voll korreliert, radial). σ_s aus der Maßstabsstreuung zwischen Aufnahmen."""
    r = cm['corr'].reindex(index=params, columns=params).fillna(0.0).to_numpy()
    np.fill_diagonal(r, 1.0)
    tau = cm['tau']['tau'].reindex(params).fillna(0.0).to_numpy()
    day = np.array([_day_sd(p, deg, wkd) for p in params])
    s = cov_from_sd(sd1, r) + cov_from_sd(sd2, r) + 2 * np.diag(tau ** 2 + day ** 2)
    if x1 is not None and x2 is not None:
        xm = (np.asarray(x1, dtype=float) + np.asarray(x2, dtype=float)) / 2
        views = [_scale_view(p) for p in params]
        for view, sd_s in (cm.get('scale_sd') or {}).items():
            if sd_s and pd.notna(sd_s):
                v = np.where([w == view for w in views], xm, 0.0)
                s = s + 2 * sd_s ** 2 * np.outer(v, v)
    return nearest_psd(s)


def _targets(rows: pd.DataFrame) -> dict:
    """param → (Ziel, Skalen-SD, Quelle): PI-Ziel, wenn vorhanden (patientenspezifisch), sonst
    Populationsmittel; skaliert immer mit der Populations-SD, damit Komponenten vergleichbar sind."""
    if rows.empty:
        return {}
    pop = rows[(rows['art'] == 'Population') & rows['referenz_sd'].notna() & rows['abweichung'].notna()]
    pop = pop.drop_duplicates('param').set_index('param')
    pit = rows[rows['art'] == 'PI-Ziel'].drop_duplicates('param').set_index('param')
    out = {}
    for p, r in pop.iterrows():
        if p in pit.index:
            out[p] = (float(pit.at[p, 'referenz']), float(r['referenz_sd']), pit.at[p, 'quelle'])
        else:
            out[p] = (float(r['referenz']), float(r['referenz_sd']), r['quelle'])
    return out


def compare_visits(vv, pid, t1, t2, params, cm, deg, wkd, alpha, state1=None) -> dict | None:
    a = vv[(vv['patient_id'] == pid) & (vv['date'] == t1)].set_index('param')
    b = vv[(vv['patient_id'] == pid) & (vv['date'] == t2)].set_index('param')
    params = [p for p in params if p in a.index and p in b.index]
    if not params:
        return None
    x1, x2 = a.loc[params, 'value'].to_numpy(), b.loc[params, 'value'].to_numpy()
    s = sigma_delta(params, a.loc[params, 'sd'].fillna(0).to_numpy(), b.loc[params, 'sd'].fillna(0).to_numpy(),
                    cm, deg, wkd, x1, x2)
    res = change_test(x2 - x1, s)
    years = (pd.Timestamp(t2) - pd.Timestamp(t1)).days / 365.25
    table = pd.DataFrame({'param': params, 't1': x1, 't2': x2, 'Δ': x2 - x1,
                          'Δ/Jahr': (x2 - x1) / years if years > 0 else np.nan,
                          'SD_Δ': np.sqrt(np.diag(s)), 'RCI': res['rci']})
    table['auffällig'] = np.abs(table['RCI']) > Z95
    out = dict(table=table, res=res, drifting=res['p'] < alpha, years=years, sigma=s)
    tg = _targets(state1) if state1 is not None else {}
    tp = [p for p in params if p in tg]
    if tp:
        idx = [params.index(p) for p in tp]
        scale = np.array([tg[p][1] for p in tp])
        tgt = np.array([tg[p][0] for p in tp])
        dev = deviation_test(x1[idx], tgt, scale, alpha)
        comp, comp_sd = toward_target((x2 - x1)[idx] / scale, x1[idx] / scale, tgt / scale,
                                      s[np.ix_(idx, idx)] / np.outer(scale, scale))
        out.update(deviant=dev['deviant'], dev_p=dev['p'], toward=comp, toward_sd=comp_sd, target_params=tp,
                   typ=typology(dev['deviant'], out['drifting']))
    return out


@st.cache_data
def cohort_changes(mtime: float, include_tangent: bool, deg: float, wkd: float, alpha: float) -> pd.DataFrame:
    d = load(mtime)
    vv = visit_values(mtime, include_tangent)
    cm = d['uncert']['change']
    _, _, pat, _, _ = pi_analysis(mtime, include_tangent)
    pat = pat.set_index('patient_id')
    visits = d['visits'].set_index(['patient_id', 'date'])
    core = vv[vv['param'].isin(CORE)].groupby(['patient_id', 'date'])['param'].nunique()
    full = core[core == len(CORE)].reset_index()
    rows = []
    for pid, g in full.groupby('patient_id'):
        if len(g) < 2:
            continue
        t1, t2 = g['date'].min(), g['date'].max()
        vrow = visits.loc[(pid, t1)] if (pid, t1) in visits.index else pd.Series(dtype=object)
        st1 = _state(vv, pat.loc[pid] if pid in pat.index else None, vrow, pid, t1)
        r = compare_visits(vv, pid, t1, t2, CORE, cm, deg, wkd, alpha, st1)
        rows.append(dict(patient_id=pid, t1=t1, t2=t2, jahre=r['years'], D2=r['res']['d2'], p=r['res']['p'],
                         driftend=r['drifting'], abweichend=r.get('deviant'), typ=r.get('typ'),
                         richtung_norm=r.get('toward'), **{f'Δ{k}': v for k, v in zip(r['table']['param'],
                                                                                          r['table']['Δ'])}))
    return pd.DataFrame(rows)


def tab_change(d, mtime, count_tangent):
    if d['uncert'] is None or 'change' not in d['uncert']:
        st.info('Fehlermodell für Veränderungen fehlt – bitte `python degen_pipeline.py` bzw. „Von API laden“.')
        return
    cm = d['uncert']['change']
    vv = visit_values(mtime, count_tangent)
    vs = visit_shapes(mtime)
    _, _, pat, _, _ = pi_analysis(mtime, count_tangent)
    pat = pat.set_index('patient_id')
    visits = d['visits'].set_index(['patient_id', 'date'])

    st.caption('Σ_Δ = Σ_mess(t₁) + Σ_mess(t₂) + 2·(τ_gleicher Tag² + τ_Tag²). Σ_mess aus MC-/Rater-SD mit der '
               'Fehlerkorrelation zwischen Parametern (Rater am selben Bild); τ_gleicher Tag aus verschiedenen '
               'Aufnahmen desselben Tages; τ_Tag = Tag-zu-Tag-Fehler, voreingestellt aus der Schwankung der '
               '(biologisch konstanten) PI zwischen Visiten. Echte Haltungsschwankung zwischen Tagen ist darin nur '
               'teilweise enthalten → Ergebnisse eher zu liberal.')
    tau_day = cm.get('tau_day_pi')
    c = st.columns(3)
    alpha = c[0].select_slider('α', [0.01, 0.05, 0.1], value=0.05)
    deg = c[1].slider('τ_Tag Winkel (°)', 0.0, 10.0, float(round(tau_day, 1)) if pd.notna(tau_day) else 0.0, 0.1,
                      help=f"Voreinstellung aus PI: {tau_day:.1f}° ({cm.get('n_day_pairs_pi')} Visitenpaare)"
                      if pd.notna(tau_day) else 'nicht schätzbar')
    wkd = c[2].slider('τ_Tag Längen/Form (WKD)', 0.0, 0.5, 0.0, 0.01, help='Platzhalter, nicht identifiziert')
    sc = cm.get('scale_sd') or {}
    if sc:
        pairs = cm.get('scale_pairs') or {}
        st.caption(f"Maßstabsfehler je Aufnahme (Kalibrierlinie über die Wirbelbreite/-tiefe) σ_s: seitlich "
                   f"{100 * sc.get('lat', np.nan):.1f} % ({pairs.get('lat', 0)} Paare), a.p. "
                   f"{100 * sc.get('ap', np.nan):.1f} % ({pairs.get('ap', 0)} Paare), geschätzt aus verschiedenen "
                   'Aufnahmen desselben Tages – geht voll korreliert und proportional zum Abstand von S1 in alle '
                   'Längen/Formkoordinaten ein. **Absolute Höhenänderungen sind damit nur grob beurteilbar** (kleinste '
                   f'sicher erkennbare relative Änderung ≈ 1,96·√2·σ_s ≈ {100 * Z95 * np.sqrt(2) * sc.get("ap", np.nan):.0f} % a.p.). '
                   f"Aufnahmen, deren Maßstab > {100 * dp.CALIB_MAX_DEV:.0f} % vom Patientenmedian abweicht, sind nur "
                   f"markiert ({cm.get('n_calib_flagged', 0)} Messungen, calib_inconsistent) – nicht korrigiert, weil eine "
                   'echte Höhenänderung genauso aussieht.')
    unknown = [LABELS.get(p, p) for p in cm['tau'].index[cm['tau']['tau'].isna()] if '.' not in p]
    if unknown:
        st.caption(f"τ_gleicher Tag nicht schätzbar (< {dp.MIN_TAU_PAIRS} Paare) → 0 für: {', '.join(unknown)}")

    multi = vv.groupby('patient_id')['date'].nunique()
    pids = sorted(multi[multi >= 2].index)
    if not pids:
        st.info('Keine Patienten mit ≥ 2 Visiten.')
        return
    c1, c2, c3 = st.columns(3)
    pid = c1.selectbox('Patient', pids, key='chg_pid')
    dates = sorted(vv.loc[vv['patient_id'] == pid, 'date'].unique())
    fmt = lambda x: pd.Timestamp(x).strftime('%Y-%m-%d')  # noqa: E731
    t1 = c2.selectbox('Visite t₁', dates, index=0, format_func=fmt, key=f'chg_t1_{pid}')
    t2 = c3.selectbox('Visite t₂', dates, index=len(dates) - 1, format_func=fmt, key=f'chg_t2_{pid}')
    if t1 == t2:
        st.warning('Zwei verschiedene Visiten wählen.')
        return

    # Röntgenbilder der beiden Visiten, damit Verlauf und Pfeile am Bild nachvollziehbar sind
    if d['raw'] is not None:
        with st.expander('Röntgenbilder t₁ | t₂', expanded=True):
            view = st.segmented_control('Ansicht', ['lat', 'ap'], default='lat', key='chg_view',
                                        format_func=lambda v: 'seitlich' if v == 'lat' else 'a.p.') or 'lat'
            for col, t, k in zip(st.columns(2), (t1, t2), ('t1', 't2')):
                with col:
                    st.markdown(f'**{fmt(t)}**')
                    _image_panel(d, pid, t, view, f'chg_img_{k}')
            st.caption('Overlay aller Rater (Nodes, Deckplatten, Hüftachse, VPA-Linien bzw. Endwirbel/Apex). In die '
                       'Verlaufswerte gehen nur Stehaufnahmen ein. Bild-URLs gelten 24 h ab „Von API laden“.')

    # Verlauf
    pv = vv[vv['patient_id'] == pid]
    avail = [p for p in dp.PARAMS if p in set(pv['param'])]
    show = st.multiselect('Verlauf', avail, default=[p for p in ('PT', 'SS', 'LL', 'L1PA', 'cCobb_max') if p in avail],
                          format_func=lambda p: LABELS.get(p, p))
    if show:
        tr = pv[pv['param'].isin(show)].copy()
        tau = tr['param'].map(cm['tau']['tau']).fillna(0)
        day = tr['param'].map(lambda p: _day_sd(p, deg, wkd))
        scl = tr['param'].map(lambda p: (cm.get('scale_sd') or {}).get(_scale_view(p) or '', 0.0) or 0.0)
        tr['ci'] = Z95 * np.sqrt(tr['sd'].fillna(0) ** 2 + tau ** 2 + day ** 2 + (scl * tr['value']) ** 2)
        tr['Parameter'] = tr['param'].map(lambda p: LABELS.get(p, p))
        fig = px.line(tr, x='date', y='value', facet_row='Parameter', error_y='ci', markers=True,
                      labels={'value': '', 'date': ''})
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig.update_layout(height=170 * len(show), margin=dict(t=10, b=10))
        st.plotly_chart(fig)
        st.caption('Fehlerbalken: 95 %-Intervall aus Messfehler + Aufnahmefehler (gleicher Tag + Tag-zu-Tag).')

    # Skalarvektor
    st.subheader('Veränderung der Parameter')
    vrow = visits.loc[(pid, t1)] if (pid, t1) in visits.index else pd.Series(dtype=object)
    st1 = _state(vv, pat.loc[pid] if pid in pat.index else None, vrow, pid, t1)
    common = [p for p in dp.PARAMS if p != 'PI' and p in set(pv.loc[pv['date'] == t1, 'param'])
              and p in set(pv.loc[pv['date'] == t2, 'param'])]
    sel = st.multiselect('Komponenten', common, default=[p for p in common if p in CORE + ['cCobb_max', 'L4_tilt']]
                         or common, format_func=lambda p: LABELS.get(p, p))
    r = compare_visits(vv, pid, t1, t2, sel, cm, deg, wkd, alpha, st1)
    if r:
        m = st.columns(4)
        m[0].metric('D² (FG)', f"{r['res']['d2']:.1f} ({r['res']['df']})")
        m[1].metric('p', f"{r['res']['p']:.3f}", help='H₀: keine Veränderung über den Fehler hinaus')
        m[2].metric('Richtung Norm', f"{r['toward']:+.2f} ± {Z95 * r['toward_sd']:.2f}" if 'toward' in r else '–',
                    help='Komponente von Δ in Richtung Ziel/Norm, in Norm-SD-Einheiten; + = Annäherung')
        m[3].metric('Typ', r.get('typ', '–'),
                    help='Zustand bei t₁ (deutlich abweichend: p < α gegen Norm/Ziel) × Veränderung (D² p < α)')
        tab = r['table'].assign(param=lambda x: x['param'].map(lambda p: LABELS.get(p, p)))
        st.dataframe(tab, hide_index=True, column_config={
            c_: st.column_config.NumberColumn(format='%.2f') for c_ in ('t1', 't2', 'Δ', 'Δ/Jahr', 'SD_Δ', 'RCI')})
        st.caption(f"Abstand {r['years']:.1f} Jahre. RCI = Δ/SD_Δ; |RCI| > 1,96 → auffällig. Der Omnibus-D² "
                   'verliert Power, wenn sich nur eine Komponente ändert – deshalb beide ansehen.')
        pi = vv[(vv['patient_id'] == pid) & (vv['param'] == 'PI') & vv['date'].isin([t1, t2])].set_index('date')
        if len(pi) == 2:
            dpi = pi.at[t2, 'value'] - pi.at[t1, 'value']
            sd_pi = np.sqrt(pi['sd'].fillna(0).pow(2).sum() + 2 * deg ** 2)
            st.caption(f"Kontrolle: ΔPI = {dpi:+.1f}° (erwartet 0 ± {Z95 * sd_pi:.1f}°)"
                       + (' – **auffällig: Aufnahme- oder Annotationsproblem?**' if abs(dpi) > Z95 * sd_pi else ''))

    # Formvektor: alle Stationen, 2D seitlich / 2D a.p. / 3D
    shape_section(vs, pid, cm, deg, wkd, fmt)

    # Kohorte
    st.subheader(f"Kohorte: erste vs. letzte Visite ({', '.join(CORE)})")
    coh = cohort_changes(mtime, count_tangent, deg, wkd, alpha)
    if coh.empty:
        st.info('Keine Patienten mit allen Kernparametern an ≥ 2 Visiten.')
        return
    counts = coh['typ'].value_counts()
    cc = st.columns(5)
    cc[0].metric('Patienten', len(coh))
    for col, t in zip(cc[1:], ('unauffällig', 'kompensiert', 'beginnende Dekompensation', 'dekompensierend')):
        col.metric(t, int(counts.get(t, 0)))
    fig = px.scatter(coh, x='jahre', y='richtung_norm', color='typ', hover_name='patient_id', size=coh['D2'].clip(upper=40),
                     labels={'jahre': 'Beobachtungsdauer (Jahre)', 'richtung_norm': 'Veränderung Richtung Norm (SD-Einheiten)'})
    fig.add_hline(y=0, line_color='gray')
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    st.dataframe(coh.round(3), hide_index=True, height=300)
    st.download_button('Kohorte als CSV', coh.to_csv(index=False).encode(), 'degen_verlauf_kohorte.csv', 'text/csv')


# ---------------------------------------------------------------------------
# Winkelverlauf relativ zum Indexdatum (Patient.index_date, meist OP)
# ---------------------------------------------------------------------------

@st.cache_data
def course_inputs(mtime: float, include_tangent: bool, ap_type: str):
    """Alle Verlaufsserien (patient_id, series, date, value), Indexdaten, uuid→patient_id und klinische Sets."""
    d = load(mtime)
    meas = d['measure']
    ok = {'ok', 'tangente'} if include_tangent else {'ok'}
    p = d['params'][d['params']['status'].isin(ok)].merge(
        meas[['patient_id', 'date', 'img_type', 'view']], left_on='idx', right_index=True).dropna(subset=['date'])
    p = p[p['param'].isin(GLOBAL) & (p['view'] == 'lat')]      # globale Parameter: alle seitlichen Aufnahmen
    glob = p.groupby(['patient_id', 'param', 'date'], as_index=False)['value'].mean().rename(columns={'param': 'series'})
    _, _, pat, _, _ = pi_analysis(mtime, include_tangent)
    pi = pat.set_index('patient_id')['PI']                   # PI als Patientenkonstante (Phase 3)
    raw = d['raw']
    ap_raw, lat_raw = raw[raw['img_type'] == 'Xray-LWS-ap'], raw[raw['img_type'] == 'Xray-LWS-lat']
    ap = cs.ap_region_values(ap_raw, aa.build_angles_df(ap_raw), ap_type)
    ll = cs.ll_values(lat_raw, aa.build_angles_df(lat_raw, lateral_spans=(ap_type,)), ap_type)
    pill = ll.assign(series='PI_LL', value=ll['patient_id'].map(pi) - ll['value']).dropna(subset=['value'])
    values = pd.concat([glob, ll, pill, ap], ignore_index=True)
    rp = raw['referring_patient'].map(lambda x: x if isinstance(x, dict) else {})
    index_dates = {pid: pd.to_datetime(r.get('index_date'), errors='coerce') for pid, r in zip(raw['patient_id'], rp)
                   if r.get('index_date')}
    uuid_to_pid = {r['uuid']: pid for pid, r in zip(raw['patient_id'], rp) if r.get('uuid')}
    for f in ('degen_entries', 'clinical_entries'):          # clinical_entries.pkl: Cache der clinical_timecourse_app
        try:
            entries = pd.read_pickle(f'{dp.DATA_DIR}/{f}.pkl')
            break
        except FileNotFoundError:
            entries = None
    clin = cs.patient_clinical(entries, uuid_to_pid, index_dates) if entries is not None else None
    return values, index_dates, clin


COURSE_LABELS = {**LABELS, 'LL': 'LL (T12–S1)', 'PI_LL': 'PI − LL', 'U-LU': 'U-LU (a.p.)', 'L-LU': 'L-LU (a.p.)', 'LU': 'LU = max(U-LU, L-LU) (a.p.)'}


def course_figure(c: pd.DataFrame, overlay: bool):
    c = c.assign(name=c['series'].map(lambda x: COURSE_LABELS.get(x, x)))
    b = cs.binned_mean(c).assign(name=lambda x: x['series'].map(lambda y: COURSE_LABELS.get(y, y)))
    kw = dict(color='name') if overlay else dict(facet_col='name', facet_col_wrap=3)
    fig = px.line(c.assign(grp=c['patient_id'] + c['series']).sort_values('t_months'), x='t_months', y='delta',
                  line_group='grp', hover_name='patient_id', markers=True,
                  labels={'t_months': 'Monate seit Indexdatum', 'delta': 'Δ zur Baseline (°)', 'name': ''}, **kw)
    fig.update_traces(opacity=0.25, showlegend=False)
    mean = px.line(b, x='t', y='delta', error_y='sd', markers=True, hover_data=['n'], **kw)
    for tr in mean.data:
        tr.update(line=dict(width=4), showlegend=overlay)
        fig.add_trace(tr)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.add_hline(y=0, line_color='gray')
    fig.update_layout(height=380 if overlay else 330 * -(-c['series'].nunique() // 3) + 60, margin=dict(t=30, b=10))
    return fig


def tab_course(d, mtime, count_tangent):
    if d['raw'] is None:
        st.info('Rohmessungen fehlen – bitte „Von API laden“.')
        return
    ap_type = st.radio('Methode (LL, PI − LL, L-LU, U-LU, LU)', ['Cobb-regional', 'Spline'], horizontal=True,
                       help='Cobb-regional: vom Untersucher gezeichnete Linien. Spline: aus der Spline abgeleiteter Winkel. LL = '
                            'zusammengesetzte Spanne T12→S1 (wie Frontend/ICC-Snapshot), a.p.-Regionen je Untersucher und Bild '
                            'das Maximum. Danach Mittel über Untersucher/Aufnahmen der Visite.')
    values, index_dates, clin = course_inputs(mtime, count_tangent, ap_type)
    st.caption('Zeitlicher Anker: `Patient.index_date` (meist OP). **Baseline = 0**: letzte Visite vor dem '
               'Indexdatum; alle Visiten am Indexdatum oder danach sind Veränderungen dazu. Patienten ohne Indexdatum, '
               'ohne Vor- oder ohne Nachvisite entfallen. PI − LL = Patienten-PI (Phase 3) − LL der Visite. '
               'Dünne Linien: einzelne Patienten; dick: Kohortenmittel ± SD je Zeitfenster (je Patient zuerst gemittelt).')
    keep = None
    with st.expander('Filter (klinische Ereignisse)', expanded=False):
        if clin is None:
            st.info('Keine klinischen Einträge geladen – „Von API laden“ (lädt /api/clinicalentrydata/).')
        else:
            st.caption('Therapie, Spezifikation, Fusion: Index-Ereignis (Eintrag am Indexdatum, sonst der nächste danach). Komplikationen: alle Einträge '
                       'des Patienten. Felder sind noch nicht vollständig befüllt – „(nicht erfasst)“ wählbar.')
            sel, cols = {}, st.columns(3)
            for i, (f, label) in enumerate(cs.FILTERS.items()):
                opts = sorted(set().union(*clin[f]), key=lambda x: (x == cs.NA, int(x) if x.isdigit() else 0, x))
                sel[f] = cols[i % 3].multiselect(label, opts, key=f'course_{f}')
            keep = cs.apply_filters(clin, sel)
    if keep is not None:
        values = values[values['patient_id'].isin(keep)]
    c = cs.course(values, index_dates)
    n_idx = len({p for p in values['patient_id'].unique() if p in index_dates})
    st.caption(f"{values['patient_id'].nunique()} Patienten in der Auswahl, davon {n_idx} mit Indexdatum.")

    views = {'Lateral: LL / PI − LL': (['LL', 'PI_LL'], False),
             'a.p. + lateral: LL, PI − LL, L-LU, U-LU, LU': (['LL', 'PI_LL', 'L-LU', 'U-LU', 'LU'], True),
             'Globale Parameter': (GLOBAL, False)}
    for tab, (title, (series, overlay)) in zip(st.tabs(list(views)), views.items()):
        with tab:
            sub = c[c['series'].isin(series)]
            if sub.empty:
                st.info('Keine Patienten mit Indexdatum, Vor- und Nachvisite in dieser Auswahl.')
                continue
            st.caption(' · '.join(f"{COURSE_LABELS.get(s, s)}: {g['patient_id'].nunique()} Pat."
                                  for s, g in sub.groupby('series')))
            st.plotly_chart(course_figure(sub, overlay), key=f'course_fig_{title}')
            st.download_button('CSV', sub.to_csv(index=False).encode(), 'degen_verlauf_index.csv', 'text/csv',
                               key=f'course_csv_{title}')


def main():
    st.set_page_config(page_title='Degenerative Skoliose', layout='wide')
    st.title('Degenerative Skoliose – Verlaufsmodell')
    st.caption('Forschungssoftware, keine klinische Entscheidungsunterstützung.')
    count_tangent = sidebar()
    try:
        mtime = os.path.getmtime(CACHE)
    except OSError:
        st.info('Noch keine Daten. Links Token eingeben und „Von API laden“.')
        return
    d = load(mtime)
    if d['raw'] is None:
        st.warning('Datenstand stammt von vor Schritt 1c (keine Rohmessungen/koronaren Kurven gespeichert). '
                   'Bitte einmal „Von API laden“.')
    t1, t2, t3, t4, t5, t6, t8, t7 = st.tabs(['Abdeckung', 'Visite', 'PI (Patientenkonstante)', 'Messunsicherheit',
                                              'Normabweichung', 'Verlauf', 'Winkelverlauf (Index)',
                                              'Kalibrierung & QC'])
    with t1:
        tab_coverage(d['measure'], d['cov'], d['visits'], count_tangent)
    with t2:
        tab_visit(d)
    with t3:
        tab_pi(d, mtime, count_tangent)
    with t4:
        tab_uncertainty(d)
    with t5:
        tab_norm(d, mtime, count_tangent)
    with t6:
        tab_change(d, mtime, count_tangent)
    with t8:
        tab_course(d, mtime, count_tangent)
    with t7:
        tab_calibration(d['measure'], d['calib'], d['vbh'])


main()
