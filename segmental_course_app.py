"""
Segmentaler Therapieverlauf – Streamlit (Konzept: docs/degen_skoliose_verlauf_segmental_bottom_up.md)
Stand (Schritt 4): Tabs F1–F6 der Auswertung (segmental_analysis.py), Bericht über den Sidebar-Button.
Bisher: Schritt 0 – Kohorte (VALID), Therapie-Episoden, Abdeckung; Schritt 2/2b – Tab Episode (Segmentwinkel, Slip, MDC,
Overlay mit Nodes, Spline und Hinterkanten-Ecken); Schritt 3 – Implantate (Overlay, Fusionsstrecke, Cages, QC).

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run segmental_course_app.py
"""
import io
import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

import plotly.express as px

import segmental_analysis as sa
import segmental_course_pipeline as sp
from spine_geom import SPINE_LABELS
from spine_geom.segmental import GROUPS, episode_angles, frontend_spline, node_tangents, segment_classes

STAGE_H = 600                    # Nodes liegen im Konva-Stage-Raum (Bild auf Höhe 600 skaliert)

st.set_page_config(page_title='Segmentaler Therapieverlauf', layout='wide')
STATUS = ['auswertbar', 'kein Verlauf', 'keine Baseline', 'keine gemeinsame Ebene']


@st.cache_data
def load(mtime_ep: float, mtime_raw: float):
    """Argumente nur als Cache-Schlüssel (neuer Pipeline-Lauf → neu laden)."""
    ep = pd.read_pickle(sp.EPISODES)
    raw = pd.read_pickle(sp.RAW)
    valid = set(sp.patients(raw).query('valid')['patient'])
    d = {'ep': ep[ep['patient'].isin(valid)], 'n_valid': len(valid)}     # nur aktuell validierte Patienten (§3.8)
    for k, path in (('visits', sp.VISITS), ('angles', sp.ANGLES), ('mdc', sp.MDC), ('slip', sp.SLIPS),
                    ('implants', sp.IMPLANTS)):
        d[k] = pd.read_pickle(path) if os.path.exists(path) else None
    if d['angles'] is not None:                                           # Rohmessungen nur für das Overlay
        d['meas'] = raw[raw['m_uuid'].isin(set(d['angles']['m_uuid']))].rename(columns={'id': 'image_id'}) \
            .set_index('m_uuid')[['img', 'img_type', 'xray_measurement_data', 'image_id']]
    d['corners'] = pd.read_csv(sp.CORNERS) if os.path.exists(sp.CORNERS) else None
    d['suggest'] = json.load(open(sp.SUGGEST)) if os.path.exists(sp.SUGGEST) else {}
    return d


def slip_corners(image_id, m_uuid) -> tuple[dict, str]:
    """Hinterkanten-Ecken {Label: (PS, PI)} in Stage-Koordinaten: manuell (Durchgang 1) vor Modell."""
    c = D['corners']
    if c is not None:
        g = c[(c['image_id'] == image_id) & (c['pass'] == 1) & (c['level'] != 'SKIP')].dropna(subset=['ps_x', 'pi_x'])
        if len(g):
            return {r.level: ((r.ps_x, r.ps_y), (r.pi_x, r.pi_y)) for r in g.itertuples()}, 'manuell'
    hit = D['suggest'].get(m_uuid) or {}
    f = hit.get('f')
    cor = (hit.get('resp') or {}).get('corners') or {}
    return ({lab: ((v['PS']['x'] / f, v['PS']['y'] / f), (v['PI']['x'] / f, v['PI']['y'] / f))
             for lab, v in cor.items()} if f else {}), 'Modell'


@st.cache_data(show_spinner=False)
def fetch_image(url: str):
    r = requests.get(url, timeout=30)
    return r.content if r.ok else None


IMPLANT_COLOR = {'pedicle': (255, 80, 200), 'iliac': (255, 140, 0), 'cage': (0, 255, 255)}


def overlay(m, seg_vals: dict, title: str, unit: str = '°', corners: dict | None = None):
    """Röntgenbild mit Nodes, Frontend-Spline und Senkrechten am Node, Segmentwerte (mit Vorzeichen) daneben.
    corners {Label: (PS, PI)} in Stage-Koordinaten: Hinterkante als Linie PS–PI (für den Slip).
    Implantate des Bildes (seg_implants): Linie Start → Ende, Punkt am Start (Schraubenkopf / Einbringseite)."""
    content = fetch_image(m['img'])
    if content is None:
        st.warning(f'{title}: Bild nicht ladbar (signierte URL > 24 h alt?) – Daten neu laden.')
        return
    img = Image.open(io.BytesIO(content)).convert('RGB')
    f = img.height / STAGE_H
    so = m['xray_measurement_data']['stateObject']
    nodes = [(n['x'], n['y']) for n in so['nodes']]
    labels = SPINE_LABELS[so['vertebra_range'][0]:so['vertebra_range'][1] + 1]
    dr = ImageDraw.Draw(img)
    w = max(2, img.height // 400)
    dr.line([tuple(p * f) for p in frontend_spline(nodes)], fill=(0, 200, 255), width=w)
    for (x, y), (anchor, d) in zip(nodes, node_tangents(nodes)):
        n = np.array([-d[1], d[0]]) * 25                               # Senkrechte, ±25 Stage-px
        dr.line([tuple((anchor - n) * f), tuple((anchor + n) * f)], fill=(255, 200, 0), width=w)
        dr.ellipse([(x - 3) * f, (y - 3) * f, (x + 3) * f, (y + 3) * f], fill=(255, 60, 60))
    imp = D['implants'] if D['implants'] is not None else pd.DataFrame(columns=['image_id', 'cls'])
    for r in imp[(imp['image_id'] == m['image_id']) & (imp['cls'] != 'NONE')].itertuples():
        (x0, y0), (x1, y1) = r.ends
        dr.line([(x0 * f, y0 * f), (x1 * f, y1 * f)], fill=IMPLANT_COLOR[r.cls], width=2 * w)
        dr.ellipse([(x0 - 4) * f, (y0 - 4) * f, (x0 + 4) * f, (y0 + 4) * f], outline=IMPLANT_COLOR[r.cls], width=w)
    for ps, pi in (corners or {}).values():
        dr.line([(ps[0] * f, ps[1] * f), (pi[0] * f, pi[1] * f)], fill=(80, 255, 80), width=w)
        for x, y in (ps, pi):
            dr.ellipse([(x - 2.5) * f, (y - 2.5) * f, (x + 2.5) * f, (y + 2.5) * f], fill=(80, 255, 80))
    for k in range(len(nodes) - 1):
        seg = f'{labels[k]}/{labels[k + 1]}'
        if seg in seg_vals:
            (x1, y1), (x2, y2) = nodes[k], nodes[k + 1]
            dr.text(((max(x1, x2) + 30) * f, (y1 + y2) / 2 * f), f'{seg} {seg_vals[seg]:+.1f}{unit}',
                    fill=(255, 255, 0), font_size=max(14, img.height // 60))
    xs, ys = [x * f for x, _ in nodes], [y * f for _, y in nodes]
    h = max(ys) - min(ys)                                              # auf die Wirbelsäule zuschneiden
    img = img.crop((max(0, min(xs) - 0.45 * h), max(0, min(ys) - 0.12 * h),
                    min(img.width, max(xs) + 0.45 * h), min(img.height, max(ys) + 0.12 * h)))
    st.image(img, caption=title, width='stretch')


def _mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


# --- Sidebar: Daten -------------------------------------------------------------------------------
st.sidebar.header('Daten')
token = st.sidebar.text_input('Bearer Token', type='password',
                              help='xray-annotator.renecol.org → Account info → Copy token')
if st.sidebar.button('Von API laden + neue Patienten rechnen', disabled=not token.strip()):
    with st.status('Lade und rechne …', expanded=True) as s:
        sp.run(token.strip(), progress=s.write)
        s.update(label='Fertig', state='complete')
    st.cache_data.clear()
if st.sidebar.button('Nur neue Patienten rechnen (Cache)'):
    with st.status('Rechne …', expanded=True) as s:
        sp.update(progress=s.write, token=token.strip() or None)
    st.cache_data.clear()
st.sidebar.caption('Bereits berechnete Patienten werden nicht neu gerechnet. Mit Token fragt „Nur neue Patienten '
                   'rechnen“ fehlende Slip-Modellwerte ab, ohne die Messungen neu zu laden. '
                   'Nach Codeänderungen: `XRAY_TOKEN=… python segmental_course_pipeline.py --rebuild`.')

if not os.path.exists(sp.EPISODES):
    st.info('Noch keine Episoden berechnet – Token eingeben und laden, oder „Nur neue Patienten rechnen“.')
    st.stop()
D = load(_mtime(sp.EPISODES), _mtime(sp.RAW))
ep, n_valid = D['ep'], D['n_valid']

st.title('Segmentaler Therapieverlauf')
st.caption('Machbarkeitsnachweis – vorläufige Zahlen, immer mit n lesen.')
tab_cov, tab_ep, t1, t2, t3, t4, t5, t6 = st.tabs(['Abdeckung', 'Episode', 'F1 Verfahren × Segment', 'F2 Kopplung',
                                                   'F3 Cage', 'F4 Fusionslänge', 'F5 Regional / Nachbarn',
                                                   'F6 Korrekturverlust'])
WHEN = st.sidebar.radio('Zeitpunkt für F1–F5', ['late', 'early'], horizontal=True,
                        help='late = letzte Visite vor dem nächsten Ereignis, early = erste Visite ≤ 3 Monate')
if st.sidebar.button('Bericht schreiben (reports/segmental_therapieeffekt.md)'):
    with st.spinner('Bootstrap …'):
        st.sidebar.success(sa.write_report())


@st.cache_data(show_spinner='Auswertung …')
def analysis(mtime: float, when: str):
    d = sa.load()
    ch, ep, mdc = d['ch'], d['ep'], d['mdc']
    entries = pd.read_pickle(f'{sp.dp.DATA_DIR}/degen_entries.pkl')
    return {'f1': sa.f1(ch, mdc, when), 'f2': sa.f2(ch, when), 'f3': sa.f3(ch, ep, mdc, when), 'f4': sa.f4(ch, when),
            'f5a': sa.f5a(ch, when), 'f5b': sa.f5b(ch, when), 'f6': sa.f6(ch, mdc, entries)}


NUM = {c: st.column_config.NumberColumn(format='%.1f') for c in
       ('Median', 'Q25', 'Q75', 'CI_lo', 'CI_hi', '> MDC95 %', 'Verlust > MDC95 %', 'rho', 'months', 'loss', 'mdc95')}
CI_NOTE = f'CI = 95-%-Cluster-Bootstrap über Patienten, erst ab {sa.N_MIN_CI} Patienten. Machbarkeitsnachweis – n beachten.'
A = analysis(_mtime(sp.CHANGES), WHEN) if os.path.exists(sp.CHANGES) else None

# --- Tab Abdeckung ---------------------------------------------------------------------------------
with tab_cov:
    last = ep['added_at'].max()
    c = st.columns(4)
    c[0].metric('Validierte Patienten', n_valid)
    c[1].metric('Berechnet', ep['patient'].nunique())
    c[2].metric('Neu im letzten Lauf', ep.loc[ep['added_at'] == last, 'patient'].nunique(), help=f'Lauf {last}')
    c[3].metric('Ohne Therapieereignis', int((ep['status'] == 'keine Therapie').sum()))

    eps = ep[ep['status'] != 'keine Therapie']
    st.subheader('Episoden je Gruppe')
    if eps.empty:
        st.info('Keine Therapie-Episoden bei den validierten Patienten.')
    else:
        tab = pd.crosstab(eps['group'], eps['status']).reindex(index=list(GROUPS), columns=STATUS, fill_value=0)
        tab['gesamt'] = tab.sum(axis=1)
        st.dataframe(tab, width='content')
        aus = eps[eps['status'] == 'auswertbar']
        st.caption(f'Auswertbar: {len(aus)} Episoden bei {aus["patient"].nunique()} Patienten · '
                   f'Ebenen a.p.+seitl. {int(aus["planes"].map(len).eq(2).sum())}, nur eine Ebene '
                   f'{int(aus["planes"].map(len).eq(1).sum())} · Baseline > 12 Mon. {int(eps["baseline_old"].sum())} · '
                   f'Aufnahme am Ereignistag (= Baseline) {int(eps["same_day_visit"].sum())}')

        st.subheader('Episoden')
        f = st.columns(2)
        groups = f[0].multiselect('Gruppe', list(GROUPS), default=list(GROUPS))
        status = f[1].multiselect('Status', STATUS, default=STATUS)
        show = eps[eps['group'].isin(groups) & eps['status'].isin(status)].copy()
        show['neu'] = show['added_at'] == last
        for col in ('levels', 'planes', 'baseline_planes'):
            show[col] = show[col].map(lambda v: ', '.join(v) if isinstance(v, list) else '')
        cols = ['patient_id', 'date', 'group', 'revision', 'levels', 'fused_segments', 'fused_to_pelvis', 'status',
                'planes', 'baseline_date', 'baseline_planes', 'early_date', 'late_date', 'n_post_visits',
                'months_late', 'months_from_index', 'baseline_old', 'group_source', 'therapy_raw', 'neu']
        st.dataframe(show[cols].sort_values(['patient_id', 'date']), hide_index=True, width='stretch',
                     column_config={'date': st.column_config.DateColumn('Ereignis'),
                                    'baseline_date': st.column_config.DateColumn('Baseline'),
                                    'early_date': st.column_config.DateColumn('early'),
                                    'late_date': st.column_config.DateColumn('late'),
                                    'months_late': st.column_config.NumberColumn('Mon. bis late', format='%.1f'),
                                    'months_from_index': st.column_config.NumberColumn('Mon. ab Index', format='%.1f'),
                                    'levels': 'Segmente (Text)', 'planes': 'Ebenen'})
        st.download_button('CSV', show[cols].to_csv(index=False).encode(), 'seg_episodes.csv', 'text/csv')

        if 'fusion_qc' in eps:
            st.subheader('QC Fusionsstrecke: Implantat-Detektion vs. Freitext')
            fu = eps[eps['group'] == 'Fusion'].copy()
            st.dataframe(fu['fusion_qc'].value_counts().rename('Episoden'), width='content')
            for col in ('levels', 'fused_implant', 'implant_pre'):
                fu[col] = fu[col].map(lambda v: ', '.join(v) if isinstance(v, list) else '')
            fu['cages'] = fu['cages'].map(lambda v: ', '.join(f'{a} ({b or "?"})' for a, b in v)
                                          if isinstance(v, list) else '')
            fu['fused_total'] = fu['fused_total'].map(lambda v: ', '.join(v) if isinstance(v, list) else '')
            st.dataframe(fu[['patient_id', 'date', 'status', 'fusion_qc', 'fused_src', 'fused_total', 'levels', 'fused_implant', 'fused_segments',
                             'fused_to_pelvis', 'pelvis_implant', 'implant_pre', 'cages', 'therapy_raw']]
                         .sort_values(['fusion_qc', 'patient_id']), hide_index=True, width='stretch',
                         column_config={'date': st.column_config.DateColumn('OP'), 'levels': 'Text',
                                        'fused_implant': 'Implantat (1. Visite post)', 'implant_pre': 'Schrauben prä'})

# --- Tab Episode ----------------------------------------------------------------------------------
def episode_tab():
    if D['visits'] is None:
        st.info('Noch keine Segmentwinkel – `python segmental_course_pipeline.py --rebuild`.')
        return
    aus = ep[ep['status'] == 'auswertbar'].sort_values(['patient_id', 'date'])
    if aus.empty:
        st.info('Keine auswertbare Episode.')
        return
    c = st.columns([2, 1])
    i = c[0].selectbox('Episode', aus.index, format_func=lambda k: f"{aus.at[k, 'patient_id']} · "
                       f"{aus.at[k, 'date']:%Y-%m-%d} · {aus.at[k, 'group']}"
                       f"{' (Revision)' if aus.at[k, 'revision'] else ''} · {', '.join(aus.at[k, 'levels']) or '–'}")
    e = aus.loc[i]
    sv = D['visits'][D['visits']['patient'] == e['patient']]
    t = episode_angles(e, sv)
    if t.empty:
        st.info('Keine Segmentwinkel mit Baseline und Verlauf in derselben Ebene.')
        return
    t = t.merge(D['mdc'][['plane', 'seg', 'mdc95']], on=['plane', 'seg'], how='left')
    when = c[1].radio('Verlauf', ['early', 'late'], index=1, horizontal=True,
                      help='early = erste Visite ≤ 3 Monate nach dem Ereignis, late = letzte Visite vor dem nächsten')
    st.caption('Oberstes Segment je Messung entfällt (Spline-Tangente einseitig); L5/S1 gegen die gezeichnete '
               'S1-Deckplatte. Overlay zeigt die Spline-Senkrechten.')
    st.caption(f"Ereignis {e['date']:%d.%m.%Y} · {e['therapy_raw']} · Index {e['index_date']:%d.%m.%Y}"
               if pd.notna(e['index_date']) else f"Ereignis {e['date']:%d.%m.%Y} · {e['therapy_raw']}")
    join = lambda v: ', '.join(v) if isinstance(v, (list, tuple)) and len(v) else '–'
    if e['group'] == 'Fusion' and 'fused_implant' in e:
        st.markdown(f"**Fusionsstrecke** Implantat: {join(e['fused_implant'])}"
                    f"{' + Becken' if e['pelvis_implant'] else ''} · Text: {join(e['levels'])} "
                    f"({int(e['fused_segments']) if pd.notna(e['fused_segments']) else '?'} Segm.{', Becken' if e['fused_to_pelvis'] else ''}) · QC: **{e['fusion_qc']}**"
                    f" · Cages: {join([f'{lv} von {sd}' if sd else lv for lv, sd in e['cages']])}"
                    f"{' · Schrauben schon prä: ' + join(e['implant_pre']) if e['implant_pre'] else ''}")
    if e['group'] == 'Fusion' and 'fused_total' in e:
        st.markdown(f"→ verwendet ({e['fused_src']}): neu {join(e['fused_new'])} · insgesamt fusioniert "
                    f"{join(e['fused_total'])}")
    seg_lv = (e['fused_total'] if 'fused_total' in e and isinstance(e['fused_total'], list) else e['levels']) \
        if e['group'] == 'Fusion' else e['levels']
    for plane, title, unit in (('ap', 'a.p. (+ = Öffnung nach Patient links)', '°'),
                               ('lat', 'seitlich (+ = Lordose)', '°'),
                               ('slip', 'Slip in % WKD (+ = Anterolisthese)', '%')):
        g = t[t['plane'] == plane]
        if g.empty:
            continue
        st.subheader(title)
        d0, d1 = g['date_pre'].iloc[0], g[f'date_{when}'].iloc[0]
        if pd.isna(d1):
            st.info(f'Keine {when}-Visite in dieser Ebene.')
            continue
        tab = g[['seg', 'pre', when, f'd_{when}', f'k_{when}', 'mdc95', 'edge']].copy()
        tab.insert(1, 'Klasse', tab['seg'].map(segment_classes(e['group'], list(seg_lv), tab['seg'])))
        tab['relevant'] = tab[f'd_{when}'].abs() > tab['mdc95']
        tab['edge'] = tab['edge'].map({True: '⚠ L5/S1 ohne S1-Deckplatte (Spline, zu klein)', False: ''})
        fmt = '%+.1f' + unit.replace('%', '%%')
        if plane == 'slip':
            src = D['visits'][(D['visits']['patient'] == e['patient']) & (D['visits']['plane'] == 'slip')]
            src = src.pivot_table(index='seg', columns='date', values='src', aggfunc='first')
            tab['Quelle'] = [' → '.join(str(src.at[sg, d]) if sg in src.index and d in src.columns else '–'
                                        for d in (d0, d1)) for sg in tab['seg']]
        st.dataframe(tab, hide_index=True, width='stretch', column_config={
            'seg': 'Segment', 'pre': st.column_config.NumberColumn(f'prä {d0:%d.%m.%Y}', format=fmt),
            when: st.column_config.NumberColumn(f'{when} {d1:%d.%m.%Y}', format=fmt),
            f'd_{when}': st.column_config.NumberColumn('Δ', format=fmt),
            f'k_{when}': st.column_config.NumberColumn('Korrektur K', format=fmt,
                                                       help='|prä| − |post|, > 0 = verbessert (Slip: Reposition)'),
            'mdc95': st.column_config.NumberColumn('MDC95', format=fmt.replace('+', '')),
            'relevant': st.column_config.CheckboxColumn('|Δ| > MDC95'), 'edge': 'Hinweis'})
        a = D['angles']
        cols = st.columns(2)
        for col, d, lab in ((cols[0], d0, 'prä'), (cols[1], d1, when)):
            m = a[(a['patient'] == e['patient']) & (a['plane'] == ('lat' if plane == 'slip' else plane))
                  & (a['date'] == d)]
            mu = m['m_uuid'].iloc[0]                                           # erste Messung = erster Rater
            with col:
                if plane == 'slip':
                    img_id = D['meas'].at[mu, 'image_id']
                    cor, csrc = slip_corners(img_id, mu)
                    sl = D['slip'][(D['slip']['image_id'] == img_id)
                                   & (D['slip']['m_uuid'].isna() | (D['slip']['m_uuid'] == mu))]
                    overlay(D['meas'].loc[mu], sl.groupby('seg')['slip'].mean().to_dict(),
                            f"{lab} {d:%d.%m.%Y} · Ecken {csrc}", '%', cor)
                else:
                    overlay(D['meas'].loc[mu], m[m['m_uuid'] == mu].set_index('seg')['angle'].to_dict(),
                            f"{lab} {d:%d.%m.%Y} · {D['meas'].at[mu, 'img_type']} · "
                            f"Rater {m['rater_id'].iloc[0]} von {m['rater_id'].nunique()}")



with tab_ep:
    episode_tab()

# --- Tabs F1–F6 -----------------------------------------------------------------------------------
def table(df, **kw):
    st.dataframe(df, hide_index=True, width='stretch', column_config={**NUM, **kw})


if A is None:
    for t in (t1, t2, t3, t4, t5, t6):
        t.info('Noch keine seg_changes – `python segmental_course_pipeline.py --rebuild`.')
    st.stop()

with t1:
    st.caption('Δ = post − prä je Segment (Median). ' + CI_NOTE)
    by_seg, by_cls, lis = A['f1']
    plane = st.radio('Größe', list(sa.PLANES), format_func=sa.PLANES.get, horizontal=True, key='f1p')
    g = by_seg[by_seg['plane'] == plane]
    if len(g):
        hm = g.pivot_table(index='seg', columns='group', values='Median')
        hm = hm.loc[sorted(hm.index, key=sa._seg_key)]
        n = g.pivot_table(index='seg', columns='group', values='n').reindex_like(hm)
        fig = px.imshow(hm, color_continuous_scale='RdBu_r', color_continuous_midpoint=0, aspect='auto',
                        labels={'color': 'Median Δ'}, text_auto='.1f')
        fig.update_traces(customdata=n.values, hovertemplate='%{y} · %{x}<br>Median Δ %{z:.1f}<br>n %{customdata}')
        st.plotly_chart(fig, width='stretch')
    table(g)
    st.subheader('Δ und K je Segmentklasse')
    table(by_cls[by_cls['plane'] == plane])
    if plane == 'slip':
        st.subheader(f'Listhese (|Slip| ≥ {sa.LISTHESIS:.0f} % WKD) prä vs. post, %')
        table(lis)

with t2:
    e2, rho2, w2, seg_rho2 = A['f2']
    st.caption('Je Episode: Σ K a.p. über die behandelten Segmente gegen ΔLL (Deckplatten, degen). Negatives ρ hieße: '
               'mehr Skoliosekorrektur, mehr Lordoseverlust. ' + CI_NOTE)
    if {'K_ap_sum', 'd_LL'} <= set(e2.columns):
        st.plotly_chart(px.scatter(e2, x='K_ap_sum', y='d_LL', color='group', hover_data=['ep_date'],
                                   labels={'K_ap_sum': 'Σ K a.p. (°)', 'd_LL': 'Δ LL (°)'}), width='stretch')
    table(rho2)
    st.subheader('Je Segment: K a.p. bzw. Slip-Reposition gegen Δ seitlich')
    table(seg_rho2)

with t3:
    cg, xt, s3 = A['f3']
    st.caption('Einbringseite (a.p., Implantat-Detektion) gegen Öffnung des Segments präop. (a.p.-Segmentwinkel); '
               'neutral = |α_prä| < MDC95. ' + CI_NOTE)
    if len(cg):
        st.dataframe(xt, width='content')
        table(s3)
        st.plotly_chart(px.strip(cg, x='approach', y='k_ap', hover_data=['patient_id', 'seg'],
                                 labels={'k_ap': 'K a.p. am Cage-Segment (°)'}), width='stretch')
        table(cg[['patient_id', 'ep_date', 'seg', 'side', 'pre_ap', 'k_ap', 'd_lat', 'k_slip', 'open_side', 'approach']],
              ep_date=st.column_config.DateColumn('OP'), pre_ap=st.column_config.NumberColumn(format='%+.1f'),
              k_ap=st.column_config.NumberColumn(format='%+.1f'), d_lat=st.column_config.NumberColumn(format='%+.1f'),
              k_slip=st.column_config.NumberColumn(format='%+.1f'))
    else:
        st.info('Keine Cages mit a.p.-Seite.')

with t4:
    e4, s4, r4 = A['f4']
    st.caption('kurz = 1–2 neu fusionierte Segmente, lang = ≥ 3 oder Becken. K_mean = mittlere Korrektur a.p. je '
               'fusioniertem Segment, rel_mean = K/|prä|, pre_abs = mittlere |prä| (Confounding). ' + CI_NOTE)
    if len(e4):
        st.plotly_chart(px.scatter(e4, x='fused_n', y='K_mean', color='Länge', size='pre_abs',
                                   hover_data=['patient_id', 'ep_date'],
                                   labels={'fused_n': 'neu fusionierte Segmente', 'K_mean': 'mittlere K a.p. (°)'}),
                        width='stretch')
    st.markdown(f"Spearman Segmente ~ K_mean: ρ = {r4['rho']:.2f} (n = {r4['n']}, CI {r4['CI_lo']:.2f}; "
                f"{r4['CI_hi']:.2f})")
    table(s4)

with t5:
    reg, contrib = A['f5a']
    rate, new = A['f5b']
    st.subheader('F5a – Δ regionaler/globaler Parameter')
    st.caption('cCobb_max, L4/L5-Tilt (a.p.); LL, SS, PT, L1PA, T1PA, SVA (seitlich) aus degen_pipeline. ' + CI_NOTE)
    table(reg)
    st.subheader('Beitrag der Segmente zur lumbalen Gesamtänderung')
    st.caption('Δ_Segment · sign(ΣΔ) / Σ|Δ| je Episode; > 0 = trägt in Richtung der Gesamtänderung bei.')
    if len(contrib):
        st.plotly_chart(px.bar(contrib, x='seg', y='Median', color='group', barmode='group', facet_row='plane',
                               category_orders={'seg': sa.LUMBAR}), width='stretch')
    table(contrib)
    st.subheader('F5b – Nachbarsegmente kurzstreckiger Fusionen (Rate pro Jahr)')
    st.caption('Fusion mit 1–2 Segmenten ohne Becken, ab 3 Monaten Verlauf; Referenz = alle lumbalen Segmente '
               'konservativer Episoden.')
    table(rate)
    st.markdown(f'Neu aufgetretene Listhesen (|Slip| prä < {sa.LISTHESIS:.0f} %, post ≥ {sa.LISTHESIS:.0f} %)')
    table(new)

with t6:
    b6, ev6 = A['f6']
    st.caption('Korrekturverlust seit early = (α(t) − α_early) · sign(α_prä − α_early); > 0 = zurück in Richtung '
               'Ausgangswert. Operative Episoden. ' + CI_NOTE)
    if len(b6):
        plane6 = st.radio('Größe', list(sa.PLANES), format_func=sa.PLANES.get, horizontal=True, key='f6p')
        g6 = b6[b6['plane'] == plane6].sort_values(['cls', 'Zeit'])
        st.plotly_chart(px.line(g6.assign(Zeit=g6['Zeit'].astype(str)), x='Zeit', y='Median', color='cls', markers=True,
                                category_orders={'Zeit': sa.BIN_LABELS}, hover_data=['n', 'Patienten']),
                        width='stretch')
        table(g6)
    st.subheader(f'Ereignisse: erste Visite mit Verlust > MDC95 ({len(ev6)})')
    table(ev6, ep_date=st.column_config.DateColumn('OP'))
