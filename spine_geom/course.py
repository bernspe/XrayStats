"""Winkelverlauf relativ zum Indexdatum (Patient.index_date): Baseline = letzte Visite VOR dem Indexdatum,
Verlauf = Veränderung der Visiten AM/NACH dem Indexdatum. Rein tabellarisch, ohne Streamlit."""
import numpy as np
import pandas as pd

NA = '(nicht erfasst)'
# ClinicalEntryData.data-Felder, nach denen gefiltert werden kann
FILTERS = {'therapy': 'Therapie', 'therapy_specification': 'Therapie-Spezifikation',
           'fused_segments': 'Fusionierte Segmente', 'fused_to_pelvis': 'Fusion bis Becken',
           'complication': 'Komplikation', 'long_term_complication': 'Langzeit-Komplikation'}
INDEX_ONLY = ('therapy', 'therapy_specification', 'fused_segments', 'fused_to_pelvis')   # Index-Ereignis
# complication/long_term_complication: Vereinigung über alle Einträge des Patienten (treten nach der OP auf)
BINS = [0, 3, 6, 12, 24, 36, 60, np.inf]            # Monate nach Indexdatum
BIN_LABELS = ['0–3', '3–6', '6–12', '12–24', '24–36', '36–60', '>60']


def values_of(v) -> set[str]:
    """Feldwert → Menge von Strings; leer/fehlend → {NA}."""
    if isinstance(v, (list, tuple, set, np.ndarray)):
        s = {str(x) for x in v if str(x).strip()}
    elif v is None or (isinstance(v, float) and np.isnan(v)) or v == '':
        s = set()
    elif isinstance(v, (bool, np.bool_)):
        s = {'ja' if v else 'nein'}
    elif isinstance(v, (int, float, np.integer, np.floating)):
        s = {str(int(v))}
    else:
        s = {str(v).strip()} - {''}
        if s and next(iter(s)).lstrip('-').isdigit():
            s = {str(int(next(iter(s))))}
        elif s and next(iter(s)) in ('True', 'False'):
            s = {'ja' if next(iter(s)) == 'True' else 'nein'}
    return s or {NA}


def index_entry(g: pd.DataFrame, idx):
    """Index-Ereignis: Eintrag am Indexdatum, sonst der nächste danach, sonst der erste (g nach rd sortiert)."""
    if idx is not None and not pd.isna(idx):
        after = g[g['rd'] >= pd.Timestamp(idx)]
        if len(after):
            return after.iloc[0]
    return g.iloc[0]


def patient_clinical(entries: pd.DataFrame, uuid_to_pid: dict, index_dates: dict | None = None) -> pd.DataFrame:
    """Je Patient (patient_id) ein Set pro Filterfeld. INDEX_ONLY-Felder stammen aus index_entry(). Fehlende Spalten (z. B. long_term_complication, noch nicht
    in der API) → alles {NA}."""
    e = entries.assign(pid=entries['patient'].map(uuid_to_pid)).dropna(subset=['pid'])
    e = e.assign(rd=pd.to_datetime(e['referring_date'], errors='coerce')).sort_values('rd')
    def field(g, f, pid):
        if f not in g:
            return {NA}
        if f in INDEX_ONLY:
            return values_of(index_entry(g, (index_dates or {}).get(pid))[f] if f in g else None)
        u = set().union(*(values_of(x) for x in g[f]))
        return u - {NA} or {NA}

    rows = {pid: {f: field(g, f, pid) for f in FILTERS} for pid, g in e.groupby('pid')}
    return pd.DataFrame.from_dict(rows, orient='index')


def apply_filters(clin: pd.DataFrame, selected: dict) -> set:
    """Patienten, die je Feld mindestens einen gewählten Wert haben (leere Auswahl = kein Filter)."""
    keep = set(clin.index)
    for f, sel in selected.items():
        if sel:
            keep &= {p for p, s in clin[f].items() if s & set(sel)}
    return keep


def course(values: pd.DataFrame, index_dates: dict) -> pd.DataFrame:
    """values: patient_id, series, date, value → + t_months, delta.
    Baseline je (Patient, Serie) = Wert der letzten Visite mit date < index_date; Baseline selbst erscheint als Punkt
    (t=0, Δ=0) mit base=True. Ohne Indexdatum oder ohne Baseline-Wert entfällt die Serie."""
    out = []
    for (pid, s), g in values.groupby(['patient_id', 'series']):
        idx = index_dates.get(pid)
        if idx is None or pd.isna(idx):
            continue
        idx = pd.Timestamp(idx)
        pre, post = g[g['date'] < idx], g[g['date'] >= idx]
        if pre.empty or post.empty:
            continue
        base = pre.sort_values('date')['value'].iloc[-1]
        out.append(pd.DataFrame({'patient_id': pid, 'series': s, 'date': [idx], 'value': [base], 't_months': 0.0,
                                 'delta': 0.0, 'base': True}))
        out.append(post.assign(t_months=(post['date'] - idx).dt.days / 30.4375, delta=post['value'] - base, base=False))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=['patient_id', 'series', 'date', 'value', 't_months', 'delta', 'base'])


def binned_mean(c: pd.DataFrame) -> pd.DataFrame:
    """Kohortenmittel der Post-Visiten je Zeitfenster (Monate nach Indexdatum); n = Patienten."""
    p = c[~c['base']].assign(bin=lambda x: pd.cut(x['t_months'], BINS, labels=BIN_LABELS, right=False))
    per_pat = p.groupby(['series', 'bin', 'patient_id'], observed=True).agg(delta=('delta', 'mean'),
                                                                           t=('t_months', 'mean')).reset_index()
    return (per_pat.groupby(['series', 'bin'], observed=True)
            .agg(delta=('delta', 'mean'), sd=('delta', 'std'), t=('t', 'mean'), n=('patient_id', 'nunique'))
            .reset_index())


def _visit_means(raw: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """a: image_id, rater_id, series, value → Mittel je (Patient, Serie, Visite)."""
    img = raw[['id', 'patient_id', 'acquisition_date']].rename(columns={'id': 'image_id', 'acquisition_date': 'date'})
    a = a.merge(img.drop_duplicates('image_id'), on='image_id')
    a['date'] = pd.to_datetime(a['date']).dt.normalize()
    return a.groupby(['patient_id', 'series', 'date'], as_index=False)['value'].mean()


def ap_region_values(raw: pd.DataFrame, angles: pd.DataFrame, cobb_type: str) -> pd.DataFrame:
    """a.p.-Regionalwinkel U-LU, L-LU und LU = max(U-LU, L-LU) je Rater×Bild, dann Mittel je Visite.
    cobb_type: 'Cobb-regional' (gezeichnete Linien) oder 'Spline' (aus der Spline abgeleitet) – nie vermischt.
    angles = annotation_analysis.build_angles_df(raw) (Spalten image_id, rater_id, cobbType, region, angle)."""
    a = angles[(angles['cobbType'] == cobb_type) & angles['region'].isin(['U-LU', 'L-LU'])]
    a = a.dropna(subset=['angle']).groupby(['image_id', 'rater_id', 'region'])['angle'].max().unstack('region')
    a['LU'] = a.max(axis=1)
    return _visit_means(raw, a.stack().rename('value').reset_index().rename(columns={'region': 'series'}))


def ll_values(raw: pd.DataFrame, angles: pd.DataFrame, cobb_type: str) -> pd.DataFrame:
    """Lumbale Lordose (T12→S1, aus den Segmenten der Methode zusammengesetzt) je Rater×Bild, Mittel je Visite.
    angles = build_angles_df(raw, lateral_spans=(cobb_type,)); Zeilen mit region == 'LL'."""
    a = angles[(angles['cobbType'] == cobb_type) & (angles['region'] == 'LL')].dropna(subset=['angle'])
    return _visit_means(raw, a.rename(columns={'angle': 'value'}).assign(series='LL')[['image_id', 'rater_id', 'series', 'value']])
