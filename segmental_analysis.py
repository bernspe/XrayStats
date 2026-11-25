"""Auswertung F1–F6 des segmentalen Therapieverlaufs (Konzept §4: docs/degen_skoliose_verlauf_segmental_bottom_up.md).

Reine Pandas-Funktionen auf den Caches der Pipeline (seg_changes, seg_episodes, seg_mdc), genutzt von
segmental_course_app.py und write_report(). Deskriptiv: Median/IQR, Cluster-Bootstrap-CI über Patienten erst ab
N_MIN_CI Patienten, Anteil |Δ| > MDC95. Machbarkeitsnachweis – Zahlen immer mit n lesen.

    python segmental_analysis.py        # schreibt reports/segmental_therapieeffekt.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import segmental_course_pipeline as sp
from spine_geom.course import BIN_LABELS, BINS
from spine_geom.segmental import _seg_key

N_MIN_CI = 5                     # darunter keine CI, nur Einzelwerte (Konzept §6, Schritt 4)
N_BOOT = 1000
LUMBAR = ['T12/L1', 'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
PLANES = {'ap': 'a.p. (°)', 'lat': 'seitlich (°)', 'slip': 'Slip (% WKD)'}
LISTHESIS = 10.0                 # |Slip| ≥ 10 % WKD = Listhese (Schwelle wie train_slip.ipynb)


def load() -> dict:
    d = {k: pd.read_pickle(p) for k, p in (('ch', sp.CHANGES), ('ep', sp.EPISODES), ('mdc', sp.MDC))}
    d['ep'] = d['ep'][d['ep']['status'] == 'auswertbar']
    return d


def boot_ci(df: pd.DataFrame, col: str, stat=np.median, seed: int = 0) -> tuple[float, float]:
    """95-%-CI per Cluster-Bootstrap über Patienten; NaN unter N_MIN_CI Patienten."""
    g = [x[col].dropna().to_numpy() for _, x in df.groupby('patient')]
    g = [x for x in g if len(x)]
    if len(g) < N_MIN_CI:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    v = [stat(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(N_BOOT)]
    return tuple(np.percentile(v, [2.5, 97.5]))


def summarize(df: pd.DataFrame, by: list[str], col: str = 'd', mdc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Je Gruppe: n (Werte), Patienten, Median, IQR, CI (ab N_MIN_CI Patienten), Anteil |col| > MDC95."""
    if mdc is not None:
        df = df.merge(mdc[['plane', 'seg', 'mdc95']], on=['plane', 'seg'], how='left')
    rows = []
    for key, g in df.dropna(subset=[col]).groupby(by, sort=False):
        lo, hi = boot_ci(g, col)
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r.update(n=len(g), Patienten=g['patient'].nunique(), Median=g[col].median(),
                 Q25=g[col].quantile(.25), Q75=g[col].quantile(.75), CI_lo=lo, CI_hi=hi)
        if 'mdc95' in g:
            r['> MDC95 %'] = 100 * (g[col].abs() > g['mdc95']).mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    if 'seg' in by and len(out):
        out = out.sort_values([c for c in by if c != 'seg'] + ['seg'], key=lambda s: s.map(_seg_key)
                              if s.name == 'seg' else s)
    return out


def _rho(x, y) -> dict:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return {'n': int(ok.sum()), 'rho': np.nan}
    return {'n': int(ok.sum()), 'rho': float(spearmanr(x[ok], y[ok])[0])}


def _rho_ci(df: pd.DataFrame, x: str, y: str, seed: int = 0) -> dict:
    """Spearman ρ mit Cluster-Bootstrap-CI über Patienten (ab N_MIN_CI Patienten)."""
    d = df.dropna(subset=[x, y])
    r = _rho(d[x].to_numpy(float), d[y].to_numpy(float))
    r['Patienten'] = d['patient'].nunique() if len(d) else 0
    r['CI_lo'] = r['CI_hi'] = np.nan
    if r['Patienten'] >= N_MIN_CI:
        pats = d['patient'].unique()
        rng = np.random.default_rng(seed)
        v = []
        for _ in range(N_BOOT):
            s = pd.concat([d[d['patient'] == p] for p in rng.choice(pats, len(pats))])
            v.append(_rho(s[x].to_numpy(float), s[y].to_numpy(float))['rho'])
        r['CI_lo'], r['CI_hi'] = np.nanpercentile(v, [2.5, 97.5])
    return r


def tp(ch: pd.DataFrame, when: str = 'late') -> pd.DataFrame:
    return ch[ch[f'is_{when}']]


# --- F1: Verfahren × Segmenthöhe ---------------------------------------------------------------------
def f1(ch, mdc, when='late') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Δ je Gruppe × Größe × Segment; je Segmentklasse Δ und K (Betragskorrektur – bei Δ heben sich links- und
    rechtskonvexe Segmente auf); Listhese-Anteil prä/post (Slip)."""
    c = tp(ch, when)
    c = c[c['plane'].isin(PLANES)]
    by_seg = summarize(c, ['plane', 'group', 'seg'], 'd', mdc)
    by_cls = pd.concat([summarize(c, ['plane', 'group', 'cls'], 'd', mdc).assign(Größe='Δ'),
                        summarize(c, ['plane', 'group', 'cls'], 'k').assign(Größe='K')], ignore_index=True)
    sl = c[(c['plane'] == 'slip') & c['d'].notna()]
    lis = sl.groupby(['group', 'seg']).agg(n=('d', 'size'), prae=('pre', lambda v: 100 * (v.abs() >= LISTHESIS).mean()),
                                           post=('value', lambda v: 100 * (v.abs() >= LISTHESIS).mean())).reset_index()
    return by_seg, by_cls, lis


# --- F2: Kopplung koronar ↔ sagittal ------------------------------------------------------------------
def f2(ch, when='late') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Je Episode: koronare Korrektur (Σ K a.p. über die behandelten Segmente, bei konservativ alle lumbalen;
    dazu K von cCobb_max) gegen ΔLL (degen, Deckplatten) und Σ Δ seitlich L1/L2–L5/S1. Je Segment: K a.p. und
    K Slip gegen Δ seitlich am selben Segment. Spearman ρ je Gruppe."""
    c = tp(ch, when)
    treated = c['cls'].isin(['fusioniert', 'dekomprimiert', 'alle'])
    seg = c[c['seg'].isin(LUMBAR) & treated]
    ep = seg[seg['plane'] == 'ap'].groupby(['patient', 'ep_date', 'group'])['k'].sum(min_count=1).rename('K_ap_sum')
    ll_seg = seg[(seg['plane'] == 'lat') & (seg['seg'] != 'T12/L1')].groupby(['patient', 'ep_date', 'group'])['d'] \
        .sum(min_count=1).rename('dLL_seg')
    reg = c[c['plane'] == 'regional'].pivot_table(index=['patient', 'ep_date', 'group'], columns='seg',
                                                  values=['k', 'd'])
    reg.columns = [f'{a}_{b}' for a, b in reg.columns]
    e = pd.concat([ep, ll_seg, reg.get('k_cCobb_max'), reg.get('d_LL')], axis=1).reset_index()
    rho = pd.DataFrame([{'group': g, 'x': x, 'y': y, **_rho_ci(d, x, y)}
                        for g, d in e.groupby('group') for x, y in (('K_ap_sum', 'd_LL'), ('K_ap_sum', 'dLL_seg'),
                                                                     ('k_cCobb_max', 'd_LL'))
                        if x in d and y in d])
    w = c[c['seg'].isin(LUMBAR)].pivot_table(index=['patient', 'ep_date', 'group', 'seg', 'cls'], columns='plane',
                                              values=['k', 'd']).reset_index()
    w.columns = ['_'.join(t).strip('_') for t in w.columns]
    seg_rho = pd.DataFrame([{'group': g, 'x': x, 'y': 'd_lat', **_rho_ci(d, x, 'd_lat')}
                            for g, d in w.groupby('group') for x in ('k_ap', 'k_slip') if x in d and 'd_lat' in d])
    return e, rho, w, seg_rho


# --- F3: Cage-Seite ↔ Öffnungsrichtung ----------------------------------------------------------------
def f3(ch, ep, mdc, when='late') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Je Cage mit a.p.-Seite: Öffnung des Segments präop. (+ = links) → konvex (Einbringseite = offene Seite),
    konkav, neutral (|α_prä| < MDC95). K a.p. und Δ seitlich am Cage-Segment je Zugang."""
    c = tp(ch, when)
    rows = []
    for _, e in ep[ep['group'] == 'Fusion'].iterrows():
        for lv, side in e['cages'] if isinstance(e.get('cages'), list) else []:
            if side not in ('L', 'R'):
                continue
            r = {'patient': e['patient'], 'patient_id': e['patient_id'], 'ep_date': e['date'], 'seg': lv, 'side': side}
            for plane in ('ap', 'lat', 'slip'):
                x = c[(c['patient'] == e['patient']) & (c['ep_date'] == e['date']) & (c['plane'] == plane)
                      & (c['seg'] == lv)]
                r[f'pre_{plane}'] = x['pre'].iloc[0] if len(x) else np.nan
                r[f'k_{plane}'] = x['k'].iloc[0] if len(x) else np.nan
                r[f'd_{plane}'] = x['d'].iloc[0] if len(x) else np.nan
            rows.append(r)
    cg = pd.DataFrame(rows)
    if cg.empty:
        return cg, cg, cg
    m = mdc[mdc['plane'] == 'ap'].set_index('seg')['mdc95']
    cg['open_side'] = np.where(cg['pre_ap'] > 0, 'L', np.where(cg['pre_ap'] < 0, 'R', None))
    cg['approach'] = np.where(cg['pre_ap'].isna(), 'keine a.p.-Baseline',
                              np.where(cg['pre_ap'].abs() < cg['seg'].map(m), 'neutral',
                                       np.where(cg['side'] == cg['open_side'], 'konvex', 'konkav')))
    xt = pd.crosstab(cg['side'].map({'L': 'von links', 'R': 'von rechts'}),
                     cg['open_side'].map({'L': 'offen links', 'R': 'offen rechts'}).fillna('–'), margins=True)
    summ = pd.concat([summarize(cg.assign(plane=p), ['approach'], f'{v}_{p}').assign(Größe=f'{v}_{p}')
                      for p, v in (('ap', 'k'), ('lat', 'd'), ('slip', 'k'))], ignore_index=True)
    return cg, xt, summ


# --- F4: lang- vs. kurzstreckige Fusion ---------------------------------------------------------------
def f4(ch, when='late') -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fusion-Episoden: kurz = 1–2 neu fusionierte Segmente, lang = ≥ 3 oder Becken. Je Episode mittlere K a.p.
    über die fusionierten Segmente, relative Korrektur K/|prä| und mittlere |prä| (Confounding: lange Fusionen
    haben größere Ausgangskurven)."""
    c = tp(ch, when)
    f = c[(c['group'] == 'Fusion') & (c['cls'] == 'fusioniert') & (c['plane'] == 'ap')].copy()
    f['rel'] = f['k'] / f['pre'].abs().where(f['pre'].abs() > 1)
    e = f.groupby(['patient', 'patient_id', 'ep_date']).agg(fused_n=('fused_n', 'first'), pelvis=('pelvis', 'first'),
                                                         K_mean=('k', 'mean'), rel_mean=('rel', 'mean'),
                                                         pre_abs=('pre', lambda v: v.abs().mean())).reset_index()
    e['Länge'] = np.where((e['fused_n'] >= 3) | e['pelvis'], 'lang', 'kurz')
    summ = pd.concat([summarize(e, ['Länge'], col).assign(Größe=col) for col in ('K_mean', 'rel_mean', 'pre_abs')],
                     ignore_index=True)
    return e, summ, _rho_ci(e, 'fused_n', 'K_mean')


# --- F5: segmental → regional → global; Nachbarsegmente ---------------------------------------------------
def f5a(ch, when='late') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Δ der regionalen/globalen Parameter je Gruppe und Beitrag der Segmente zur lumbalen Gesamtänderung:
    Δ_Segment · sign(ΣΔ) / Σ|Δ| je Episode und Ebene (Σ über T12/L1–L5/S1); > 0 = trägt in Richtung der
    Gesamtänderung bei, die Anteile eines Segments mit gleicher Richtung summieren sich zu 1."""
    c = tp(ch, when)
    reg = summarize(c[c['plane'] == 'regional'], ['group', 'seg'], 'd')
    s = c[c['plane'].isin(['ap', 'lat']) & c['seg'].isin(LUMBAR)].dropna(subset=['d']).copy()
    grp = s.groupby(['patient', 'ep_date', 'plane'])['d']
    s['share'] = s['d'] * np.sign(grp.transform('sum')) / grp.transform(lambda v: v.abs().sum())
    contrib = summarize(s, ['plane', 'group', 'seg'], 'share')
    return reg, contrib


def f5b(ch, when='late') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nachbarsegmente kurzstreckiger Fusionen (1–2 Segmente, ohne Becken): Änderungsrate Δ pro Jahr in UIV+1,
    UIV+2, LIV−1, fern gegen die Rate aller Segmente bei konservativen Episoden (Spontanverlauf). Dazu neu
    aufgetretene Listhesen (|Slip| prä < 10 %, post ≥ 10 %) je Klasse."""
    c = tp(ch, when)
    c = c[c['plane'].isin(PLANES) & (c['months'] >= 3)].copy()            # Rate erst ab 3 Monaten sinnvoll
    c['rate'] = c['d'] / (c['months'] / 12)
    short = (c['group'] == 'Fusion') & (c['fused_n'] <= 2) & ~c['pelvis']
    ref = (c['group'] == 'konservativ') & c['seg'].isin(LUMBAR)
    sel = pd.concat([c[short & c['cls'].isin(['UIV+1', 'UIV+2', 'LIV−1', 'fern'])],
                     c[ref].assign(cls='konservativ (Referenz)')])
    rate = summarize(sel, ['plane', 'cls'], 'rate')
    sl = sel[sel['plane'] == 'slip'].dropna(subset=['pre', 'value'])
    new = sl.assign(neu=(sl['pre'].abs() < LISTHESIS) & (sl['value'].abs() >= LISTHESIS)) \
        .groupby('cls').agg(Segmente=('neu', 'size'), neu=('neu', 'sum')).reset_index()
    return rate, new


# --- F6: Korrekturverlust -------------------------------------------------------------------------------
def f6(ch, mdc, entries: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Operative Episoden: Korrekturverlust seit early je Zeitklasse × Größe × Segmentklasse; Ereignis = erste
    Visite mit loss > MDC95 (je Episode × Größe × Segment), mit Langzeitkomplikationen des Patienten."""
    c = ch[ch['group'].isin(['Fusion', 'Dekompression']) & ch['plane'].isin(PLANES) & ~ch['is_early']] \
        .dropna(subset=['loss']).merge(mdc[['plane', 'seg', 'mdc95']], on=['plane', 'seg'], how='left')
    c['Zeit'] = pd.cut(c['months'], BINS, labels=BIN_LABELS, right=False)
    by_bin = summarize(c.drop(columns='mdc95'), ['plane', 'Zeit', 'cls'], 'loss')
    share = c.assign(v=c['loss'] > c['mdc95']).groupby(['plane', 'Zeit', 'cls'], observed=True)['v'].mean() * 100
    if len(by_bin):
        by_bin['Verlust > MDC95 %'] = [share.get(tuple(k)) for k in by_bin[['plane', 'Zeit', 'cls']].to_numpy()]
    if len(by_bin):
        by_bin['Zeit'] = pd.Categorical(by_bin['Zeit'], BIN_LABELS, ordered=True)
        by_bin = by_bin.sort_values(['plane', 'Zeit', 'cls'])
    ev = c[c['loss'] > c['mdc95']].sort_values('months') \
        .groupby(['patient', 'ep_date', 'plane', 'seg'], as_index=False).first()
    ev = ev[['patient', 'patient_id', 'ep_date', 'group', 'plane', 'seg', 'cls', 'months', 'loss', 'mdc95']]
    if entries is not None and len(ev):
        ltc = entries.groupby('patient')['long_term_complication'].agg(
            lambda v: ', '.join(sorted({str(x).strip() for x in v if isinstance(x, str) and x.strip()
                                        and x.strip().lower() not in ('keine', 'nan')})))
        ev['Langzeitkomplikation'] = ev['patient'].map(ltc)
    return by_bin, ev


# --- Bericht -------------------------------------------------------------------------------------------
def _md(df: pd.DataFrame, n: int = 60) -> str:
    return df.head(n).to_markdown(index=False, floatfmt='.1f') if len(df) else '_keine Daten_'


def write_report(path: str = 'reports/segmental_therapieeffekt.md') -> str:
    d = load()
    ch, ep, mdc = d['ch'], d['ep'], d['mdc']
    entries = pd.read_pickle(f'{sp.dp.DATA_DIR}/degen_entries.pkl')
    by_seg, by_cls, lis = f1(ch, mdc)
    e2, rho2, _, seg_rho2 = f2(ch)
    cg, xt, s3 = f3(ch, ep, mdc)
    e4, s4, r4 = f4(ch)
    reg, contrib = f5a(ch)
    rate, new = f5b(ch)
    b6, ev6 = f6(ch, mdc, entries)
    n_ep = ep.groupby('group').size().to_dict()
    txt = f"""# Segmentaler Therapieverlauf – Machbarkeitsnachweis (vorläufig)

Erzeugt mit `python segmental_analysis.py` aus den Caches der Pipeline (`{sp.PIPELINE_VERSION}`).
Konzept: `docs/degen_skoliose_verlauf_segmental_bottom_up.md`. **Alle Zahlen sind vorläufig und dienen dem Nachweis,
dass jede Frage aus den Daten beantwortbar ist** – nicht der Antwort selbst.

- Kohorte: validierte Patienten; auswertbare Episoden je Gruppe: {n_ep}.
- Zeitpunkt: `late` (letzte Visite vor dem nächsten Ereignis). Δ = post − prä, K = |prä| − |post| (> 0 = Korrektur,
  beim Slip Reposition). Einheiten: a.p./seitlich in °, Slip in % WKD.
- Median, IQR (Q25–Q75), CI = 95-%-Cluster-Bootstrap über Patienten, **erst ab {N_MIN_CI} Patienten**, darunter nur
  Einzelwerte. `> MDC95 %` = Anteil der Werte über dem Messfehler (Rater-Streuung; Slip konservativ 15,6 % WKD).
- n = Anzahl Werte (Episode × Segment), Patienten = Anzahl Patienten.

## F1 – Welche Veränderungen erzielte welches Verfahren, je Segmenthöhe?

### Δ je Gruppe × Segment
{_md(by_seg, 120)}

### Δ und K je Segmentklasse (K = |prä| − |post|)
{_md(by_cls)}

### Listhese (|Slip| ≥ {LISTHESIS:.0f} % WKD) prä vs. post, Anteil in %
{_md(lis)}

## F2 – Ging eine Skoliosekorrektur zu Lasten der Lordose?

Je Episode: `K_ap_sum` = Σ K a.p. über die behandelten Segmente (konservativ: alle lumbalen), `k_cCobb_max` = Korrektur
der Hauptkurve, `d_LL` = Δ Lordose L1–S1 (Deckplatten, degen), `dLL_seg` = Σ Δ seitlich L1/L2–L5/S1.
Negatives ρ zwischen Korrektur und ΔLL hieße: mehr Skoliosekorrektur, mehr Lordoseverlust.

{_md(rho2)}

Je Segment (gleiches Segment): K a.p. bzw. Reposition des Slips gegen Δ seitlich.

{_md(seg_rho2)}

## F3 – Hängt die Cage-Einbringseite mit der Öffnungsrichtung zusammen?

Cages mit a.p.-Seite: {len(cg)}. Kreuztabelle Einbringseite × Öffnung des Segments präop.:

{xt.to_markdown() if len(xt) else '_keine Daten_'}

Korrektur am Cage-Segment je Zugang (konvex = von der offenen Seite):

{_md(s3)}

## F4 – Größere segmentale Korrektur bei langstreckiger Fusion?

Kurz = 1–2 neu fusionierte Segmente, lang = ≥ 3 oder Becken. `K_mean` = mittlere K a.p. je fusioniertem Segment,
`rel_mean` = K/|prä|, `pre_abs` = mittlere |prä| (Confounding). Spearman Anzahl Segmente ~ K_mean:
ρ = {r4['rho']:.2f} (n = {r4['n']}, CI {r4['CI_lo']:.2f}; {r4['CI_hi']:.2f}).

{_md(s4)}

## F5a – Segmental → regional → global

Δ der regionalen/globalen Parameter (degen_pipeline, Status ok):

{_md(reg, 80)}

Beitrag der Segmente zur lumbalen Gesamtänderung (Δ_Segment · sign(ΣΔ) / Σ|Δ|, > 0 = in Richtung der
Gesamtänderung; Median je Gruppe):

{_md(contrib, 80)}

## F5b – Nachbarsegmente kurzstreckiger Fusionen

Änderungsrate pro Jahr (ab 3 Monaten Verlauf), Referenz = alle lumbalen Segmente konservativer Episoden:

{_md(rate)}

Neu aufgetretene Listhesen (|Slip| prä < {LISTHESIS:.0f} %, post ≥ {LISTHESIS:.0f} %):

{_md(new)}

## F6 – Ging die Korrektur wieder verloren?

Korrekturverlust seit `early` (> 0 = zurück in Richtung Ausgangswert) je Zeitklasse (Monate nach Ereignis):

{_md(b6, 80)}

Ereignisse (erste Visite mit Verlust > MDC95, je Episode × Größe × Segment): {len(ev6)}

{_md(ev6, 80)}

## Grenzen

Kleine Fallzahl (Machbarkeitsnachweis); Indikations-Confounding (konservativ = Spontanverlauf, keine Kontrollgruppe);
MDC aus Rater-Streuung unterschätzt den Messfehler zwischen Visiten; Slip-Modell schrumpft große Slips; kein Slip auf
WS-lat; Fusionsstrecke ±1 an UIV/LIV unsicher (siehe Konzept §8).
"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write(txt)
    return path


if __name__ == '__main__':
    print(write_report())
