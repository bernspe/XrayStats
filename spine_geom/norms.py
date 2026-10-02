"""Normabweichung (Spec Phase 4): Populations-z-Werte, PI-abgeleitete Ziele, Zielbereiche,
Klassifikationen – jeweils mit Unsicherheit und Quellenangabe. Werte stehen in norms.yaml.

Referenzwahl (Nutzerentscheidung 2026-09-29): Standard MEANS-Gesamtnorm; ein zu Alter und Geschlecht
passendes Stratum (Yukawa) hat Vorrang, dann ein reines Altersstratum (Wang ≥ 50 J., Gharbi), dann die
Gesamtnorm. Bei Gleichstand: mit SD vor ohne SD, dann Reihenfolge in norms.yaml.
"""
from __future__ import annotations

import math
import os

import yaml

NORMS_PATH = os.path.join(os.path.dirname(__file__), 'norms.yaml')
Z95 = 1.959964
ALIAS = {'SVA_wkd': 'SVA', 'C7_CSVL_wkd': 'C7_CSVL'}   # Parametername der Pipeline → Name in norms.yaml
OWN_UNIT = {'SVA': 'WKD', 'C7_CSVL': 'WKD'}             # unsere SVA ist in WKD → mm-Normen passen nicht; Rest Grad
THIN_DATA_AGE = 70                    # Spec §3.6: Datenlage über 70 J. dünn


def load_norms(path: str = NORMS_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def population_reference(param: str, age: float | None, sex: str | None, norms: dict) -> dict | None:
    """Beste Referenz (mean, sd, source, unit, stratum) für param oder None.
    Rang: Alters- + Geschlechtsstratum (3) > Altersstratum (2) > Gesamtnorm (0); bei Gleichstand
    mit SD vor ohne SD, dann der frühere Eintrag in norms.yaml (MEANS steht vorn).
    Altersgrenzen halboffen: [lo, hi] heißt lo ≤ Alter < hi."""
    cands = []
    for rank, e in enumerate(norms['population'].get(param, [])):
        base = dict(source=e['source'], unit=e['unit'])
        for s in e.get('strata') or []:
            lo, hi = s.get('age', (None, None))
            if ('sex' in s and s['sex'] != sex) or age is None or lo is None or not lo <= age < hi:
                continue
            cands.append(((3 if 'sex' in s else 2, s.get('sd') is not None, -rank),
                          dict(base, mean=s['mean'], sd=s.get('sd'), n=s.get('n'),
                               stratum=f"{s.get('sex', 'alle')} {lo}–{hi - 1 if hi < 120 else '…'} J.")))
        if e.get('mean') is not None:
            cands.append(((0, e.get('sd') is not None, -rank), dict(base, mean=e['mean'], sd=e.get('sd'),
                                                                   stratum='gesamt')))
    return max(cands, key=lambda c: c[0])[1] if cands else None


def _classify(value: float, sd: float | None, cls: dict) -> tuple[str, bool]:
    """Label und Flag 'unsicher' (Schwelle innerhalb ±1,96·SD)."""
    label = cls['labels'][sum(value >= t for t in cls['thresholds'])]
    uncertain = sd is not None and any(abs(value - t) < Z95 * sd for t in cls['thresholds'])
    return label, uncertain


def deviation_state(values: dict, pi: tuple[float, float] | None, age: float | None, sex: str | None,
                    norms: dict | None = None) -> list[dict]:
    """Abweichungen einer Visite. values: param → (Wert, SD). pi: Patienten-PI (Schätzer, SE).

    Zeilen: art ('Population' | 'PI-Ziel' | 'Zielbereich' | 'Klassifikation'), param, wert, wert_sd,
    referenz, referenz_sd, abweichung, abweichung_sd, z, quelle, hinweis."""
    norms = norms or load_norms()
    v = {ALIAS.get(k, k): x for k, x in values.items()}
    if pi is not None:
        v['PI'] = pi                                        # Patientenkonstante statt Visitenwert
        if 'LL' in v:
            v['PI_LL'] = (pi[0] - v['LL'][0], math.hypot(pi[1], v['LL'][1] or 0))
    rows = []

    for param, (val, sd) in v.items():
        ref = population_reference(param, age, sex, norms)
        if ref is None:
            continue
        hints = []
        src = norms['sources'][ref['source']]
        comparable = ref['unit'] == OWN_UNIT.get(param, 'deg')
        if not comparable:
            hints.append(f"Einheit: eigene Messung {OWN_UNIT.get(param, 'deg')}, Norm {ref['unit']} – nicht vergleichbar")
        rng = src.get('age_range')
        if age is not None and rng and not rng[0] <= age < rng[1]:
            hints.append(f'Alter {age:.0f} J. außerhalb Normkohorte ({rng[0]}–{rng[1]})')
        if age is not None and age > THIN_DATA_AGE:
            hints.append('Normdaten > 70 J. dünn')
        dev = val - ref['mean'] if comparable else None
        rows.append(dict(art='Population', param=param, wert=val, wert_sd=sd, referenz=ref['mean'],
                         referenz_sd=ref['sd'], abweichung=dev, abweichung_sd=sd if comparable else None,
                         z=dev / ref['sd'] if comparable and ref['sd'] else None,
                         quelle=f"{ref['source']} ({ref['stratum']})", hinweis='; '.join(hints)))

    if pi is not None:
        for param, entries in norms['pi_targets'].items():
            if param not in v:
                continue
            val, sd = v[param]
            for e in entries:
                target = e['a'] * pi[0] + e['b']
                t_sd = abs(e['a']) * pi[1]
                dev = val - target
                dev_sd = math.hypot(sd or 0, t_sd)
                z = dev / math.hypot(dev_sd, e['resid_sd']) if e.get('resid_sd') else None
                rows.append(dict(art='PI-Ziel', param=param, wert=val, wert_sd=sd, referenz=target, referenz_sd=t_sd,
                                 abweichung=dev, abweichung_sd=dev_sd, z=z, quelle=e['source'], hinweis=e.get('note', '')))

    for param, r in norms['target_ranges'].items():
        if param not in v:
            continue
        val, sd = v[param]
        dev = val - min(max(val, r['low']), r['high'])       # 0 innerhalb des Bereichs
        rows.append(dict(art='Zielbereich', param=param, wert=val, wert_sd=sd, referenz=f"{r['low']} … {r['high']}",
                         referenz_sd=None, abweichung=dev, abweichung_sd=sd, z=None, quelle=r['source'],
                         hinweis=r.get('note', '')))

    ss_ideal = next((e for e in norms['pi_targets'].get('SS', [])), None)
    if pi is not None and ss_ideal:
        v['SS_ideal'] = (ss_ideal['a'] * pi[0] + ss_ideal['b'], abs(ss_ideal['a']) * pi[1])
    for name, cls in norms['classifications'].items():
        if cls['param'] not in v:
            continue
        val, sd = v[cls['param']]
        label, unsure = _classify(val, sd, cls)
        rows.append(dict(art='Klassifikation', param=name, wert=val, wert_sd=sd, referenz=label, referenz_sd=None,
                         abweichung=None, abweichung_sd=None, z=None, quelle=cls['source'],
                         hinweis=('Schwelle innerhalb ±1,96·SD – Klasse unsicher; ' if unsure else '') + cls.get('note', '')))
    return rows
