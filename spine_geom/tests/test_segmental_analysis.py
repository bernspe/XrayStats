import numpy as np
import pandas as pd

import segmental_analysis as sa


def _ch(n_pat):
    rows = []
    for p in range(n_pat):
        for seg, d in (('L4/L5', 2.0 + p), ('L5/S1', -1.0)):
            rows.append({'patient': f'p{p}', 'plane': 'ap', 'group': 'Fusion', 'seg': seg, 'd': d})
    return pd.DataFrame(rows)


def test_ci_only_from_five_patients():
    small = sa.summarize(_ch(4), ['plane', 'seg'])
    assert small['CI_lo'].isna().all() and small['n'].tolist() == [4, 4]
    big = sa.summarize(_ch(6), ['plane', 'seg'])
    r = big.set_index('seg').loc['L4/L5']
    assert r['Median'] == 4.5 and r['CI_lo'] <= r['Median'] <= r['CI_hi'] and r['Patienten'] == 6


def test_mdc_share_and_segment_order():
    ch = _ch(3).assign(seg=lambda d: d['seg'].map({'L4/L5': 'L5/S1', 'L5/S1': 'L4/L5'}))
    mdc = pd.DataFrame({'plane': 'ap', 'seg': ['L4/L5', 'L5/S1'], 'mdc95': [0.5, 3.5]})
    t = sa.summarize(ch, ['plane', 'seg'], 'd', mdc)
    assert t['seg'].tolist() == ['L4/L5', 'L5/S1']                       # kranial → kaudal sortiert
    assert abs(t.set_index('seg').loc['L5/S1', '> MDC95 %'] - 100 / 3) < 1e-9   # d = 2, 3, 4 gegen MDC 3,5


def test_f6_time_bins_ordered():
    ch = pd.DataFrame({'patient': 'p', 'ep_date': pd.Timestamp('2020-01-01'), 'patient_id': 'P', 'group': 'Fusion',
                       'plane': 'ap', 'seg': 'L4/L5', 'cls': 'fusioniert', 'is_early': False,
                       'months': [40.0, 4.0, 1.0], 'loss': [1.0, 2.0, 3.0]})
    mdc = pd.DataFrame({'plane': ['ap'], 'seg': ['L4/L5'], 'mdc95': [2.5]})
    b, ev = sa.f6(ch, mdc)
    assert b['Zeit'].astype(str).tolist() == ['0–3', '3–6', '36–60']
    assert ev['months'].tolist() == [1.0]                                  # erstes Überschreiten der MDC
