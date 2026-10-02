import numpy as np
import pandas as pd

from degen_pipeline import calib_consistency


def _meas(rows):
    df = pd.DataFrame(rows, columns=['patient_id', 'view', 'date', 'image_id', 'wkd_px', 'lum_len_px'])
    return df.assign(posture='standing', qc_flags=[set() for _ in range(len(df))])


def test_deviant_scale_only_flagged_never_corrected():
    # 2PVD-Fall: echte Höhenänderung nach OP sieht aus wie ein Kalibrierfehler → nur Hinweis, WKD bleibt
    m = _meas([('P', 'ap', '2024-01', 1, 40.0, 200.0), ('P', 'ap', '2024-06', 2, 40.0, 200.0),
               ('P', 'ap', '2024-09', 3, 40.0, 250.0), ('P', 'lat', '2024-01', 4, 30.0, 150.0)])
    out = calib_consistency(m)
    assert list(m['wkd_px']) == [40.0, 40.0, 40.0, 30.0]
    assert 'calib_inconsistent' in m.at[2, 'qc_flags'] and not m.at[0, 'qc_flags'] and not m.at[3, 'qc_flags']
    assert out['n_calib_flagged'] == 1


def test_scale_sd_only_from_same_day_pairs():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(150):
        for day in range(2):                                      # zwei Visiten mit echter Höhenänderung (+10 %) …
            for k in range(2):                                    # … und je 2 Aufnahmen am selben Tag
                rows.append((f'P{p}', 'lat', f'd{day}', p * 10 + day * 2 + k,
                             30.0 * (1 + rng.normal(0, 0.04)), 150.0 * (1.1 if day else 1.0)))
    out = calib_consistency(_meas(rows))
    assert 0.03 < out['scale_sd']['lat'] < 0.05                   # ≈ 4 % – die echte Änderung zählt nicht mit
    assert out['scale_pairs']['lat'] == 300 and np.isnan(out['scale_sd']['ap'])
