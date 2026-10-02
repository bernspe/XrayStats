"""Synthetische Geometrie mit analytisch bekannten Werten (Spec §0.4, Phase 1 Abnahme)."""
import numpy as np
import pytest

from spine_geom import SPINE_LABELS
from spine_geom.frames import (calib_level_factors, facing_lateral, sacral_frame, to_frame,
                               to_patient, wkd_px)
from spine_geom.landmarks import extract
from spine_geom.sagittal import REQUIRES, missing_reason, available, sagittal_params, vpa

SP = 40.0  # Centroid-Abstand in px


def _rot(v, deg):
    a = np.radians(deg)
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])


def synth(pt=12.0, ss=38.0, tau_l1=-15.0, l1pa=6.0, facing=1, top='T1', with_cobb=True, with_hip=True,
          s1_extent=(20.0, 20.0), tau_l4=10.0):
    """xmd in Bildpixeln aus Patientengeometrie mit bekanntem PT, SS, τ_L1, L1PA."""
    H = np.array([0.0, 0.0])
    M = H + 150 * np.array([-np.sin(np.radians(pt)), np.cos(np.radians(pt))])
    e = np.array([np.cos(np.radians(ss)), -np.sin(np.radians(ss))])
    labels = SPINE_LABELS[SPINE_LABELS.index(top):]
    cent = {}
    for k, lab in enumerate(reversed(labels)):          # S1 unten, dann aufwärts
        cent[lab] = M + np.array([5.0 * np.sin(k / 3) if k > 1 else 0.0, -12.0 + SP * k])  # L5–S1-Sehne durch M
    # L1 exakt auf gewünschten L1PA drehen (Abstand zu H beibehalten)
    u = M - H
    cent['L1'] = H + _rot(u, -l1pa) / np.linalg.norm(u) * np.linalg.norm(cent['L1'] - H)
    el1 = np.array([np.cos(np.radians(tau_l1)), -np.sin(np.radians(tau_l1))])
    l1c = cent['L1'] + np.array([0.0, 12.0])
    el4 = np.array([np.cos(np.radians(tau_l4)), -np.sin(np.radians(tau_l4))])
    l4c = cent['L4'] + np.array([0.0, 12.0])

    img = lambda p: to_patient(np.asarray(p), facing) + np.array([400.0, 800.0]) * [1, 1]  # noqa: E731
    line = lambda a, b, mode=None: {'x1': img(a)[0], 'y1': img(a)[1], 'x2': img(b)[0], 'y2': img(b)[1],  # noqa: E731
                                    **({'mode': mode} if mode else {})}
    # L5-Unterkante (Distraktor) und S1-/L1-Deckplatte als Cobb-Linien
    cobb = [line(cent['L5'] + [-18, -14], cent['L5'] + [18, -14], 'Cobb-segmental'),
            line(M + s1_extent[0] * e, M - s1_extent[1] * e, 'Cobb-regional'),
            line(l1c - 20 * el1, l1c + 20 * el1, 'Cobb-regional'),
            line(l4c - 20 * el4, l4c + 20 * el4, 'Cobb-segmental')] if with_cobb else []
    xmd = {'stateObject': {
        'vertebra_range': [SPINE_LABELS.index(top), 25],
        'nodes': [{'x': img(cent[lab])[0], 'y': img(cent[lab])[1]} for lab in labels],
        'cobbLines': cobb,
        'hipLine': line(H + [4, 3], H - [4, 3]) if with_hip else None,
        'calibrationLine': line(cent['L4'] - [17.5, 0], cent['L4'] + [17.5, 0]),
    }}
    return xmd, dict(PT=pt, SS=ss, PI=pt + ss, LL=ss - tau_l1, L1PA=l1pa,
                     LDI=100 * (ss - tau_l4) / (ss - tau_l1))


def _params(xmd):
    lm = extract(xmd)
    facing, flags = facing_lateral(lm.s1_endplate, lm.centroids)
    return lm, facing, flags, sagittal_params(lm, facing, lm.calib_px)


@pytest.mark.parametrize('pt,ss', [(12, 38), (-5, 30), (30, 25), (0, 45)])
@pytest.mark.parametrize('facing', [1, -1])
def test_pi_equals_pt_plus_ss_and_known_values(pt, ss, facing):
    xmd, exp = synth(pt=pt, ss=ss, facing=facing)
    lm, f, flags, p = _params(xmd)
    assert f == facing                          # S1-Deckplatte entscheidet, Lordose-Cue ist hier synthetisch unscharf
    assert lm.s1_source == 'cobb' and lm.l1_source == 'cobb'
    assert abs(p['PI'] - (p['PT'] + p['SS'])) < 0.01
    for k in ('PI', 'PT', 'SS', 'LL', 'L1PA', 'LDI'):
        assert abs(p[k] - exp[k]) < 0.01, k


def test_mirror_invariance():
    a = _params(synth(facing=1)[0])[3]
    b = _params(synth(facing=-1)[0])[3]
    assert a.keys() == b.keys()
    for k in a:
        assert abs(a[k] - b[k]) < 1e-6, k


def test_vpa_sign():
    H, M = np.array([0.0, 0.0]), np.array([-20.0, 150.0])
    assert abs(vpa(M * 2, M, H)) < 1e-9                       # auf der Linie H→M
    assert vpa(M * 2 + [10, 0], M, H) > 0                     # anterior
    assert vpa(M * 2 - [10, 0], M, H) < 0                     # posterior


def test_tangent_fallback_and_requirements():
    xmd, _ = synth(with_cobb=False, top='T10')
    lm, f, flags, p = _params(xmd)
    assert lm.s1_source == 'tangent' and 'endplate_from_tangent' in lm.flags
    assert 'PI' in p and 'T4PA' not in p and 'T1PA' not in p
    have = available(lm, lm.calib_px)
    assert missing_reason('T4PA', have) == 'T4 nicht annotiert'
    assert missing_reason('SVA_wkd', have) == 'C7 nicht annotiert'


def test_no_hip():
    lm, f, flags, p = _params(synth(with_hip=False)[0])
    assert 'SS' in p and 'PI' not in p and 'L1PA' not in p
    assert missing_reason('PI', available(lm, lm.calib_px)) == 'keine Hüfte'
    assert set(REQUIRES) >= set(p)


def test_calibration_level_and_factors():
    lm = extract(synth()[0])
    assert lm.calib_level == 'L4' and abs(lm.calib_px - 35) < 1e-9
    rows = [(i, 'ap', 'L4', 100.0) for i in range(25)] + [(i, 'ap', 'L3', 97.0) for i in range(25)] \
        + [(i, 'ap', 'L2', 90.0) for i in range(5)] + [(i, 'ap', 'L2', 90.0) for i in range(5)]
    f = calib_level_factors(rows)
    assert abs(f[('ap', 'L3')]['factor'] - 0.97) < 1e-9 and f[('ap', 'L3')]['n'] == 25
    assert ('ap', 'L2') not in f                                  # < 20 Paare → ungültig
    assert f[('lat', 'L3')] == {'factor': 1.0, 'n': 0}            # Protokoll-Level ohne Daten
    assert abs(wkd_px(97.0, 'ap', 'L3', f) - 100.0) < 1e-9
    assert wkd_px(97.0, 'ap', 'L2', f) is None


def test_sacral_frame():
    A, P = np.array([10.0, 5.0]), np.array([-10.0, 15.0])
    M, R = sacral_frame(A, P)
    loc = to_frame([A, P, M + [0, 1]], M, R)
    assert np.allclose(loc[0], [np.linalg.norm(A - P) / 2, 0])
    assert np.allclose(loc[1], [-np.linalg.norm(A - P) / 2, 0])
    assert loc[2][1] > 0                                          # kranial bleibt kranial


@pytest.mark.parametrize('facing', [1, -1])
def test_orientation_from_lordosis_only(facing):
    # Lordose: L3 liegt anterior der Sehne L1–S1; keine Deckplatte vorhanden
    cent = {'L1': [0, 0], 'L3': [15, -80], 'S1': [0, -160]}
    img = {k: to_patient(v, facing) for k, v in cent.items()}
    f, flags = facing_lateral(None, img)
    assert f == facing and 'orientation_unknown' not in flags
    f, flags = facing_lateral(np.array([to_patient([20, -150], facing), to_patient([-20, -130], facing)]), img)
    assert f == facing and 'orientation_conflict' not in flags


def test_long_offcenter_cobb_line_gives_same_result():
    """Rater ziehen Cobb-Linien weit über die Deckplatte hinaus – nur die Richtung zählt, M ist der
    Schnittpunkt mit der L5–S1-Sehne."""
    ref = _params(synth()[0])[3]
    for extent in ((90.0, 10.0), (15.0, 70.0)):
        p = _params(synth(s1_extent=extent)[0])[3]
        for k in ('PI', 'PT', 'SS', 'L1PA'):
            assert abs(p[k] - ref[k]) < 1e-6, (extent, k)
