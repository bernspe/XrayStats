import numpy as np
import pytest

from spine_geom import SPINE_LABELS
from spine_geom.coronal import centroid_cobb, curves, end_vertebrae, tilts_discrete, tilts_spline
from spine_geom.descriptors import sagittal_shape, segment_heights
from spine_geom.frames import facing_lateral
from spine_geom.landmarks import extract
from spine_geom.qc import craniocaudal_monotonic, robust_z
from spine_geom.tests.test_sagittal import synth

LABELS = SPINE_LABELS[SPINE_LABELS.index('T1'):]      # 18 Wirbel
SP = 1.0                                              # WKD-Einheiten


def s_curve(curve_deg=30.0):
    """x = A·sin(2π y / L) über T1–S1: zwei Kurven mit je curve_deg Cobb (analytisch)."""
    n = len(LABELS)
    L = (n - 1) * SP
    A = np.tan(np.radians(curve_deg / 2)) * L / (2 * np.pi)
    ys = [(n - 1 - k) * SP for k in range(n)]           # T1 oben, S1 bei y = 0
    return {lab: np.array([A * np.sin(2 * np.pi * y / L), y]) for lab, y in zip(LABELS, ys)}


@pytest.mark.parametrize('method', ['discrete', 'spline'])
def test_s_curve_end_vertebrae_apex_convexity(method):
    c = s_curve(30)
    t = tilts_discrete(c) if method == 'discrete' else tilts_spline(c)
    cv = curves(c, t)
    # wahres Neigungsextrem liegt genau zwischen T9 und T10 (y = L/2) → beide korrekt
    assert len(cv) == 2 and cv[0]['upper'] == 'T1' and cv[1]['lower'] == 'S1'
    assert cv[0]['lower'] == cv[1]['upper'] in ('T9', 'T10')
    assert [k['convexity'] for k in cv] == ['rechts', 'links']
    assert cv[0]['apex'] in ('T5', 'T6') and cv[1]['apex'] in ('L2', 'L3')
    for k in cv:
        assert abs(abs(k['angle']) - 30) < 3.0, k                  # Randwirbel einseitig/geglättet


def test_discrete_exact_on_interior_arc():
    """Kreisbogen, äquidistant: zentrale Differenz = exakte Tangente an inneren Punkten (θ = φ)."""
    labs = SPINE_LABELS[SPINE_LABELS.index('T10'):]                # 9 Wirbel, kranial zuerst
    phi = np.linspace(20, -20, len(labs))
    c = {lab: 10 * np.array([1 - np.cos(np.radians(f)), np.sin(np.radians(f))]) for lab, f in zip(labs, phi)}
    t = tilts_discrete(c)
    assert np.allclose([t[k] for k in labs[1:-1]], phi[1:-1], atol=1e-9)
    assert abs(centroid_cobb(t, labs[1], labs[-2]) - (phi[1] - phi[-2])) < 1e-9


def test_noise_does_not_split_curve():
    t = dict(zip(['L1', 'L2', 'L3', 'L4', 'L5'], [0.0, 10.0, 9.0, 18.0, 20.0]))
    assert end_vertebrae(t) == [('L1', 'L5')]


def test_straight_spine_has_no_curve():
    c = {lab: np.array([0.0, -k]) for k, lab in enumerate(LABELS)}
    assert curves(c, tilts_discrete(c)) == [] and curves(c, tilts_spline(c)) == []


def test_sagittal_shape_invariant_to_mirror_and_scale():
    shapes = []
    for facing in (1, -1):
        lm = extract(synth(facing=facing)[0])
        f, _ = facing_lateral(lm.s1_endplate, lm.centroids)
        shapes.append(sagittal_shape(lm, f, lm.calib_px))
    for k in shapes[0]:
        assert np.allclose(shapes[0][k], shapes[1][k]), k
    assert shapes[0]['L4'][1] > shapes[0]['L5'][1] > 0              # kranial = +y im Sakrumsystem


def test_segment_heights_and_qc():
    c = {'L3': np.array([0, 0]), 'L4': np.array([0, -30]), 'S1': np.array([0, -90])}
    assert segment_heights(c, 30.0) == {'L3-L4': 1.0}                # L4–S1 nicht benachbart
    assert craniocaudal_monotonic({'L3': (0, 10), 'L4': (0, 20)})
    assert not craniocaudal_monotonic({'L3': (0, 20), 'L4': (0, 10)})
    assert not robust_z([1, 1, 1, 1, 10]).any()                     # MAD = 0 → keine Ausreißer-Division
    assert abs(robust_z([1, 2, 3, 4, 100])[-1]) > 3


def test_tilt_to_horizontal_is_unsigned_and_direction_free():
    from spine_geom.coronal import tilt_to_horizontal
    for line in ([[0, 0], [10, 1]], [[10, 1], [0, 0]], [[0, 1], [10, 0]]):
        assert abs(tilt_to_horizontal(line) - np.degrees(np.arctan(0.1))) < 1e-9
    assert tilt_to_horizontal([[0, 0], [0, 5]]) == 90.0


def test_coronal_global_ctpa_and_c7_csvl():
    from spine_geom.coronal import coronal_global
    s1 = np.array([0.0, 0.0])
    c = {'T1': np.array([np.tan(np.radians(2.0)) * 400, 400.0]), 'C7': np.array([-12.0, 420.0])}
    out = coronal_global(c, s1, wkd=40.0)
    assert abs(out['CTPA'] - 2.0) < 1e-9 and abs(out['C7_CSVL_wkd'] - 0.3) < 1e-9
    assert coronal_global(c, None, 40.0) == {} and 'C7_CSVL_wkd' not in coronal_global(c, s1, None)


def test_coronal_shape_frame():
    from spine_geom.descriptors import coronal_shape
    # Bild: y nach unten; S1 bei (100, 500), Beckenachse um 10° gekippt, L5 genau „über“ S1 senkrecht zur Achse
    a = np.radians(10)
    direction = np.array([np.cos(a), np.sin(a)])                     # Bildkoordinaten
    up_img = np.array([np.sin(a), -np.cos(a)])                      # senkrecht dazu, nach kranial (Bild-y < 0)
    c = {'S1': np.array([100.0, 500.0]), 'L5': np.array([100.0, 500.0]) + 40 * up_img,
         'L4': np.array([100.0, 500.0]) + 80 * up_img + 20 * direction}
    for d in (direction, -direction):                                # Achsenrichtung egal
        s = coronal_shape(c, 1, c['S1'], d, 40.0)
        assert np.allclose(s['S1'], [0, 0]) and np.allclose(s['L5'], [0, 1.0]) and np.allclose(s['L4'], [0.5, 2.0])
    assert coronal_shape(c, 1, None, direction, 40.0) == {}


def test_spline_needs_five_points():
    c = {k: (0.1 * i, -float(i)) for i, k in enumerate(['L1', 'L2', 'L3', 'L4'])}
    assert tilts_spline(c) == {}                     # scipy make_smoothing_spline: mind. 5 Punkte
