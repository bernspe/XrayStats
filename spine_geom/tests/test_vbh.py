"""Wirbelkörperhöhe aus Cobb-segmental-Linien als Maßstab (DECISIONS.md 2026-09-30)."""
import numpy as np

from spine_geom.frames import patient_scale, vbh_factors
from spine_geom.landmarks import extract


def _line(p, deg, half=(15.0, 25.0), **kw):
    """Linie durch p mit Neigung deg, ungleich lang überstehend (Länge darf nicht eingehen)."""
    e = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    a, b = p - e * half[0], p + e * half[1]
    return dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1], mode='Cobb-segmental', **kw)


def _xmd(tilt=20.0, h=28.0, sp=40.0):
    """L2–L4 auf einer um `tilt` gekippten Geraden; L3 mit schrägen, nicht parallelen Endplatten (Keil)."""
    t = np.array([np.sin(np.radians(tilt)), np.cos(np.radians(tilt))])     # Achse kranial → kaudal
    c = {k: np.array([100.0, 100.0]) + t * sp * j for j, k in enumerate(['L2', 'L3', 'L4'])}
    lines = [_line(c['L3'] - t * h / 2, -tilt + 8), _line(c['L3'] + t * h / 2, -tilt - 5)]
    so = dict(vertebra_range=[21, 23], nodes=[dict(x=v[0], y=v[1]) for v in c.values()], cobbLines=lines)
    return so, c, t


def test_height_at_axis_crossing_independent_of_tilt_wedge_and_length():
    for tilt in (0.0, 20.0, -35.0):
        so, _, _ = _xmd(tilt)
        assert abs(extract(dict(stateObject=so)).vb_heights['L3'] - 28.0) < 1e-6
    assert set(extract(dict(stateObject=so)).vb_heights) == {'L3'}   # L2/L4 ohne Linien → kein Wert


def test_unchanged_model_suggestion_and_duplicates_are_ignored():
    so, c, t = _xmd()
    sug = _line(c['L3'] - t * 14, -12)
    so['suggestedLines'] = [dict(sug)]
    so['cobbLines'] = [dict(sug, src='model'), so['cobbLines'][1]]      # Deckplatte nur als Modellvorschlag
    assert 'L3' not in extract(dict(stateObject=so)).vb_heights
    so['cobbLines'][0] = _line(c['L3'] - t * 13, -12, src='model')      # vom Rater verschoben → zählt
    assert abs(extract(dict(stateObject=so)).vb_heights['L3'] - 27.0) < 1e-6
    so['cobbLines'].append(_line(c['L3'] - t * 10, 0))                  # doppelte Deckplatte → mehrdeutig
    assert 'L3' not in extract(dict(stateObject=so)).vb_heights


def test_vbh_factors_need_min_n():
    rows = [('lat', {'L3': 0.8 * w}, w) for w in np.linspace(30, 50, 25)]
    assert vbh_factors(rows) == {('lat', 'L3'): {'factor': 0.8, 'n': 25}}
    assert vbh_factors(rows[:19]) == {}


def test_patient_scale_uses_own_anatomy_so_changing_source_does_not_jump():
    # Patient hat höhere Wirbel als die Population (0,9 statt 0,77 WKD); rauschfrei, Vergrößerung Visite 2 = 1,2×
    pop = {'L3': 0.77, 'L4': 0.77}
    obs = [(40.0, {'L3': 36.0, 'L4': 36.0}),        # Visite 1: Linie + Höhen
           (40.0, {}),                              # Visite 1, anderer Rater: nur Linie
           (None, {'L3': 43.2}),                    # Visite 2: nur L3, keine Linie
           (None, {})]                              # nichts → kein Maßstab
    s = patient_scale(obs, pop)
    # Prior (SIGMA_ANAT) zieht die Patientenanatomie bei nur 2 Linien etwas zur Population → Toleranz 2 %;
    # mit Populationsfaktor allein wäre das Verhältnis (43,2/0,77)/40 = 1,40 statt 1,2
    assert abs(s[1] - 40.0) < 1e-6 and abs(s[0] / 40.0 - 1) < 0.05
    assert abs(s[2] / s[0] / 1.2 - 1) < 0.02
    assert s[3] is None
