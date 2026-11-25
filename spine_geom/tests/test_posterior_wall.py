import numpy as np

from spine_geom.posterior_wall import posterior_walls, segment_slips


def _box(img, cx, cy, w, h, deg=0.0):
    """Helles, um deg gedrehtes Rechteck; gibt die Ecken (oben-hinten, oben-vorne) zurück."""
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    du, dv = (xx - cx) * ca + (yy - cy) * sa, -(xx - cx) * sa + (yy - cy) * ca
    img[(np.abs(du) < w / 2) & (np.abs(dv) < h / 2)] = 0.8
    corner = lambda p, q: np.array([cx + p * ca - q * sa, cy + p * sa + q * ca])
    return corner(w / 2, -h / 2), corner(-w / 2, -h / 2)


def test_synthetic_anterolisthesis_and_sacrum():
    # 4 Wirbel (60x50 px, Abstand 70), L3 8 px nach anterior (links) versetzt; S1 um 20° gekippt
    img = np.full((420, 220), 0.2)
    cs = [(100, 60), (92, 130), (100, 200), (100, 270), (100, 345)]
    for x, y in cs[:4]:
        _box(img, x, y, 60, 50)
    s1_ps, s1_as = _box(img, *cs[4], 70, 50, deg=-20)
    img += np.random.default_rng(0).normal(0, 0.03, img.shape)
    labels = ['L2', 'L3', 'L4', 'L5', 'S1']

    walls = posterior_walls(img, cs, labels)
    assert np.allclose(walls[1][0], (122, 105), atol=2) and np.allclose(walls[1][1], (122, 155), atol=2)
    assert np.allclose(walls[4][0], s1_ps, atol=4), (walls[4], s1_ps)
    slips = segment_slips(walls, cs, labels)
    assert -10 < slips[0] < -6 and 6 < slips[1] < 10 and abs(slips[2]) < 2 and slips[3] is not None

    # reines Rauschen → kaum plausible Kanten (MIN_EDGE_SNR ist auf Recall bei echten Bildern getrimmt)
    hits = [w is not None for k in range(1, 8)
            for w in posterior_walls(np.random.default_rng(k).normal(0.5, 0.05, img.shape), cs, labels)]
    assert np.mean(hits) < 0.2
