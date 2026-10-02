"""Phase-5-Abnahme: Fehlerrate unter H₀ = α ± Toleranz, Power bei bekannter Drift (Simulation)."""
import numpy as np

from spine_geom.change import change_test, cov_from_sd, deviation_test, toward_target, typology


def _sim(drift, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    sd_m = np.array([3.0, 2.0, 4.0, 1.5])                    # Messfehler je Visite
    corr = np.array([[1, .5, .3, 0], [.5, 1, .2, 0], [.3, .2, 1, .1], [0, 0, .1, 1]])
    tau = np.array([2.0, 1.0, 2.5, 0.5])                     # Aufnahmefehler je Aufnahme
    s_m = cov_from_sd(sd_m, corr)
    s_delta = 2 * s_m + 2 * np.diag(tau ** 2)
    truth = np.array([35.0, 20.0, 45.0, 10.0])
    ps = []
    for _ in range(n):
        x1 = truth + rng.multivariate_normal(np.zeros(4), s_m) + rng.normal(0, tau)
        x2 = truth + drift + rng.multivariate_normal(np.zeros(4), s_m) + rng.normal(0, tau)
        ps.append(change_test(x2 - x1, s_delta)['p'])
    return np.array(ps)


def test_type_one_error_under_h0():
    rate = np.mean(_sim(np.zeros(4)) < 0.05)
    assert abs(rate - 0.05) < 0.01, rate


def test_power_with_known_drift():
    # Drift in nur einer von 4 Komponenten: der Omnibus-D² (4 FG) braucht ≈ 4 σ_Δ für 80 % Power
    # (bei 3 σ_Δ nur ~70 %) → in der App zusätzlich RCI je Komponente zeigen.
    assert np.mean(_sim(np.array([20.0, 0, 0, 0]), n=1000) < 0.05) > 0.8


def test_rci_and_projection():
    r = change_test([4.0, 0.0], np.diag([4.0, 1.0]))
    assert np.allclose(r['rci'], [2.0, 0.0]) and abs(r['d2'] - 4.0) < 1e-12 and r['df'] == 2
    comp, sd = toward_target([3.0, 4.0], x1=[0, 0], target=[0, 10], sigma_delta=np.eye(2))
    assert comp == 4.0 and sd == 1.0
    assert toward_target([1, 1], [0, 0], [0, 0], np.eye(2))[0] == 0.0


def test_typology_and_deviation():
    assert typology(True, True) == 'dekompensierend' and typology(False, False) == 'unauffällig'
    assert deviation_test([30.0], [10.0], [5.0])['deviant']
    assert not deviation_test([11.0, 9.0], [10.0, 10.0], [5.0, 5.0])['deviant']


def test_nearest_psd_repairs_inconsistent_correlations():
    from spine_geom.change import nearest_psd
    r = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]])        # paarweise plausibel, gemeinsam unmöglich
    assert np.linalg.eigvalsh(r).min() < 0
    fixed = nearest_psd(r)
    assert np.linalg.eigvalsh(fixed).min() > 0 and np.allclose(fixed, fixed.T)
    good = np.array([[2.0, .5], [.5, 1.0]])
    assert np.allclose(nearest_psd(good), good)


def test_mean_shift_regional_and_relative():
    from spine_geom.change import mean_shift
    # 4 Wirbel × 2 Achsen: Block 0–1 „thorakal“, 2 = L1, 3 = L2
    dx = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 0.2, 0.0])
    s = np.eye(8)
    whole = mean_shift(dx, s, 4, 2, members=[0, 1, 2, 3])
    assert np.allclose(whole['g'], [0.8, 0.375]) and np.allclose(whole['sd'], np.sqrt(4 / 16))
    rel = mean_shift(dx, s, 4, 2, members=[0, 1], ref=2)                # BWS bewegt sich genau wie L1
    assert np.allclose(rel['g'], [0, 0]) and np.allclose(rel['sd'], np.sqrt(0.5 + 1.0))
    assert rel['test']['df'] == 2 and rel['test']['p'] == 1.0
