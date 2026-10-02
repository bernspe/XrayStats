import numpy as np

from spine_geom.patient import between_image_tau2, patient_constant, pooled_sd, robust_sd


def simulate(n_pat=400, sigma_m=4.0, tau=3.0, seed=0):
    rng = np.random.default_rng(seed)
    pats = []
    for _ in range(n_pat):
        true = rng.normal(55, 10)
        k = rng.integers(1, 5)                                     # Bilder je Patient
        imgs = [true + rng.normal(0, tau) + rng.normal(0, sigma_m, rng.integers(1, 4)) for _ in range(k)]
        pats.append((true, imgs))
    return pats


def test_recovers_sigma_tau_and_calibrated_se():
    pats = simulate()
    sigma = pooled_sd(r for _, imgs in pats for r in imgs)
    means = [[r.mean() for r in imgs] for _, imgs in pats]
    ns = [[len(r) for r in imgs] for _, imgs in pats]
    tau2 = between_image_tau2(means, ns, sigma)
    assert abs(sigma - 4.0) < 0.3 and abs(np.sqrt(tau2) - 3.0) < 0.6
    z = [(est - true) / se for (true, _), m, n in zip(pats, means, ns)
         for est, se, *_ in [patient_constant(m, n, sigma, tau2)]]
    assert 0.85 < np.mean(np.abs(z) < 1.96) / 0.95 < 1.05            # SE deckt ~95 %


def test_outlier_image_flagged():
    est, se, z, flagged = patient_constant([55, 56, 54, 80], [3, 3, 3, 3], sigma_m=3.0, tau2=1.0)
    assert flagged.tolist() == [False, False, False, True]
    assert abs(est - 55) < 1e-9 and abs(z[3]) > 3 and np.all(np.abs(z[:3]) < 3)


def test_single_image():
    est, se, z, flagged = patient_constant([60], [2], sigma_m=4.0, tau2=4.0)
    assert est == 60 and abs(se - np.sqrt(8 + 4)) < 1e-12 and z[0] == 0 and not flagged.any()


def test_two_images_never_both_dropped():
    est, se, z, flagged = patient_constant([50, 90], [1, 1], sigma_m=3.0, tau2=1.0)
    assert not flagged.any() and abs(z[0]) > 3                       # beide auffällig, keiner entscheidbar


def test_robust_sd_ignores_gross_errors():
    rng = np.random.default_rng(3)
    groups = [list(rng.normal(0, 2.0, 3)) for _ in range(2000)]
    for g in groups[:60]:                                         # 3 % grobe Fehler wie in den Daten
        g[0] += 30
    assert abs(robust_sd(groups) - 2.0) < 0.15 and pooled_sd(groups) > 3.0
