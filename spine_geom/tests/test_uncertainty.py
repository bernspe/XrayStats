import numpy as np

from spine_geom.frames import facing_lateral
from spine_geom.landmarks import extract
from spine_geom.sagittal import sagittal_params
from spine_geom.tests.test_sagittal import synth
from spine_geom.uncertainty import mc_sd, mdc95, pooled_cov


def test_pooled_cov_recovers_known_covariance():
    rng = np.random.default_rng(1)
    true = np.array([[0.04, 0.01], [0.01, 0.09]])
    groups = [rng.multivariate_normal(rng.normal(0, 5, 2), true, rng.integers(2, 5)) for _ in range(3000)]
    cov, df, dropped, total = pooled_cov(groups)
    assert np.allclose(cov, true, atol=0.005) and df > 3000 and dropped == 0
    assert pooled_cov([np.zeros((1, 2))]) == (None, 0, 0, 1)


def test_pooled_cov_trims_gross_errors():
    rng = np.random.default_rng(2)
    groups = [rng.normal(0, 0.05, (3, 2)) for _ in range(500)]
    groups[0][0] += [0, 1.0]                                    # ein Label um einen Wirbel verrutscht
    raw, *_ = pooled_cov(groups)
    cov, _, dropped, total = pooled_cov(groups, max_dev=0.5)
    assert dropped == 1 and total == 1500
    assert np.sqrt(cov[1, 1]) < 0.055 < np.sqrt(raw[1, 1])


def test_mc_endplate_angle_noise_propagates_to_ss_only():
    lm = extract(synth()[0])
    f, _ = facing_lateral(lm.s1_endplate, lm.centroids)
    fn = lambda x: sagittal_params(x, f, lm.calib_px)             # noqa: E731
    sd = mc_sd(fn, lm, f, lm.calib_px, {'ang_S1': 3.0 ** 2}, k=2000)
    assert abs(sd['SS'] - 3.0) < 0.15 and abs(sd['PI'] - 3.0) < 0.15   # PI = PT + SS, PT unberührt
    assert sd['PT'] < 1e-9 and sd['L1PA'] < 1e-9


def test_mc_hip_noise_moves_pt_not_ss():
    lm = extract(synth()[0])
    f, _ = facing_lateral(lm.s1_endplate, lm.centroids)
    fn = lambda x: sagittal_params(x, f, lm.calib_px)             # noqa: E731
    sd = mc_sd(fn, lm, f, lm.calib_px, {'H': np.eye(2) * 0.1 ** 2}, k=2000)
    assert sd['SS'] < 1e-9 and sd['PT'] > 0.5
    assert abs(sd['PI'] - sd['PT']) < 1e-6                        # nur PT-Anteil bewegt sich


def test_mdc95():
    assert abs(mdc95(1.0) - 2.7718) < 1e-3
