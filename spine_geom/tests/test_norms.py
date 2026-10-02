import copy
import math

from spine_geom.norms import deviation_state, load_norms, population_reference

NORMS = load_norms()


def _rows(rows, art, param):
    return [r for r in rows if r['art'] == art and r['param'] == param]


def test_default_is_means_overall():
    ref = population_reference('PI', 35, 'diverse', NORMS)                # kein Stratum passt
    assert ref['source'] == 'MEANS_spinopelvic' and ref['stratum'] == 'gesamt' and ref['mean'] == 52.0


def test_yukawa_sex_decade_wins_half_open_ages():
    assert population_reference('PI', 65, 'female', NORMS)['mean'] == 56.0              # Yukawa Tab. 5
    assert population_reference('PI', 69.9, 'male', NORMS)['mean'] == 52.0
    assert population_reference('PT', 70.0, 'female', NORMS)['mean'] == 21.0            # 70 gehört zu 70–79
    # ≥ 80: kein Yukawa-Stratum → Altersstratum Wang (> 75) statt MEANS gesamt
    ref = population_reference('PI', 82, 'female', NORMS)
    assert ref['source'] == 'Wang2025' and ref['mean'] == 49.28
    # ohne Geschlecht/Alter → MEANS gesamt
    assert population_reference('PI', None, None, NORMS)['source'] == 'MEANS_spinopelvic'


def test_stratum_with_sd_beats_stratum_without():
    n = copy.deepcopy(NORMS)
    ref = population_reference('T1PA', 70, 'male', n)
    assert ref['source'] == 'Gharbi2024' and ref['sd'] == 7.1                          # MEANS 13,7 ohne SD verliert
    n['population']['T1PA'] = [e for e in n['population']['T1PA'] if e['source'] == 'MEANS_global']
    assert population_reference('T1PA', 70, 'male', n)['mean'] == 13.7


def test_ll_only_l1_s1_sources():
    assert {e['source'] for e in NORMS['population']['LL']} == {'MEANS_spinopelvic', 'Wang2025'}
    assert population_reference('LL', 40, 'female', NORMS)['source'] == 'MEANS_spinopelvic'
    assert population_reference('LL', 76, 'female', NORMS)['mean'] == 47.30


def test_population_z_uses_patient_pi_and_pi_ll():
    rows = deviation_state({'PI': (99.0, 9.0), 'LL': (50.0, 4.0)}, pi=(62.0, 3.0), age=None, sex=None, norms=NORMS)
    pi = _rows(rows, 'Population', 'PI')[0]
    assert pi['wert'] == 62.0 and abs(pi['z'] - 10.0 / 10.7) < 1e-9                  # Visitenwert 99 ignoriert
    pill = _rows(rows, 'Population', 'PI_LL')[0]
    assert pill['wert'] == 12.0 and abs(pill['wert_sd'] - 5.0) < 1e-9
    assert _rows(rows, 'Klassifikation', 'schwab_PI_LL')[0]['referenz'] == '+'


def test_pi_target_with_uncertainty():
    rows = deviation_state({'L1PA': (15.0, 2.0)}, pi=(60.0, 4.0), age=60, sex='female', norms=NORMS)
    hills = [r for r in _rows(rows, 'PI-Ziel', 'L1PA') if 'minimales' in r['hinweis']][0]
    assert hills['referenz'] == 11.0 and hills['abweichung'] == 4.0
    assert abs(hills['abweichung_sd'] - math.hypot(2.0, 2.0)) < 1e-9                   # 0,5 · SE_PI = 2
    assert abs(hills['z'] - 4.0 / math.hypot(math.hypot(2.0, 2.0), 2.0)) < 1e-9


def test_sva_units_not_compared_and_flags():
    rows = deviation_state({'SVA_wkd': (2.0, 0.3), 'C7_CSVL_wkd': (0.3, 0.1), 'PT': (20.0, 2.0)}, pi=None, age=95,
                           sex='female', norms=NORMS)
    sva = _rows(rows, 'Population', 'SVA')[0]
    assert sva['abweichung'] is None and 'Einheit' in sva['hinweis']
    assert _rows(rows, 'Population', 'C7_CSVL')[0]['abweichung'] is None
    pt = _rows(rows, 'Population', 'PT')[0]
    assert 'außerhalb Normkohorte' in pt['hinweis'] and '> 70' in pt['hinweis']        # Wang reicht nur bis 90


def test_target_range_and_uncertain_class():
    rows = deviation_state({'T4L1PA_mismatch': (0.0, 1.0), 'PT': (21.0, 2.5)}, pi=None, age=60, sex='male',
                           norms=NORMS)
    assert _rows(rows, 'Zielbereich', 'T4L1PA_mismatch')[0]['abweichung'] == 0.0
    pt = _rows(rows, 'Klassifikation', 'schwab_PT')[0]
    assert pt['referenz'] == '+' and 'unsicher' in pt['hinweis']
