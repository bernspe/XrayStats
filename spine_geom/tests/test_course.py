import pandas as pd

from spine_geom.course import NA, apply_filters, course, patient_clinical, values_of


def test_values_of():
    assert values_of('') == {NA} and values_of([]) == {NA} and values_of(None) == {NA}
    assert values_of('3') == {'3'} and values_of(True) == {'ja'} and values_of(['L4/5']) == {'L4/5'}


def test_course_baseline_is_last_pre_visit():
    d = pd.to_datetime
    v = pd.DataFrame({'patient_id': 'p', 'series': 'LL', 'date': d(['2020-03-01', '2020-01-01', '2020-06-01', '2021-06-01']),
                      'value': [40., 30., 50., 60.]})
    c = course(v, {'p': d('2020-06-01')})
    assert c['value'].tolist()[0] == 40 and c['delta'].tolist() == [0, 10, 20]      # Baseline, Index-Tag zählt als Post
    assert course(v, {}).empty and course(v, {'p': d('2019-01-01')}).empty            # ohne Index / ohne Baseline


def test_clinical_filter():
    e = pd.DataFrame({'patient': ['u1', 'u1', 'u2'], 'referring_date': ['2020-01-01', '2020-06-01', '2020-02-01'],
                      'therapy': ['fusion', 'konservativ', ''], 'complication': [[], ['Infekt'], []],
                      'fused_segments': ['4', '0', '0']})
    c = patient_clinical(e, {'u1': 'A', 'u2': 'B'})
    assert c.loc['A', 'therapy'] == {'fusion'} and c.loc['A', 'complication'] == {'Infekt'}
    assert c.loc['B', 'complication'] == {NA} and c.loc['B', 'long_term_complication'] == {NA}
    assert apply_filters(c, {'therapy': ['fusion']}) == {'A'} and apply_filters(c, {'therapy': []}) == {'A', 'B'}


def test_index_entry_beats_first_entry():
    e = pd.DataFrame({'patient': ['u1'] * 3, 'referring_date': ['2020-01-01', '2020-06-01', '2021-01-01'],
                      'therapy': ['konservativ', 'konservativ', 'Dekompression']})
    d = pd.Timestamp
    u = {'u1': 'A'}
    assert patient_clinical(e, u, {'A': d('2021-01-01')}).loc['A', 'therapy'] == {'Dekompression'}   # am Index
    assert patient_clinical(e, u, {'A': d('2020-09-01')}).loc['A', 'therapy'] == {'Dekompression'}   # nächster danach
    assert patient_clinical(e, u, {'A': d('2022-01-01')}).loc['A', 'therapy'] == {'konservativ'}     # nichts danach → erster
    assert patient_clinical(e, u).loc['A', 'therapy'] == {'konservativ'}                              # ohne Index → erster
