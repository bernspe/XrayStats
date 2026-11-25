import pandas as pd

from spine_geom.segmental import episodes, parse_levels, therapy_events, therapy_group, to_pelvis


def test_parse_levels():
    assert parse_levels('Fusion L4/5') == ['L4/L5']
    assert parse_levels(['Facettengelenksinfiltration LWK 4/5 und LWK 5 / SWK 1']) == ['L4/L5', 'L5/S1']
    assert parse_levels('Dekompression LWK 3/4 und LWK 4/5 beidseits von links') == ['L3/L4', 'L4/L5']
    assert parse_levels('Th12/L1; BWK 11/12') == ['T11/T12', 'T12/L1']
    assert parse_levels('Spondylodese L2–L5') == ['L2/L3', 'L3/L4', 'L4/L5']
    assert parse_levels('PRT LWK 5/SWK 1 rechts') == ['L5/S1']
    assert parse_levels('Th12-Ilium') == ['T12/L1', 'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    assert parse_levels('Revision L2-Os ilium')[0] == 'L2/L3' and to_pelvis('Revision L2-Os ilium')
    assert parse_levels('Lumbamed-disc-Orthese, Physiotherapie 12/2020, L5/6') == []


def test_therapy_group():
    assert therapy_group('Fusion') == ('Fusion', False, 'map')
    assert therapy_group('Konversion in 4-Stab') == ('Fusion', True, 'map')
    assert therapy_group('Indikation', ['Fusion L4/5']) == (None, False, 'map')     # Indikation ist keine Therapie
    assert therapy_group('', ['Facettengelenksinfiltration L2/3'])[0] == 'konservativ'
    assert therapy_group('', ['TLIF L4/5', 'Dekompression'])[0] == 'Fusion'
    assert therapy_group('', ['Gangschule'])[0] == 'konservativ'
    assert therapy_group('', []) == (None, False, 'none')


def _entries():
    return pd.DataFrame({'patient': 'u', 'uuid': ['e1', 'e2', 'e3', 'e4'],
                         'referring_date': ['2020-01-10', '2020-06-01', '2020-06-01', '2021-01-01'],
                         'therapy': ['konservativ', 'Dekompression', 'Fusion', 'Indikation'],
                         'therapy_specification': [['PRT'], ['L3/4'], ['Fusion L4/5'], []],
                         'fused_segments': ['0', '0', '1', '0'], 'fused_to_pelvis': [False, False, False, False]})


def test_same_day_entries_merge_with_priority():
    ev = therapy_events(_entries())
    assert ev['group'].tolist() == ['konservativ', 'Fusion']                       # Indikation fällt weg
    assert ev.iloc[1]['levels'] == ['L3/L4', 'L4/L5'] and ev.iloc[1]['fused_segments'] == 1


def test_episode_windows():
    d = pd.to_datetime
    v = pd.DataFrame({'date': d(['2019-12-01', '2019-12-01', '2020-03-01', '2020-06-01', '2020-07-15', '2021-02-01']),
                      'plane': ['ap', 'lat', 'ap', 'lat', 'ap', 'ap']})
    ep = episodes(therapy_events(_entries()), v, index_date='2020-01-10')
    kons, fus = ep.iloc[0], ep.iloc[1]
    assert kons['baseline_date'] == d('2019-12-01') and kons['late_date'] == d('2020-06-01')   # bis zum Fusionstag
    assert kons['planes'] == ['ap', 'lat'] and kons['status'] == 'auswertbar'
    assert fus['baseline_date'] == d('2020-06-01') and fus['early_date'] == d('2020-07-15')    # Ereignistag = Baseline
    assert fus['late_date'] == d('2021-02-01') and fus['planes'] == ['ap'] and fus['same_day_visit']
    assert fus['baseline_planes'] == ['lat']
    assert round(fus['months_from_index'], 1) == 4.7
    lone = episodes(therapy_events(_entries()), v[v['date'] > d('2021-01-01')])
    assert lone['status'].tolist() == ['keine Baseline', 'keine Baseline']


def test_cache_specifications_parse():
    """Alle therapy_specification-Werte im Cache laufen durch; Werte mit Höhenangabe liefern Segmente."""
    try:
        e = pd.read_pickle('data/degen_entries.pkl')
    except FileNotFoundError:
        return
    for spec in e['therapy_specification'].dropna():
        for s in (spec if isinstance(spec, list) else [spec]):
            segs = parse_levels(s)
            if any(k in str(s) for k in ('L4/5', 'LWK 3/4', 'L2/3', 'L3/4')):
                assert segs, s


# --- Schritt 1: Segmentwinkel, Beckenlinie ---------------------------------------------------------------
import numpy as np

from spine_geom import SPINE_LABELS
from spine_geom.segmental import frontend_spline, node_tilts, pelvis_ref, segment_angles

LUMBAR = ['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1']


def _arc(bulge_x):
    """7 Nodes T12–S1 (Bild-y nach unten, Abstand 40 px), Mitte um bulge_x px seitlich ausgelenkt."""
    y = np.arange(7) * 40.0 + 100
    return list(zip(300 + bulge_x * np.sin(np.pi * np.arange(7) / 6), y))


def test_frontend_spline_is_clamped():
    p = np.array(_arc(30))
    s = frontend_spline(p)
    assert len(s) == 71 and np.allclose(s[0], p[0]) and np.allclose(s[-1], p[-1])   # 10·n + 1, Endnodes exakt
    assert not np.allclose(s[30], p[3])                       # innen Kontrollpunkte, nicht interpolierend
    assert len(frontend_spline(p[:3])) == 0                   # kubisch: mind. 4 Nodes (wie das Frontend)


def test_signs_lateral_and_ap():
    lord = segment_angles(_arc(-40), LUMBAR, 'lat')            # Bauch nach Bild links = anterior → Lordose
    assert sum(s['angle'] for s in lord.values()) > 20
    assert sum(s['angle'] for s in segment_angles(_arc(40), LUMBAR, 'lat').values()) < -20   # gespiegelt = Kyphose
    left = segment_angles(_arc(40), LUMBAR, 'ap')              # radiologisch: Bild rechts = Patient links
    assert sum(s['angle'] for s in left.values()) > 20         # konvex links → Öffnung nach links
    assert all(abs(s['angle']) < 1e-9 for s in segment_angles(_arc(0), LUMBAR, 'ap').values())


def test_exact_sum_and_edge_rules():
    rng = np.random.default_rng(1)
    nodes = [(x + rng.normal(0, 8), y + rng.normal(0, 3)) for x, y in _arc(25)]
    seg = segment_angles(nodes, LUMBAR, 'ap')
    assert list(seg) == ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']        # oberstes Segment T12/L1 entfällt
    assert seg['L5/S1']['src'] == 's1_spline' and seg['L5/S1']['edge'] and not seg['L2/L3']['edge']
    th = node_tilts(nodes, 'ap')
    assert abs(sum(s['angle'] for s in seg.values()) - (th[-1] - th[1])) < 1e-9
    assert list(segment_angles(nodes[:-1], LUMBAR[:-1], 'ap')) == ['L1/L2', 'L2/L3', 'L3/L4']   # endet bei L5


def test_l5s1_from_s1_endplate():
    nodes = _arc(-40)                                          # Lordose, anterior = Bild links
    s1 = nodes[-1]
    # Sakrum-Deckplatte mit 40° Neigung, anteriores (linkes) Ende tiefer, wie im Stand
    ep = [(s1[0] - 25, s1[1] - 10 + 21), (s1[0] + 25, s1[1] - 10 - 21)]
    with_ep = segment_angles(nodes, LUMBAR, 'lat', s1_endplate=ep)['L5/S1']
    spline = segment_angles(nodes, LUMBAR, 'lat')['L5/S1']
    assert with_ep['src'] == 's1_endplate' and not with_ep['edge']
    assert with_ep['angle'] > spline['angle'] + 10             # Deckplatte zeigt die Lordose, die der Spline glättet
    flipped = [ep[1], ep[0]]                                   # Reihenfolge der Endpunkte egal
    assert abs(segment_angles(nodes, LUMBAR, 'lat', s1_endplate=flipped)['L5/S1']['angle'] - with_ep['angle']) < 1e-9


def test_segment_angles_skips_gaps_and_bad_input():
    assert segment_angles(_arc(10)[:3], LUMBAR[:3], 'ap') == {}            # < 4 Nodes: keine Spline
    labels = ['T12', 'L1', 'L2', 'L4', 'L5', 'S1', '?']                    # Lücke L2→L4 wird nicht als Segment gezählt
    assert 'L2/L4' not in segment_angles(_arc(10), labels, 'ap')


def test_parity_with_stored_frontend_values():
    """round(|α|) = gespeicherter `Spline-segmental`-Wert. Stand 24.11.2026: 85 % der Messungen exakt; der Rest
    passt zu keiner Algorithmus-Variante (veraltete gespeicherte Winkel, v. a. zwei Rater) → Schwelle 80 %."""
    try:
        raw = pd.read_pickle('data/degen_raw.pkl')
    except FileNotFoundError:
        return
    ok = []
    for x, t in zip(raw['xray_measurement_data'], raw['img_type']):
        if t not in ('Xray-LWS-ap', 'Xray-LWS-lat') or not isinstance(x, dict):
            continue
        so = x.get('stateObject') or {}
        n, vr = so.get('nodes') or [], so.get('vertebra_range')
        fr = {f"{a['cobbAngle']['vertebra_upper']}/{a['cobbAngle']['vertebra_lower']}": a['cobbAngle']['angle']
              for a in x.get('angles') or [] if a.get('cobbType') == 'Spline-segmental'
              and (a.get('cobbAngle') or {}).get('angle') is not None}
        if len(n) < 4 or not vr or not fr or vr[1] - vr[0] + 1 != len(n):
            continue
        seg = segment_angles([(p['x'], p['y']) for p in n], SPINE_LABELS[vr[0]:vr[1] + 1], t.split('-')[-1])
        ok.append(all(np.floor(abs(seg[k]['angle']) + 0.5) == v for k, v in fr.items() if k in seg))
        if len(ok) == 400:
            break
    assert len(ok) > 100 and np.mean(ok) >= 0.8, np.mean(ok)


def test_pelvis_ref():
    l5, s1 = (300, 360), (300, 400)                                         # L5–S1-Abstand 40 px
    hip = pelvis_ref({'x1': 180, 'y1': 520, 'x2': 420, 'y2': 530}, l5, s1)
    assert hip['pelvis_ref'] == 'hip' and round(hip['w'], 1) == 6.0 and hip['obliquity'] > 0
    assert pelvis_ref({'x1': 240, 'y1': 450, 'x2': 360, 'y2': 450}, l5, s1)['pelvis_ref'] == 'isg'
    assert pelvis_ref({'x1': 200, 'y1': 490, 'x2': 400, 'y2': 490}, l5, s1)['pelvis_ref'] == 'unklar'  # w = 5
    assert pelvis_ref({'x1': 200, 'y1': 380, 'x2': 400, 'y2': 380}, l5, s1)['pelvis_ref'] == 'implausibel'
    assert pelvis_ref(None, l5, s1) is None


def test_episode_angles_per_plane_timepoints():
    from spine_geom.segmental import episode_angles
    d = pd.to_datetime
    sv = pd.DataFrame({'date': d(['2020-01-01', '2020-01-01', '2020-03-01', '2020-03-10', '2020-03-10', '2021-01-01']),
                       'plane': ['ap', 'lat', 'ap', 'ap', 'lat', 'ap'], 'seg': 'L4/L5',
                       'angle': [-10., 20., -8., -4., 25., -6.], 'edge': False})
    ep = {'date': d('2020-03-01'), 'next_event': None}
    t = episode_angles(ep, sv).set_index('plane')
    assert t.loc['ap', 'pre'] == -8 and t.loc['ap', 'early'] == -4 and t.loc['ap', 'late'] == -6   # Ereignistag = prä
    assert t.loc['ap', 'k_early'] == 4 and t.loc['ap', 'd_late'] == 2
    assert t.loc['lat', 'pre'] == 20 and t.loc['lat', 'early'] == 25 and t.loc['lat', 'k_late'] == -5
    assert episode_angles({'date': d('2021-06-01'), 'next_event': None}, sv).empty           # kein Verlauf


def test_episode_angles_no_delta_across_methods():
    from spine_geom.segmental import episode_angles
    d = pd.to_datetime
    sv = pd.DataFrame({'date': d(['2020-01-01', '2020-06-01']), 'plane': 'lat', 'seg': 'L5/S1',
                       'angle': [14., 4.], 'edge': [False, True]})                 # post nur Spline-Fallback
    t = episode_angles({'date': d('2020-03-01'), 'next_event': None}, sv).iloc[0]
    assert t['edge'] and np.isnan(t['d_late']) and np.isnan(t['k_late']) and t['late'] == 4


def test_corner_slips_sign_and_scale():
    from spine_geom.segmental import corner_slips
    # Hinterkanten senkrecht, anterior = Bild links; L4 um 10 px nach anterior (links) versetzt, WKD 50 px
    c = {'L4': ((300.0, 100.0), (300.0 - 0, 140.0)), 'L5': ((310.0, 160.0), (310.0, 200.0)),
         'S1': ((310.0, 220.0), (310.0, 260.0))}
    s = corner_slips(c, 50.0)
    assert set(s) == {'L4/L5', 'L5/S1'} and abs(s['L5/S1']) < 1e-9
    assert abs(s['L4/L5'] - 20.0) < 1e-9                       # 10 px anterior von 50 px = +20 % (Antero)
    retro = corner_slips({**c, 'L4': ((320.0, 100.0), (320.0, 140.0))}, 50.0)['L4/L5']
    assert abs(retro + 20.0) < 1e-9                            # nach posterior (rechts) = Retrolisthese
    assert corner_slips(c, float('nan')) == {} and 'L3/L4' not in s


def test_slip_sources_manual_first_and_missing():
    import segmental_course_pipeline as sp
    d = pd.Timestamp('2021-01-01')
    xmd = {'stateObject': {'nodes': [], 'vertebra_range': [20, 25]}}
    meas = pd.DataFrame({'patient': 'p', 'date': d, 'img_type': 'Xray-LWS-lat', 'image_id': [1, 1, 2, 3],
                         'm_uuid': ['a', 'b', 'c', 'd'], 'rater_id': [7, 8, 7, 7], 'xray_measurement_data': [xmd] * 4})
    manual = pd.DataFrame({'image_id': [1], 'seg': ['L4/L5'], 'slip': [12.0], 'instr_both': [False],
                           'pass_diff': [1.0]})
    seg = lambda v: {'resp': {'segments': [{'segment': 'L4/L5', 'slip': v, 'unc': .1},
                                           {'segment': 'L5/S1', 'slip': 2.0, 'unc': .1}]}}
    sl, missing = sp.slips_of(meas, manual, {'a': seg(5.0), 'c': seg(-3.0)})
    assert missing == {'d'}                                   # b: Bild manuell annotiert → kein Modell nötig
    l45 = sl[sl['seg'] == 'L4/L5'].set_index('image_id')
    assert l45.at[1, 'src'] == 'manuell' and l45.at[1, 'slip'] == 12.0 and l45.at[2, 'slip'] == -3.0
    assert set(sl.loc[sl['image_id'] == 1, 'src']) == {'manuell', 'Modell'}   # L5/S1 nicht manuell → Modell


def test_implant_levels_and_fusion_range():
    from spine_geom.segmental import fusion_range, implant_levels
    c = {k: (300.0, 100.0 + 40 * i) for i, k in enumerate(['L2', 'L3', 'L4', 'L5', 'S1'])}
    dets = [{'cls': 'pedicle', 'ends': [[340, 182], [310, 178]]},      # a.p., Kopf Bild rechts → Patient links
            {'cls': 'pedicle', 'ends': [[260, 222], [290, 218]]},      # L5 rechts
            {'cls': 'cage', 'ends': [[250, 200], [350, 200]]},         # zwischen L4 (180) und L5 (220), von Bild links
            {'cls': 'iliac', 'ends': [[200, 300], [250, 280]]},
            {'cls': 'pedicle', 'ends': [[300, 400], [310, 400]]}]      # weit unter S1 → keine Höhe
    lv = implant_levels(dets, c, 'ap')
    assert [(d['level'], d['side']) for d in lv] == [('L4', 'L'), ('L5', 'R'), ('L4/L5', 'R'), ('Becken', None),
                                                       (None, None)]
    assert fusion_range({'L3', 'L5'}) == ['L3/L4', 'L4/L5'] and fusion_range({'L4'}) == []
    assert fusion_range({'L4'}, pelvis=True) == ['L4/L5', 'L5/S1']


def test_segment_classes_and_episode_implants():
    from spine_geom.segmental import episode_implants, fusion_qc, segment_classes
    segs = ['T12/L1', 'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    k = segment_classes('Fusion', ['L3/L4', 'L4/L5'], segs)
    assert [k[s] for s in segs] == ['fern', 'UIV+2', 'UIV+1', 'fusioniert', 'fusioniert', 'LIV−1']
    assert segment_classes('Dekompression', ['L4/L5'], segs)['L3/L4'] == 'benachbart'
    d = pd.to_datetime
    imp = pd.DataFrame({'date': d(['2020-01-01', '2020-06-02', '2020-06-02', '2020-06-02', '2020-06-02']),
                        'plane': ['ap', 'ap', 'ap', 'ap', 'lat'], 'cls': ['NONE', 'pedicle', 'pedicle', 'cage', 'cage'],
                        'level': [None, 'L4', 'L5', 'L4/L5', 'L4/L5'], 'side': [None, 'L', 'R', 'L', None]})
    r = episode_implants({'date': d('2020-06-01'), 'next_event': None}, imp)
    assert r['implant_pre'] == [] and r['fused_implant'] == ['L4/L5'] and r['cages'] == [('L4/L5', 'L')]
    assert fusion_qc(['L4/L5'], r['fused_implant']) == 'gleich' and fusion_qc([], ['L4/L5']) == 'nur Implantat'


def test_choose_fusion():
    from spine_geom.segmental import choose_fusion
    assert choose_fusion(['T11/T12', 'T12/L1'], 2, ['T10/T11', 'T11/T12', 'T12/L1'], [])[2] == 'Text'
    new, total, src = choose_fusion(['L1/L2', 'L2/L3', 'L5/S1'], 8, ['L1/L2', 'L2/L3', 'L3/L4'], [])
    assert src == 'Implantat' and new == ['L1/L2', 'L2/L3', 'L3/L4']                  # Text passt nicht zur Anzahl
    new, total, src = choose_fusion(['L3/L4', 'L4/L5'], 2, ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5'], ['L1', 'L2', 'L3'])
    assert new == ['L3/L4', 'L4/L5'] and total == ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5']  # Verlängerung
    assert choose_fusion([], None, [], []) == ([], [], 'keine')


def test_episode_course_loss():
    from spine_geom.segmental import episode_course
    d = pd.to_datetime
    sv = pd.DataFrame({'date': d(['2020-01-01', '2020-03-15', '2020-09-01', '2021-03-01']), 'plane': 'ap',
                       'seg': 'L3/L4', 'angle': [-20., -8., -11., -14.], 'edge': False})
    c = episode_course({'date': d('2020-03-01'), 'next_event': None}, sv)
    assert c['is_early'].tolist() == [True, False, False] and c['is_late'].tolist() == [False, False, True]
    assert c['k'].tolist() == [12., 9., 6.] and c['loss'].tolist() == [0., 3., 6.]     # Korrektur geht verloren
    c2 = episode_course({'date': d('2020-03-01'), 'next_event': None},
                        sv.assign(plane='regional', seg='LL'))                       # regionale Parameter gehen auch
    assert len(c2) == 3
