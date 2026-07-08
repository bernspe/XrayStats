"""Self-check for the Cobb-segmental subluxation scoring in annotation_analysis.py.

Run directly: python test_subluxation.py
"""

import math

from annotation_analysis import build_subluxation_segment_df, compute_subluxation_scores, spine_labels

VERTEBRAE = ["T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]


def _line(mx, my, angle_deg, length, swap=False):
    half = length / 2
    rad = math.radians(angle_deg)
    dx, dy = half * math.cos(rad), half * math.sin(rad)
    x1, y1, x2, y2 = mx - dx, my - dy, mx + dx, my + dy
    if swap:
        x1, x2, y1, y2 = x2, x1, y2, y1
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _make_xmd(mx_list, angle_list, length_list, swap_idx=None, with_nodes=True):
    my_list = [200 + 30 * i for i in range(len(VERTEBRAE))]
    lines = [
        _line(mx_list[i], my_list[i], angle_list[i], length_list[i], swap=(i == swap_idx))
        for i in range(len(VERTEBRAE))
    ]
    angles = []
    for i in range(len(VERTEBRAE) - 1):
        angles.append({
            "cobbType": "Cobb-segmental",
            "cobbAngle": {
                "angle": 5,
                "vertebra_upper": VERTEBRAE[i],
                "vertebra_lower": VERTEBRAE[i + 1],
            },
            "upperLine": lines[i],
            "lowerLine": lines[i + 1],
        })
    xmd = {"angles": angles}
    if with_nodes:
        xmd["stateObject"] = {
            "nodes": [{"x": mx_list[i], "y": my_list[i]} for i in range(len(VERTEBRAE))],
            "vertebra_range": [spine_labels.index(VERTEBRAE[0]), spine_labels.index(VERTEBRAE[-1])],
            "calibrationLine": {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 50.0},
        }
    return xmd


def test_translation_outlier_scores_highest():
    mx = [500 + 5 * i for i in range(8)]
    mx[4] += 40  # translation shift at L2
    angles = [2.0] * 8
    lengths = [60 + i for i in range(8)]
    xmd = _make_xmd(mx, angles, lengths)
    scores = compute_subluxation_scores(xmd)
    worst = scores.loc[scores["score"].idxmax(), "vertebra"]
    assert worst == "L2", f"expected L2 to be flagged, got {worst}"


def test_rotation_outlier_scores_highest():
    mx = [500 + 5 * i for i in range(8)]
    angles = [2.0] * 8
    angles[3] = -20.0  # reversed tilt at L1
    lengths = [60 + i for i in range(8)]
    xmd = _make_xmd(mx, angles, lengths)
    scores = compute_subluxation_scores(xmd)
    worst = scores.loc[scores["score"].idxmax(), "vertebra"]
    assert worst == "L1", f"expected L1 to be flagged, got {worst}"


def test_foreshortening_outlier_scores_highest():
    mx = [500 + 5 * i for i in range(8)]
    angles = [2.0] * 8
    lengths = [60 + i for i in range(8)]
    lengths[6] = 20.0  # sudden foreshortening at L4
    xmd = _make_xmd(mx, angles, lengths)
    scores = compute_subluxation_scores(xmd)
    worst = scores.loc[scores["score"].idxmax(), "vertebra"]
    assert worst == "L4", f"expected L4 to be flagged, got {worst}"


def test_endpoint_swap_is_invariant():
    mx = [500 + 5 * i for i in range(8)]
    angles = [2.0] * 8
    lengths = [60 + i for i in range(8)]
    xmd_plain = _make_xmd(mx, angles, lengths)
    xmd_swapped = _make_xmd(mx, angles, lengths, swap_idx=2)
    plain = compute_subluxation_scores(xmd_plain).set_index("vertebra")["score"]
    swapped = compute_subluxation_scores(xmd_swapped).set_index("vertebra")["score"]
    assert (plain - swapped).abs().max() < 1e-9, "endpoint order should not affect scores"


def test_segment_df_combines_upper_and_lower():
    mx = [500 + 5 * i for i in range(8)]
    mx[4] += 40
    angles = [2.0] * 8
    lengths = [60 + i for i in range(8)]
    xmd = _make_xmd(mx, angles, lengths)
    seg_df = build_subluxation_segment_df(xmd)
    # L2 is vertebra_lower of (L1, L2) and vertebra_upper of (L2, L3) — both segments
    # should pick up its elevated score via the max() combination.
    row_l1l2 = seg_df[(seg_df.vertebra_upper == "L1") & (seg_df.vertebra_lower == "L2")].iloc[0]
    row_l2l3 = seg_df[(seg_df.vertebra_upper == "L2") & (seg_df.vertebra_lower == "L3")].iloc[0]
    others_max = seg_df[~seg_df.index.isin([row_l1l2.name, row_l2l3.name])]["score"].max()
    assert row_l1l2["score"] > others_max
    assert row_l2l3["score"] > others_max


def test_classifier_pipeline_runs_end_to_end():
    """Smoke test for the segment-level classifier: build_segment_training_table
    -> train_subluxation_classifier -> compute_segment_predictions, on a tiny
    synthetic dataset (6 clean measurements, 1 with a translation shift at L2).
    Doesn't assert predictive quality (too little synthetic data for that,
    validated separately against real feedback) — just that the pipeline runs
    and returns well-formed output."""
    import pandas as pd
    from annotation_analysis import (
        build_segment_training_table, train_subluxation_classifier, compute_segment_predictions,
    )

    mx_clean = [500 + 5 * i for i in range(8)]
    angles_clean = [2.0] * 8
    lengths_clean = [60 + i for i in range(8)]
    mx_shift = list(mx_clean)
    mx_shift[4] += 40  # same translation shift as test_translation_outlier_scores_highest

    measurements = [
        {"id": i, "rater_name": "r", "xray_measurement_data": _make_xmd(mx_clean, angles_clean, lengths_clean)}
        for i in range(6)
    ]
    xmd_pos = _make_xmd(mx_shift, angles_clean, lengths_clean)
    measurements.append({"id": 100, "rater_name": "r", "xray_measurement_data": xmd_pos})
    df4d = pd.DataFrame(measurements)

    feedback_rows = [
        {"image_id": i, "rater_name": "r", "sublux_label": "keine Subluxation", "sublux_levels": ""}
        for i in range(6)
    ]
    feedback_rows.append({"image_id": 100, "rater_name": "r",
                          "sublux_label": "Subluxation vorhanden", "sublux_levels": "L1/L2;L2/L3"})
    feedback_df = pd.DataFrame(feedback_rows)

    training_df = build_segment_training_table(feedback_df, df4d)
    assert len(training_df) > 0, "expected non-empty training table"
    assert training_df["label"].sum() >= 1, "expected at least one positive segment"

    model, scaler, yellow_t, red_t = train_subluxation_classifier(training_df)
    assert yellow_t <= red_t, "yellow threshold should not exceed red threshold"

    preds = compute_segment_predictions(xmd_pos, model, scaler)
    assert not preds.empty
    assert preds["proba"].dropna().between(0, 1).all(), "predicted probabilities must be in [0,1]"


if __name__ == "__main__":
    test_translation_outlier_scores_highest()
    test_rotation_outlier_scores_highest()
    test_foreshortening_outlier_scores_highest()
    test_endpoint_swap_is_invariant()
    test_segment_df_combines_upper_and_lower()
    test_classifier_pipeline_runs_end_to_end()
    print("All subluxation scoring checks passed.")
