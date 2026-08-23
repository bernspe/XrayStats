"""Core analysis logic for annotation correctness study.

All testable, non-visual logic lives here so it can be imported,
tested, and debugged independently from the Jupyter notebook.

Main entry points for the notebook:
    build_df4d(df)                       -> df4d
    detect_label_errors(df4d)            -> label_errors DataFrame
    apply_label_corrections(df4d, errs)  -> df4d_corrected
    build_nodes_df(df4d_source)          -> nodes DataFrame
    build_angles_df(df4d_source, ...)            -> angles DataFrame (cobbType incl. inflection_r*)
    build_vertebra_range_df(df4d_source)        -> vertebra_range per measurement
    compute_icc(df, ...)                    -> dict with ICC, CI95, n
    compute_std_by_label(nodes_df)          -> DataFrame with x_std, y_std per label
    get_inflection_overlay_data(xmd, ...)    -> dict with angles + perp_lines for visualization
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotmap import DotMap
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import f as f_dist

from bspline_python import (
    interpolate_bspline,
    smooth_nodes,
    smooth_nodes_pinned,
    calculate_tangents,
    get_perpendicular_lines_at_indices,
    map_nodes_to_spline_and_keys,
    generate_COBB_angles_from_tangents,
    find_tangent_extrema_indices,
    find_tangent_inflection_indices,
    find_optimal_smoothing_window,
    Line,
    calculate_angle,
    NodeItem,
    spine_labels,
    spine_regions,
    convertMeasurementBorderToSpineRegion,
)

# Pre-built mapping: vertebra label -> spine region name
label_to_region: dict[str, str] = {}
for _region, (_start, _end) in spine_regions.items():
    for _i in range(_start, _end + 1):
        if _i < len(spine_labels):
            label_to_region[spine_labels[_i]] = _region

# CobbType classification — drives ICC grouping strategy
REGIONAL_COBB_TYPES: set[str] = {"Cobb-regional", "Spline", "reconstructed_regional"}
SEGMENTAL_COBB_TYPES: set[str] = {"Cobb-segmental", "reconstructed_segmental"}
LUMBAR_SEGMENT_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("T12", "L1"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4", "L5"), ("L5", "S1"),
})


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def label_to_index(label: str) -> int | None:
    return spine_labels.index(label) if label in spine_labels else None


def correct_measurement_label_shift(data: dict, shift_correction: int) -> dict:
    """Return a copy of *data* with vertebra labels shifted by *shift_correction*."""
    angles = []
    for angle in data.get("angles", []):
        ca = angle.get("cobbAngle", {})
        lo_idx = label_to_index(ca.get("vertebra_lower", ""))
        up_idx = label_to_index(ca.get("vertebra_upper", ""))
        if lo_idx is None or up_idx is None:
            continue
        new_lo = lo_idx + shift_correction
        new_up = up_idx + shift_correction
        if not (0 <= new_lo < len(spine_labels)) or not (0 <= new_up < len(spine_labels)):
            continue
        angles.append({
            **angle,
            "cobbAngle": {
                "angle": ca["angle"],
                "vertebra_lower": spine_labels[new_lo],
                "vertebra_upper": spine_labels[new_up],
            },
        })
    if not angles:
        angles = data.get("angles", [])
    vr = data.get("stateObject", {}).get("vertebra_range", [0, 0])
    new_vr = [
        max(0, vr[0] + shift_correction),
        min(len(spine_labels) - 1, vr[1] + shift_correction),
    ]
    return {
        **data,
        "angles": angles,
        "stateObject": {**data.get("stateObject", {}), "vertebra_range": new_vr},
    }


def reconstruct_datasets(measurements_series: pd.Series) -> list[dict]:
    """Reconstruct B-Spline + Cobb angles for a series of xray_measurement_data dicts."""
    datasets = []
    for idx, m in measurements_series.items():
        if not isinstance(m, dict):
            continue
        nodes_raw = m.get("stateObject", {}).get("nodes", [])
        if len(nodes_raw) < 3:
            continue
        nodes = [(n["x"], n["y"]) for n in nodes_raw]
        vr = m["stateObject"].get("vertebra_range", [0, 0])
        selected_vertebra = spine_labels[vr[0] : vr[1] + 1]
        spline = interpolate_bspline(nodes)
        if len(spline) < 3:
            continue
        nodesmap = [DotMap(n) for n in nodes_raw]
        mapped = map_nodes_to_spline_and_keys(nodesmap, selected_vertebra, spline)
        node_idcs = [mn.spline_idx for mn in mapped]
        tangents = calculate_tangents(nodes, [], length=10)
        perp_lines = get_perpendicular_lines_at_indices(spline, length=20, indices=node_idcs)
        try:
            segmental = generate_COBB_angles_from_tangents(tangents, mapped, perp_lines, mode="segmental")
            regional = generate_COBB_angles_from_tangents(tangents, mapped, perp_lines, mode="regional_S1")
        except (IndexError, Exception):
            segmental, regional = [], []
        datasets.append({
            "df_idx": idx,
            "mapped_nodes": mapped,
            "regionalAngles": regional,
            "segmentalAngles": segmental,
        })
    return datasets


def getSmoothedSplineAngles(xmd: dict, smooth_sigma: float) -> list[dict]:
    """Recompute API Spline angles on Gaussian-smoothed nodes, keeping user-defined vertebra pairs."""
    if not isinstance(xmd, dict) or not xmd.get("stateObject", {}).get("nodes"):
        return []
    spline_angles = [a for a in xmd.get("angles", []) if a.get("cobbType") == "Spline"]
    if not spline_angles:
        return []
    nodes_raw = xmd["stateObject"]["nodes"]
    nodes = smooth_nodes([(n["x"], n["y"]) for n in nodes_raw], sigma=smooth_sigma)
    vr = xmd["stateObject"].get("vertebra_range", [0, 0])
    selected_vertebra = spine_labels[vr[0]: vr[1] + 1]
    spline = interpolate_bspline(nodes)
    if len(spline) < 3:
        return []
    mapped = map_nodes_to_spline_and_keys(
        [NodeItem(x=x, y=y) for x, y in nodes], selected_vertebra, spline
    )
    lbl_to_idx = {m.label: m.spline_idx for m in mapped}
    result = []
    for a in spline_angles:
        ca = a.get("cobbAngle", {})
        vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
        if vu not in lbl_to_idx or vl not in lbl_to_idx:
            continue
        perps = get_perpendicular_lines_at_indices(
            spline, length=20, indices=[lbl_to_idx[vu], lbl_to_idx[vl]]
        )
        if len(perps) < 2:
            continue
        l1 = Line(x1=perps[0]["x1"], y1=perps[0]["y1"], x2=perps[0]["x2"], y2=perps[0]["y2"])
        l2 = Line(x1=perps[1]["x1"], y1=perps[1]["y1"], x2=perps[1]["x2"], y2=perps[1]["y2"])
        result.append({
            "cobbType": "Spline",
            "cobbAngle": {"angle": calculate_angle(l1, l2), "vertebra_upper": vu, "vertebra_lower": vl},
        })
    return result


def getReconstructedCobbAngles(xmd: dict, smooth_sigma: float = 0.0) -> list[dict]:
    """Reconstruct Cobb angles from raw stateObject nodes via B-Spline + perpendicular lines.

    smooth_sigma > 0 applies Gaussian smoothing to the node positions before spline fitting.
    """
    if not isinstance(xmd, dict) or not xmd.get("stateObject", {}).get("nodes"):
        return []
    nodes_raw = xmd["stateObject"]["nodes"]
    nodes = [(n["x"], n["y"]) for n in nodes_raw]
    if smooth_sigma > 0.0:
        nodes = smooth_nodes(nodes, sigma=smooth_sigma)
    vr = xmd["stateObject"].get("vertebra_range", [0, 0])
    selected_vertebra = spine_labels[vr[0] : vr[1] + 1]
    spline = interpolate_bspline(nodes)
    if len(spline) < 3:
        return []
    nodesmap = [DotMap(n) for n in nodes_raw]
    mapped = map_nodes_to_spline_and_keys(nodesmap, selected_vertebra, spline)
    node_idcs = [mn.spline_idx for mn in mapped]
    tangents = calculate_tangents(nodes, [], length=10)
    perp_lines = get_perpendicular_lines_at_indices(spline, length=20, indices=node_idcs)
    try:
        segmental = generate_COBB_angles_from_tangents(tangents, mapped, perp_lines, mode="segmental")
        regional = generate_COBB_angles_from_tangents(tangents, mapped, perp_lines, mode="regional_S1")
    except (IndexError, Exception):
        return []
    lumbar_labels_upper = {"T12", "L1", "L2", "L3", "L4", "L5"}
    lumbar_labels_lower = {"L1", "L2", "L3", "L4", "L5", "S1"}
    lumbar_segmental = [
        a for a in segmental
        if a["from_label"] in lumbar_labels_upper and a["to_label"] in lumbar_labels_lower
    ]
    result = []
    for a in regional:
        result.append({
            "cobbAngle": {"angle": a["cobb_angle"], "vertebra_lower": a["to_label"], "vertebra_upper": a["from_label"]},
            "cobbType": "reconstructed_regional",
        })
    for a in lumbar_segmental:
        result.append({
            "cobbAngle": {"angle": a["cobb_angle"], "vertebra_lower": a["to_label"], "vertebra_upper": a["from_label"]},
            "cobbType": "reconstructed_segmental",
        })
    return result


def getInflectionCobbAngles(xmd: dict, min_angle: float = 5.0, smooth_sigma: float = 0.0) -> list[dict]:
    """Cobb angles at true spline inflection points (zero crossings of tangent angle).

    Measurement boundaries are placed where the tangent angle changes sign, i.e. where
    the spine transitions from one curve direction to the other.  Each segment therefore
    spans one complete scoliosis curve.

    This is different from placing boundaries at tangent-angle extrema (apex vertebrae),
    which would bisect a curve and produce clinically meaningless sub-segments.

    smooth_sigma > 0 applies Gaussian smoothing to nodes before spline fitting.
    Angles below min_angle degrees are discarded; survivors are re-ranked so
    'inflection_r0' is always the first clinically significant angle from the top.
    cobbType = 'inflection_r0', 'inflection_r1', …
    vertebra_upper/lower: nearest mapped vertebra label for display.
    """
    if not isinstance(xmd, dict) or not xmd.get("stateObject", {}).get("nodes"):
        return []
    nodes_raw = xmd["stateObject"]["nodes"]
    if len(nodes_raw) < 3:
        return []
    nodes = [(n["x"], n["y"]) for n in nodes_raw]
    if smooth_sigma > 0.0:
        nodes = smooth_nodes(nodes, sigma=smooth_sigma)
    vr = xmd["stateObject"].get("vertebra_range", [0, 0])
    selected_vertebra = spine_labels[vr[0]: vr[1] + 1]

    spline = interpolate_bspline(nodes, num_steps=200)
    if len(spline) < 10:
        return []

    nodesmap = [DotMap(n) for n in nodes_raw]
    mapped = map_nodes_to_spline_and_keys(nodesmap, selected_vertebra, spline)
    if not mapped:
        return []

    node_spline_indices = [m.spline_idx for m in mapped]

    # Adaptive extrema: find the minimum smoothing window that keeps ≤2 extrema
    node_tangents = calculate_tangents(nodes, [], length=50)
    node_angles = [t.angle for t in node_tangents]
    opt_window = find_optimal_smoothing_window(node_angles, target_max_extrema=2)
    extrema_node_idx = find_tangent_extrema_indices(node_angles, smoothing_window=opt_window)

    extrema_spline = [
        node_spline_indices[ni]
        for ni in extrema_node_idx
        if 0 <= ni < len(node_spline_indices)
    ]

    meas_idx_raw = [node_spline_indices[0]] + extrema_spline + [node_spline_indices[-1]]
    seen: set[int] = set()
    meas_spline_idx = [x for x in meas_idx_raw if not (x in seen or seen.add(x))]

    perp_lines = get_perpendicular_lines_at_indices(spline, length=200, indices=meas_spline_idx)

    def nearest_label(si: int) -> str:
        return min(mapped, key=lambda m: abs(m.spline_idx - si)).label

    raw_angles = []
    for k, (i, j) in enumerate(zip(meas_spline_idx[:-1], meas_spline_idx[1:])):
        if k >= len(perp_lines) - 1:
            break
        try:
            angle = calculate_angle(Line(**perp_lines[k]), Line(**perp_lines[k + 1]))
        except Exception:
            continue
        raw_angles.append({
            "angle": angle,
            "vertebra_upper": nearest_label(i),
            "vertebra_lower": nearest_label(j),
        })

    filtered = [a for a in raw_angles if a["angle"] >= min_angle]
    return [
        {
            "cobbAngle": {
                "angle": a["angle"],
                "vertebra_upper": a["vertebra_upper"],
                "vertebra_lower": a["vertebra_lower"],
            },
            "cobbType": f"inflection_r{new_rank}",
        }
        for new_rank, a in enumerate(filtered)
    ]


def get_inflection_overlay_data(xmd: dict, min_angle: float = 10.0, smooth_sigma: float = 0.0) -> dict:
    """Inflection-point Cobb angles together with perpendicular line positions for visualization.

    Like getInflectionCobbAngles but also returns the perpendicular lines so a viewer can
    draw them as overlays on the X-ray image. All coordinates are in the original canvas
    space (same as stateObject.nodes); the caller must scale for display.

    Returns:
        {
          'angles':     list of angle dicts (cobbAngle + cobbType, same as getInflectionCobbAngles),
          'perp_lines': list of {x1, y1, x2, y2} for each boundary line of a passing angle
        }
    """
    if not isinstance(xmd, dict) or not xmd.get("stateObject", {}).get("nodes"):
        return {"angles": [], "perp_lines": []}
    nodes_raw = xmd["stateObject"]["nodes"]
    if len(nodes_raw) < 3:
        return {"angles": [], "perp_lines": []}

    nodes = [(n["x"], n["y"]) for n in nodes_raw]
    if smooth_sigma > 0.0:
        nodes = smooth_nodes(nodes, sigma=smooth_sigma)

    vr = xmd["stateObject"].get("vertebra_range", [0, 0])
    selected_vertebra = spine_labels[vr[0]: vr[1] + 1]

    spline = interpolate_bspline(nodes, num_steps=200)
    if len(spline) < 10:
        return {"angles": [], "perp_lines": []}

    # Use (possibly smoothed) node positions for spline mapping
    nodesmap = [DotMap({"x": x, "y": y}) for x, y in nodes]
    mapped = map_nodes_to_spline_and_keys(nodesmap, selected_vertebra, spline)
    if not mapped:
        return {"angles": [], "perp_lines": []}

    node_spline_indices = [m.spline_idx for m in mapped]

    # Adaptive extrema: find the minimum smoothing window that keeps ≤2 extrema
    node_tangents = calculate_tangents(nodes, [], length=50)
    node_angles = [t.angle for t in node_tangents]
    opt_window = find_optimal_smoothing_window(node_angles, target_max_extrema=2)
    extrema_node_idx = find_tangent_extrema_indices(node_angles, smoothing_window=opt_window)

    extrema_spline = [
        node_spline_indices[ni]
        for ni in extrema_node_idx
        if 0 <= ni < len(node_spline_indices)
    ]

    meas_idx_raw = [node_spline_indices[0]] + extrema_spline + [node_spline_indices[-1]]
    seen: set[int] = set()
    meas_spline_idx = [x for x in meas_idx_raw if not (x in seen or seen.add(x))]

    all_perp_lines = get_perpendicular_lines_at_indices(spline, length=200, indices=meas_spline_idx)

    def nearest_label(si: int) -> str:
        return min(mapped, key=lambda m: abs(m.spline_idx - si)).label

    # Collect per-segment data (angle + which perp-line indices it uses)
    raw_segments = []
    for k, (i, j) in enumerate(zip(meas_spline_idx[:-1], meas_spline_idx[1:])):
        if k >= len(all_perp_lines) - 1:
            break
        try:
            angle = calculate_angle(Line(**all_perp_lines[k]), Line(**all_perp_lines[k + 1]))
        except Exception:
            continue
        raw_segments.append({
            "angle":          angle,
            "vertebra_upper": nearest_label(i),
            "vertebra_lower": nearest_label(j),
            "idx_upper":      k,
            "idx_lower":      k + 1,
        })

    filtered = [(new_rank, s) for new_rank, s in enumerate(raw_segments) if s["angle"] >= min_angle]

    angles_out = [
        {
            "cobbAngle": {
                "angle":          s["angle"],
                "vertebra_upper": s["vertebra_upper"],
                "vertebra_lower": s["vertebra_lower"],
            },
            "cobbType": f"inflection_r{new_rank}",
        }
        for new_rank, s in filtered
    ]

    # Only keep perp lines that bound at least one passing angle
    used_indices: set[int] = set()
    for _, s in filtered:
        used_indices.add(s["idx_upper"])
        used_indices.add(s["idx_lower"])

    perp_lines_out = [
        all_perp_lines[i]
        for i in sorted(used_indices)
        if i < len(all_perp_lines)
    ]

    return {"angles": angles_out, "perp_lines": perp_lines_out}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_df4d(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw API response DataFrame into df4d (one row per measurement).

    Extracts rater_name and rater_id from measure_dataset.user before unpacking
    so they survive the subsequent pd.Series expansion.
    """
    cols = ["id", "uuid", "img", "patient_id", "measure_dataset",
            "referring_patient", "img_type", "acquisition_date"]
    df2 = df[cols]
    df3 = df2[df2["measure_dataset"].map(len) > 0]

    df4 = pd.concat(
        [df3.drop(columns=["measure_dataset"]), df3["measure_dataset"].apply(pd.Series)],
        axis=1,
    )
    id_vars = ["id", "uuid", "img", "patient_id", "referring_patient", "img_type", "acquisition_date"]
    df4b = df4.melt(id_vars=id_vars, value_name="measure_dataset", var_name="measure_dataset_index").dropna(
        subset=["measure_dataset"]
    )

    df4b["rater_name"] = df4b["measure_dataset"].apply(
        lambda m: m.get("user", {}).get("first_name", "Unknown") if isinstance(m, dict) else "Unknown"
    )
    df4b["rater_id"] = df4b["measure_dataset"].apply(
        lambda m: m.get("user", {}).get("id", None) if isinstance(m, dict) else None
    )

    measure_expanded = df4b["measure_dataset"].apply(pd.Series)
    df4c = pd.concat(
        [df4b.drop(columns=["measure_dataset"]), measure_expanded["data"], measure_expanded["uuid"].rename("m_uuid")],
        axis=1,
    )
    df4d = pd.concat([df4c.drop(columns=["data"]), df4c["data"].apply(pd.Series)], axis=1)
    return df4d


# ---------------------------------------------------------------------------
# Label-error detection & correction
# ---------------------------------------------------------------------------

def detect_label_errors(df4d: pd.DataFrame) -> pd.DataFrame:
    """Detect labelling errors (shift and empty) in df4d via k-Means clustering.

    For each image with ≥2 non-empty measurements, cluster node (x,y) positions
    and compare assigned labels to annotated labels. Measurements whose majority
    label assignment differs by >50% are flagged as shift errors.

    Returns a DataFrame with columns: df_idx, m_uuid, error ('shift'|'empty'), shift (int).
    """
    to_shift_list: list[dict] = []
    empty_list: list[dict] = []

    for img_id in df4d["id"].unique():
        xmd_series = df4d[df4d["id"] == img_id]["xray_measurement_data"].head(3)

        for idx, xmd in xmd_series.items():
            if not isinstance(xmd, dict) or len(xmd.get("angles", [])) == 0:
                empty_list.append({
                    "df_idx": idx,
                    "m_uuid": df4d.loc[idx, "m_uuid"],
                    "error": "empty",
                    "shift": 0,
                })

        non_empty = xmd_series[xmd_series.apply(
            lambda m: isinstance(m, dict) and len(m.get("angles", [])) > 0
        )]
        if len(non_empty) < 2:
            continue

        try:
            datasets = reconstruct_datasets(non_empty)
        except Exception:
            continue
        if len(datasets) < 2:
            continue

        mapped_dfs = []
        for i, d in enumerate(datasets):
            mdf = pd.DataFrame(
                [(mn.label, mn.x, mn.y) for mn in d["mapped_nodes"]],
                columns=["label", "x", "y"],
            )
            mdf["measurement_idx"] = i
            mdf["df_idx"] = d["df_idx"]
            mapped_dfs.append(mdf)

        combined = pd.concat(mapped_dfs, ignore_index=True)
        combined["label_index"] = pd.to_numeric(
            combined["label"].apply(label_to_index), errors="coerce"
        )

        label_counts = combined["label_index"].dropna().value_counts().sort_index()
        if label_counts.empty or label_counts.iloc[-1] <= 1:
            continue

        v_min = int(combined["label_index"].dropna().min())
        v_max = int(combined["label_index"].dropna().max())
        n_clusters = v_max - v_min + 1
        if n_clusters < 2:
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(combined[["x", "y"]])
        centers = (
            pd.DataFrame(kmeans.cluster_centers_, columns=["x", "y"])
            .sort_values("y")
            .reset_index(drop=True)
        )
        centers["assigned_label"] = spine_labels[v_min : v_max + 1]

        def _assign_label(row, _centers=centers):
            return _centers.iloc[((_centers["y"] - row["y"]) ** 2).argmin()]["assigned_label"]

        combined["assigned_label"] = combined.apply(_assign_label, axis=1)
        combined["label_match"] = combined["label"] == combined["assigned_label"]
        combined["assigned_label_index"] = pd.to_numeric(
            combined["assigned_label"].apply(label_to_index), errors="coerce"
        )
        combined["shift_dir"] = combined["assigned_label_index"] - combined["label_index"]

        match_counts = combined.groupby("measurement_idx")["label_match"].sum()
        max_match = match_counts.max()
        if max_match == 0:
            continue

        for meas_i, n_match in match_counts.items():
            if (max_match - n_match) / max_match > 0.5:
                df_idx = int(combined[combined["measurement_idx"] == meas_i]["df_idx"].iloc[0])
                shift_mode = combined[combined["measurement_idx"] == meas_i]["shift_dir"].mode()
                shift_val = int(shift_mode.iloc[0]) if len(shift_mode) > 0 else 0
                to_shift_list.append({
                    "df_idx": df_idx,
                    "m_uuid": df4d.loc[df_idx, "m_uuid"],
                    "error": "shift",
                    "shift": shift_val,
                })

    return pd.DataFrame(to_shift_list + empty_list)


def apply_label_corrections(df4d: pd.DataFrame, label_errors: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df4d with shift errors corrected."""
    df4d_corrected = df4d.copy()
    for _, err_row in label_errors[label_errors["error"] == "shift"].iterrows():
        idx = int(err_row["df_idx"])
        original = df4d_corrected.loc[idx, "xray_measurement_data"]
        df4d_corrected.at[idx, "xray_measurement_data"] = correct_measurement_label_shift(
            original, int(err_row["shift"])
        )
    return df4d_corrected


# ---------------------------------------------------------------------------
# Analysis DataFrames
# ---------------------------------------------------------------------------

def build_nodes_df(df4d_source: pd.DataFrame) -> pd.DataFrame:
    """One row per (image_id, rater_name, vertebra_label) with columns x, y."""
    rows = []
    for _, row in df4d_source.iterrows():
        xmd = row["xray_measurement_data"]
        if not isinstance(xmd, dict):
            continue
        nodes_raw = xmd.get("stateObject", {}).get("nodes", [])
        vr = xmd.get("stateObject", {}).get("vertebra_range", [0, 0])
        for i, node in enumerate(nodes_raw):
            label_idx = vr[0] + i
            if label_idx < len(spine_labels):
                rows.append({
                    "image_id": row["id"],
                    "rater_name": row["rater_name"],
                    "rater_id": row["rater_id"],
                    "vertebra_label": spine_labels[label_idx],
                    "x": node["x"],
                    "y": node["y"],
                })
    return pd.DataFrame(rows)


def build_vertebra_range_df(df4d_source: pd.DataFrame) -> pd.DataFrame:
    """One row per measurement: vertebra_range endpoints extracted from stateObject.

    Columns: image_id, img_type, rater_name, vr_start (index), vr_end (index),
             vr_start_label, vr_end_label, span (vr_end - vr_start).
    Rows without nodes or vertebra_range are skipped.
    """
    rows = []
    for _, row in df4d_source.iterrows():
        xmd = row.get("xray_measurement_data")
        if not isinstance(xmd, dict):
            continue
        state = xmd.get("stateObject", {})
        vr = state.get("vertebra_range", None)
        if vr is None or len(vr) < 2 or not state.get("nodes"):
            continue
        vr_start, vr_end = int(vr[0]), int(vr[1])
        if vr_end >= len(spine_labels):
            continue
        rows.append({
            "image_id":       row["id"],
            "img_type":       row["img_type"],
            "rater_name":     row["rater_name"],
            "vr_start":       vr_start,
            "vr_end":         vr_end,
            "vr_start_label": spine_labels[vr_start],
            "vr_end_label":   spine_labels[vr_end],
            "span":           vr_end - vr_start,
        })
    return pd.DataFrame(rows)


def _safe_region(upper: str, lower: str) -> str | None:
    """Wrapper around convertMeasurementBorderToSpineRegion that returns None on bad labels."""
    try:
        return convertMeasurementBorderToSpineRegion(upper, lower)
    except (ValueError, Exception):
        return None


def aggregate_angles_by_region(angles_df: pd.DataFrame, agg: str = "max") -> pd.DataFrame:
    """Collapse multiple angles per (image_id, rater, cobbType, region) to one value.

    Needed when a rater measured sub-curves within the same spine region.
    agg='max' keeps the dominant (largest) curve; 'sum' approximates the full-region angle.
    """
    return (
        angles_df
        .dropna(subset=["angle", "region"])
        .groupby(["image_id", "img_type", "rater_name", "rater_id", "cobbType", "region"],
                 as_index=False)
        ["angle"]
        .agg(agg)
    )


def compute_icc_by_region(
    angles_df: pd.DataFrame,
    cobb_types: list[str],
    active_img_types: list[str],
    agg: str = "max",
    min_images: int = 5,
) -> pd.DataFrame:
    """ICC(2,1) per (img_type, cobbType, region).

    Aggregates multiple angles per rater/image/region first (default: max), then
    runs compute_icc for every (img_type, cobbType, region) combination.

    Returns a long DataFrame with columns: img_type, cobbType, region, ICC, CI95_lower,
    CI95_upper, n. Filter or pivot as needed.
    """
    agg_df = aggregate_angles_by_region(angles_df[angles_df["cobbType"].isin(cobb_types)], agg=agg)
    rows = []
    for img_type in active_img_types:
        sub_it = agg_df[agg_df["img_type"] == img_type]
        for cobb_type in cobb_types:
            sub_ct = sub_it[sub_it["cobbType"] == cobb_type]
            for region in sorted(sub_ct["region"].dropna().unique()):
                sub = sub_ct[sub_ct["region"] == region]
                res = compute_icc(sub, "image_id", "rater_name", "angle", min_images=min_images)
                rows.append({"img_type": img_type, "cobbType": cobb_type, "region": region, **res})
    return pd.DataFrame(rows)


def build_angles_df(
    df4d_source: pd.DataFrame,
    include_reconstructed: bool = False,
    include_inflection: bool = False,
    min_inflection_angle: float = 5.0,
    smooth_sigma: float = 0.0,
) -> pd.DataFrame:
    """One row per (image_id, rater_name, cobbType, vertebra_upper, vertebra_lower) with column angle.

    smooth_sigma > 0 applies Gaussian node smoothing before B-spline fitting for
    reconstructed and inflection angles (API-native angles are never modified).
    """
    rows = []
    for _, row in df4d_source.iterrows():
        xmd = row["xray_measurement_data"]
        if not isinstance(xmd, dict):
            continue
        api_angles = xmd.get("angles", [])
        if smooth_sigma > 0.0:
            all_angles = [a for a in api_angles if a.get("cobbType") != "Spline"]
            all_angles += getSmoothedSplineAngles(xmd, smooth_sigma)
        else:
            all_angles = list(api_angles)
        if include_reconstructed:
            all_angles += getReconstructedCobbAngles(xmd, smooth_sigma=smooth_sigma)
        if include_inflection:
            all_angles += getInflectionCobbAngles(xmd, min_angle=min_inflection_angle, smooth_sigma=smooth_sigma)
        for angle in all_angles:
            ca = angle.get("cobbAngle", {})
            rows.append({
                "image_id": row["id"],
                "img_type": row["img_type"],
                "rater_name": row["rater_name"],
                "rater_id": row["rater_id"],
                "cobbType": angle.get("cobbType", ""),
                "vertebra_upper": ca.get("vertebra_upper", ""),
                "vertebra_lower": ca.get("vertebra_lower", ""),
                "region": _safe_region(ca.get("vertebra_upper", ""), ca.get("vertebra_lower", "")),
                "angle": ca.get("angle", None),
            })
    return pd.DataFrame(rows)


def compute_std_by_label(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Mean inter-rater std of node x/y per vertebra label.

    Computes std across raters *within* each image first, then averages across
    images. Grouping only by vertebra_label would inflate std with between-image
    variation (different images have entirely different coordinate systems).
    Only images with >=2 raters contribute a valid std.
    """
    within = (
        nodes_df.groupby(["vertebra_label", "image_id"])
        .agg(x_std=("x", "std"), y_std=("y", "std"), n_raters=("x", "count"))
        .reset_index()
    )
    within = within[within["n_raters"] >= 2]
    return (
        within.groupby("vertebra_label")
        .agg(x_std=("x_std", "mean"), y_std=("y_std", "mean"))
        .reindex(spine_labels)
    )


# ---------------------------------------------------------------------------
# ICC
# ---------------------------------------------------------------------------

def compute_icc(
    df: pd.DataFrame,
    image_col: str,
    rater_col: str,
    value_col: str,
    min_images: int = 10,
) -> dict:
    """ICC(2,1) for unbalanced inter-rater designs via two-way ANOVA (Shrout & Fleiss 1979).

    pingouin.intraclass_corr() requires balanced data (same raters per image). Because each
    image is annotated by 2–4 different raters from a pool, a naive pivot produces NaN for
    every row and dropna() removes all observations. This implementation derives ICC directly
    from two-way ANOVA mean squares, which handles missing cells correctly.

    Returns dict: {ICC, CI95_lower, CI95_upper, n}  — values are np.nan if insufficient data.
    """
    eligible = (
        df.groupby(image_col)[rater_col].nunique()
        .pipe(lambda s: s[s >= 2])
        .index
    )
    data = df[df[image_col].isin(eligible)][[image_col, rater_col, value_col]].dropna()
    n = data[image_col].nunique()
    if n < min_images:
        return {"ICC": np.nan, "CI95_lower": np.nan, "CI95_upper": np.nan, "n": n}
    try:
        d = data.rename(columns={image_col: "_target", rater_col: "_rater", value_col: "_y"})
        model = ols("_y ~ C(_target) + C(_rater)", data=d).fit()
        aov = sm.stats.anova_lm(model, typ=1)

        MS_B = aov.loc["C(_target)", "mean_sq"]  # between subjects
        MS_R = aov.loc["C(_rater)",  "mean_sq"]  # between raters
        MS_W = aov.loc["Residual",   "mean_sq"]  # error (within)

        k_counts = d.groupby("_target")["_rater"].count()
        k0 = n / (1.0 / k_counts).sum()  # harmonic mean of raters per image

        denom = MS_B + (k0 - 1) * MS_W + (k0 / n) * (MS_R - MS_W)
        icc = (MS_B - MS_W) / denom if denom != 0 else np.nan

        df_B = n - 1
        df_W = len(d) - n - d["_rater"].nunique() + 1
        if df_W > 0:
            Fl = (MS_B / MS_W) / f_dist.ppf(0.975, df_B, df_W)
            Fu = (MS_B / MS_W) * f_dist.ppf(0.975, df_W, df_B)
            icc_l = (Fl - 1) / (Fl + k0 - 1)
            icc_u = (Fu - 1) / (Fu + k0 - 1)
        else:
            icc_l, icc_u = np.nan, np.nan

        def _clip(v):
            return round(float(np.clip(v, -1, 1)), 3) if not np.isnan(v) else np.nan

        return {"ICC": _clip(icc), "CI95_lower": _clip(icc_l), "CI95_upper": _clip(icc_u), "n": n}
    except Exception:
        return {"ICC": np.nan, "CI95_lower": np.nan, "CI95_upper": np.nan, "n": n}


def compute_icc_by_img_type(
    df: pd.DataFrame,
    cobb_types: list[str],
    active_img_types: list[str],
    cobbtype_col: str = "cobbType",
    imgtype_col: str = "img_type",
    image_col: str = "image_id",
    rater_col: str = "rater_name",
    value_col: str = "angle",
    min_images: int = 5,
) -> pd.DataFrame:
    """ICC(2,1) per img_type × cobbType.

    Returns a pivot DataFrame: index = img_type, columns = cobbType (in the
    order given by cobb_types), values = ICC(2,1).
    """
    rows = []
    for img_type in active_img_types:
        sub_it = df[df[imgtype_col] == img_type]
        for cobb_type in cobb_types:
            sub = sub_it[sub_it[cobbtype_col] == cobb_type]
            res = compute_icc(sub, image_col, rater_col, value_col, min_images=min_images)
            rows.append({imgtype_col: img_type, cobbtype_col: cobb_type, **res})
    pivot = pd.DataFrame(rows).pivot(index=imgtype_col, columns=cobbtype_col, values="ICC")
    return pivot[[c for c in cobb_types if c in pivot.columns]]


def compute_icc_by_segment(
    angles_df: pd.DataFrame,
    cobb_types: list[str],
    active_img_types: list[str],
    min_images: int = 5,
) -> pd.DataFrame:
    """ICC(2,1) per (img_type, cobbType, vertebra_upper, vertebra_lower).

    For segmental Cobb angles each adjacent-vertebra pair (e.g. L1/L2) is the
    comparison unit. No region aggregation — each rater has at most one angle per
    segment per image. Returns a long DataFrame sorted by spine position.
    """
    _spine_idx = {lbl: i for i, lbl in enumerate(spine_labels)}
    sub = angles_df[angles_df["cobbType"].isin(cobb_types)]
    rows = []
    for img_type in active_img_types:
        sub_it = sub[sub["img_type"] == img_type]
        for cobb_type in cobb_types:
            sub_ct = sub_it[sub_it["cobbType"] == cobb_type]
            for (vu, vl), grp in sub_ct.groupby(["vertebra_upper", "vertebra_lower"]):
                res = compute_icc(grp, "image_id", "rater_name", "angle", min_images=min_images)
                res["median_std"] = round(
                    grp.groupby("image_id")["angle"].std().median(), 2
                )
                rows.append({
                    "img_type": img_type,
                    "cobbType": cobb_type,
                    "vertebra_upper": vu,
                    "vertebra_lower": vl,
                    **res,
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_u"] = result["vertebra_upper"].map(_spine_idx)
        result["_l"] = result["vertebra_lower"].map(_spine_idx)
        result = result.sort_values(["_u", "_l"]).drop(columns=["_u", "_l"])
    return result


def filter_inflection_max_common_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only inflection_r* rows whose rank all raters of the image share.

    For each image, computes the minimum of each rater's maximum rank.
    Rows with rank > that value are dropped. Adds a 'rank' column (int).
    """
    result = df.copy()
    result["rank"] = result["cobbType"].str.split("_r").str[-1].astype(int)
    per_rater_max = (
        result.groupby(["image_id", "rater_name"])["rank"].max().reset_index()
    )
    min_common = (
        per_rater_max.groupby("image_id")["rank"].min().rename("min_common_rank")
    )
    return (
        result
        .merge(min_common, on="image_id")
        .query("rank <= min_common_rank")
        .drop(columns="min_common_rank")
    )


def compute_outlier_ranking(angles_df: pd.DataFrame) -> pd.DataFrame:
    """Inter-rater std per (image_id, cobbType, vertebra_upper, vertebra_lower).

    Groups angle measurements by the four key columns, computes std and mean
    across raters, and keeps only groups with ≥2 distinct raters. Returns a
    DataFrame sorted descending by std — the top rows are the most inconsistent
    angle measurements across annotators.

    Columns: image_id, img_type, cobbType, vertebra_upper, vertebra_lower,
             n_raters, std, mean_angle.
    """
    group_cols = ["image_id", "cobbType", "vertebra_upper", "vertebra_lower"]
    extra = {"region": ("region", "first")} if "region" in angles_df.columns else {}
    agg = (
        angles_df
        .dropna(subset=["angle"])
        .groupby(group_cols)
        .agg(
            n_raters=("rater_name", "nunique"),
            std=("angle", "std"),
            mean_angle=("angle", "mean"),
            img_type=("img_type", "first"),
            **extra,
        )
        .reset_index()
    )
    result = agg[agg["n_raters"] >= 2].copy()
    result["std"] = result["std"].fillna(0.0)
    return result.sort_values("std", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Subluxation / vertebra-shift detection (Cobb-segmental lines)
# ---------------------------------------------------------------------------

_SPINE_IDX = {lbl: i for i, lbl in enumerate(spine_labels)}


def build_cobb_segmental_lines_df(xmd: dict) -> pd.DataFrame:
    """One row per vertebra endplate line, deduplicated from Cobb-segmental angles.

    Each `angles` entry with cobbType == "Cobb-segmental" carries both
    upperLine (at vertebra_upper) and lowerLine (at vertebra_lower); adjacent
    entries share a line (this entry's lowerLine == the next entry's
    upperLine), so lines are collected into a dict keyed by vertebra label.

    Position (midpoint), orientation (angle mod 180°, folded into (-90, 90]),
    and length are all invariant to which endpoint is x1/x2 — this absorbs an
    accidental left/right endpoint swap during annotation without special-casing it.
    """
    lines: dict[str, dict] = {}
    for a in xmd.get("angles", []):
        if a.get("cobbType") != "Cobb-segmental":
            continue
        ca = a.get("cobbAngle", {})
        for label_key, line_key in (("vertebra_upper", "upperLine"), ("vertebra_lower", "lowerLine")):
            label = ca.get(label_key)
            line = a.get(line_key)
            if label and line and label not in lines:
                lines[label] = line

    rows = []
    for label, line in lines.items():
        dx = line["x2"] - line["x1"]
        dy = line["y2"] - line["y1"]
        angle_deg = np.degrees(np.arctan2(dy, dx))
        if angle_deg <= -90:
            angle_deg += 180
        elif angle_deg > 90:
            angle_deg -= 180
        rows.append({
            "vertebra": label,
            "spine_idx": _SPINE_IDX.get(label, -1),
            "x1": line["x1"], "y1": line["y1"], "x2": line["x2"], "y2": line["y2"],
            "mx": (line["x1"] + line["x2"]) / 2,
            "my": (line["y1"] + line["y2"]) / 2,
            "length": float(np.hypot(dx, dy)),
            "angle_deg": float(angle_deg),
        })
    columns = ["vertebra", "spine_idx", "x1", "y1", "x2", "y2", "mx", "my", "length", "angle_deg"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("spine_idx").reset_index(drop=True)


def _robust_poly_zresiduals(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Residuals of a robust (Huber) polynomial fit, normalized by the fit's scale.

    With few points and a near-perfect fit through all-but-one of them, the
    Huber MAD scale estimate can collapse toward zero, blowing up the one real
    residual to an absurd z-value. Floor the scale at a fraction of the data's
    own spread (unit-agnostic — works the same for pixel and degree values).
    """
    x_c = (x - x.mean()) / (x.std() if x.std() > 1e-9 else 1.0)
    design = np.vander(x_c, degree + 1)
    model = sm.RLM(y, design, M=sm.robust.norms.HuberT()).fit()
    y_range = y.max() - y.min()
    floor = 0.05 * y_range if y_range > 1e-9 else 1e-3
    scale = max(model.scale, floor)
    return (y - model.predict(design)) / scale


def compute_subluxation_scores(xmd: dict, degree: int = 3) -> pd.DataFrame:
    """Per-vertebra shift/rotation/foreshortening residuals vs. a robust harmonic fit.

    Fits a polynomial (degree adapted to chain length, see below) with a Huber
    M-estimator (robust to outliers skewing the reference curve) separately to
    midpoint-x, orientation angle, and line length, each as a function of
    vertebra position along the spine. Residuals are z-normalized by each
    fit's own robust scale, then combined into one euclidean severity score
    per vertebra line.

    Needs >= 4 Cobb-segmental lines (>=3 line pairs / segments). The
    polynomial degree is capped at `n - 3` so there are always >=2 residual
    degrees of freedom for a stable scale estimate — e.g. a 4-line chain gets
    a linear fit (degree 1), a 7+-line chain gets the full requested degree.
    Returns NaN scores when there are too few lines.
    """
    lines_df = build_cobb_segmental_lines_df(xmd)
    n = len(lines_df)
    if n < 4:
        for col in ("z_pos", "z_angle", "z_length", "score"):
            lines_df[col] = float("nan")
        return lines_df

    eff_degree = min(degree, n - 3)
    x = lines_df["spine_idx"].to_numpy(dtype=float)
    lines_df["z_pos"] = _robust_poly_zresiduals(x, lines_df["mx"].to_numpy(dtype=float), eff_degree)
    lines_df["z_angle"] = _robust_poly_zresiduals(x, lines_df["angle_deg"].to_numpy(dtype=float), eff_degree)
    lines_df["z_length"] = _robust_poly_zresiduals(x, lines_df["length"].to_numpy(dtype=float), eff_degree)
    lines_df["score"] = np.sqrt(
        lines_df["z_pos"] ** 2 + lines_df["z_angle"] ** 2 + lines_df["z_length"] ** 2
    )
    return lines_df


def build_subluxation_segment_df(xmd: dict, degree: int = 3) -> pd.DataFrame:
    """One row per Cobb-segmental angle with a combined shift score.

    Score per segment = max(score of its upper line, score of its lower line) —
    a segment is flagged as soon as either of its two endplate lines deviates
    from the expected harmonic spine profile.
    """
    lines_df = compute_subluxation_scores(xmd, degree=degree)
    score_by_vertebra = lines_df.set_index("vertebra")["score"]

    rows = []
    for a in xmd.get("angles", []):
        if a.get("cobbType") != "Cobb-segmental":
            continue
        ca = a.get("cobbAngle", {})
        vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
        s_u = score_by_vertebra.get(vu, np.nan)
        s_l = score_by_vertebra.get(vl, np.nan)
        rows.append({
            "vertebra_upper": vu,
            "vertebra_lower": vl,
            "angle": ca.get("angle"),
            "score_upper": s_u,
            "score_lower": s_l,
            "score": np.nanmax([s_u, s_l]) if pd.notna(s_u) or pd.notna(s_l) else np.nan,
        })
    return pd.DataFrame(rows)


def get_calibration_px(xmd: dict) -> float:
    """Pixel length of the image's calibration reference (one known real-world
    length, usually one vertebra's height/width). Used to turn raw pixel
    offsets into a zoom/resolution-independent unit ('vertebra widths')
    that's comparable across images. Prefers the actual drawn calibrationLine
    over the stored `calibration.length` scalar, falling back to the latter
    when the line is missing."""
    cl = xmd.get("stateObject", {}).get("calibrationLine")
    if cl:
        px = float(np.hypot(cl["x2"] - cl["x1"], cl["y2"] - cl["y1"]))
        if px > 1e-6:
            return px
    length = (xmd.get("calibration") or {}).get("length")
    return float(length) if length else np.nan


def compute_spline_line_features(xmd: dict, perp_length: float = 200.0) -> pd.DataFrame:
    """Per-vertebra residuals of the actual Cobb-segmental line vs. the line the
    node-reconstructed B-spline predicts at that vertebra — an independent
    reference curve (from stateObject.nodes) rather than a fit through the
    Cobb-segmental line data itself.

    For each vertebra, maps it onto the spline (same approach as the "Spline"
    cobbType overlay in outlier_app.py) and decomposes the offset between the
    actual line's midpoint and the spline point into:
    - spline_lateral_resid: offset along the expected line's own direction
      (a listhesis-style lateral/AP shift of the vertebra), in pixels
    - spline_tangent_resid: offset along the spine's tangent (up/down;
      mostly annotation noise, not expected to carry pathology signal), in pixels
    - spline_angle_resid: difference between the actual line's orientation
      and the spline-perpendicular's orientation, folded to (-90, 90]
    - spline_lateral_resid_norm / spline_tangent_resid_norm: the two pixel
      residuals divided by get_calibration_px(xmd), so a given offset means
      roughly the same anatomical fraction regardless of image zoom/resolution.
      NaN when no calibration reference is available for the image.

    NaN columns when there are too few nodes, or the node count doesn't match
    the labeled vertebra_range (nodes are mapped to labels positionally).
    """
    lines_df = build_cobb_segmental_lines_df(xmd)
    resid_cols = ["spline_lateral_resid", "spline_tangent_resid", "spline_angle_resid",
                  "spline_lateral_resid_norm", "spline_tangent_resid_norm"]
    if lines_df.empty:
        for c in resid_cols:
            lines_df[c] = pd.Series(dtype=float)
        return lines_df

    calib_px = get_calibration_px(xmd)

    nodes_raw = xmd.get("stateObject", {}).get("nodes", [])
    vr = xmd.get("stateObject", {}).get("vertebra_range", [0, 0])
    selected_vertebra = spine_labels[vr[0]: vr[1] + 1]
    spline = interpolate_bspline([(n["x"], n["y"]) for n in nodes_raw], num_steps=200) \
        if len(nodes_raw) >= 3 and len(nodes_raw) == len(selected_vertebra) else None

    if not spline:
        for c in resid_cols:
            lines_df[c] = np.nan
        return lines_df

    nodes_items = [NodeItem(x=n["x"], y=n["y"]) for n in nodes_raw]
    mapped = map_nodes_to_spline_and_keys(nodes_items, selected_vertebra, spline)
    lbl_to_idx = {m.label: m.spline_idx for m in mapped}

    rows = []
    for _, ln in lines_df.iterrows():
        row = ln.to_dict()
        idx = lbl_to_idx.get(ln["vertebra"])
        perp = get_perpendicular_lines_at_indices(spline, length=perp_length, indices=[idx]) if idx is not None else []
        if not perp:
            row.update({c: np.nan for c in resid_cols})
            rows.append(row)
            continue

        p = perp[0]
        sx, sy = spline[idx]
        perp_dx, perp_dy = p["x2"] - p["x1"], p["y2"] - p["y1"]
        perp_norm = np.hypot(perp_dx, perp_dy)
        perp_unit = np.array([perp_dx, perp_dy]) / perp_norm
        tangent_unit = np.array([-perp_unit[1], perp_unit[0]])

        d = np.array([ln["mx"] - sx, ln["my"] - sy])
        expected_angle = np.degrees(np.arctan2(perp_dy, perp_dx))
        if expected_angle <= -90:
            expected_angle += 180
        elif expected_angle > 90:
            expected_angle -= 180
        angle_resid = ln["angle_deg"] - expected_angle
        if angle_resid > 90:
            angle_resid -= 180
        elif angle_resid <= -90:
            angle_resid += 180

        lateral_resid = float(d @ perp_unit)
        tangent_resid = float(d @ tangent_unit)
        row.update({
            "spline_lateral_resid": lateral_resid,
            "spline_tangent_resid": tangent_resid,
            "spline_angle_resid": float(angle_resid),
            "spline_lateral_resid_norm": lateral_resid / calib_px if calib_px else np.nan,
            "spline_tangent_resid_norm": tangent_resid / calib_px if calib_px else np.nan,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def compute_spline_smoothing_deviation(xmd: dict, sigma: float = 1.0, num_steps: int = 200) -> pd.DataFrame:
    """Per-vertebra local 'kink' measure: how far a vertebra's point on the raw
    node-spline moves when the nodes are Gaussian-smoothed (sigma). A vertebra
    whose node was placed off the smooth trend of its neighbours shifts a lot
    under smoothing; one that's already on a smooth curve barely moves.

    This probes the node placement itself, independent of the Cobb-segmental
    lines used by compute_spline_line_features — a genuinely separate signal.

    Returns one row per vertebra with `smooth_dev_px` (raw pixel distance
    between the raw-spline point and the smoothed-spline point for that
    vertebra) and `smooth_dev_norm` (divided by get_calibration_px(xmd)).
    NaN when there are too few/mismatched nodes, same conditions as
    compute_spline_line_features.
    """
    nodes_raw = xmd.get("stateObject", {}).get("nodes", [])
    vr = xmd.get("stateObject", {}).get("vertebra_range", [0, 0])
    selected_vertebra = spine_labels[vr[0]: vr[1] + 1]
    cols = ["vertebra", "smooth_dev_px", "smooth_dev_norm"]

    if len(nodes_raw) < 3 or len(nodes_raw) != len(selected_vertebra):
        return pd.DataFrame(columns=cols)

    nodes_canvas = [(n["x"], n["y"]) for n in nodes_raw]
    spline_raw = interpolate_bspline(nodes_canvas, num_steps=num_steps)
    spline_smooth = interpolate_bspline(smooth_nodes(nodes_canvas, sigma=sigma), num_steps=num_steps)
    if not spline_raw or not spline_smooth:
        return pd.DataFrame(columns=cols)

    nodes_items = [NodeItem(x=x, y=y) for x, y in nodes_canvas]
    mapped_raw = {m.label: m.spline_idx for m in map_nodes_to_spline_and_keys(nodes_items, selected_vertebra, spline_raw)}
    mapped_smooth = {m.label: m.spline_idx for m in map_nodes_to_spline_and_keys(nodes_items, selected_vertebra, spline_smooth)}

    calib_px = get_calibration_px(xmd)
    rows = []
    for label in selected_vertebra:
        i_raw, i_smooth = mapped_raw.get(label), mapped_smooth.get(label)
        if i_raw is None or i_smooth is None:
            rows.append({"vertebra": label, "smooth_dev_px": np.nan, "smooth_dev_norm": np.nan})
            continue
        dev = float(np.hypot(spline_raw[i_raw][0] - spline_smooth[i_smooth][0],
                              spline_raw[i_raw][1] - spline_smooth[i_smooth][1]))
        rows.append({
            "vertebra": label,
            "smooth_dev_px": dev,
            "smooth_dev_norm": dev / calib_px if calib_px else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subluxation classifier (segment-level, trained on subluxation_app.py feedback)
#
# Validated by session-time cross-validation (GroupKFold by image, 5-fold):
# RandomForest on these 6 features reaches AUC ~0.71 / AP ~0.18 (baseline
# ~0.08) predicting whether a human labeled a given Cobb-segmental segment as
# a subluxation. Moderate, real, but not diagnostic-grade — treat the output
# as a triage/ranking aid, not a verdict.
# ---------------------------------------------------------------------------

SUBLUXATION_MODEL_FEATURES = [
    "spl_ang_max", "spl_lat_n_max", "z_length_max", "z_pos_max", "z_angle_max", "smooth_dev_max",
]


def _parse_positive_segments(levels_str, all_segments: list[str]) -> set[str]:
    """A feedback row's sublux_levels cell -> set of 'VU/VL' segment strings.

    Handles both the current 'VU/VL' notation (segment_options() in
    subluxation_app.py) and the older single-vertebra-name notation used
    before that — a lone vertebra name is expanded to both segments
    touching it, since it's ambiguous which one was meant.
    """
    if pd.isna(levels_str) or not str(levels_str).strip():
        return set()
    positives = set()
    for tok in str(levels_str).split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if "/" in tok:
            positives.add(tok)
        else:
            for seg in all_segments:
                vu, vl = seg.split("/")
                if tok in (vu, vl):
                    positives.add(seg)
    return positives


def _segment_features(merged: pd.DataFrame, vu: str, vl: str) -> dict:
    """SUBLUXATION_MODEL_FEATURES for one segment from a per-vertebra table
    that already has z_pos/z_angle/z_length, spline_angle_resid,
    spline_lateral_resid_norm, and smooth_dev_norm columns, indexed by
    vertebra label."""
    u, l = merged.loc[vu], merged.loc[vl]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "spl_ang_max": np.nanmax([abs(u["spline_angle_resid"]), abs(l["spline_angle_resid"])]),
            "spl_lat_n_max": np.nanmax([abs(u["spline_lateral_resid_norm"]), abs(l["spline_lateral_resid_norm"])]),
            "z_length_max": np.nanmax([abs(u["z_length"]), abs(l["z_length"])]),
            "z_pos_max": np.nanmax([abs(u["z_pos"]), abs(l["z_pos"])]),
            "z_angle_max": np.nanmax([abs(u["z_angle"]), abs(l["z_angle"])]),
            "smooth_dev_max": np.nanmax([u["smooth_dev_norm"], l["smooth_dev_norm"]]),
        }


def _merged_vertebra_features(xmd: dict) -> pd.DataFrame:
    """Per-vertebra table combining the three feature sources, indexed by
    vertebra label — shared by build_segment_training_table() and
    compute_segment_predictions() so they can't drift apart."""
    z_df = compute_subluxation_scores(xmd)
    spl_df = compute_spline_line_features(xmd)
    dev_df = compute_spline_smoothing_deviation(xmd)
    return (
        z_df.merge(spl_df[["vertebra", "spline_angle_resid", "spline_lateral_resid_norm"]],
                   on="vertebra", how="left")
            .merge(dev_df[["vertebra", "smooth_dev_norm"]], on="vertebra", how="left")
            .set_index("vertebra")
    )


def build_segment_training_table(feedback_df: pd.DataFrame, df4d: pd.DataFrame) -> pd.DataFrame:
    """One row per (labeled measurement x Cobb-segmental segment) with
    SUBLUXATION_MODEL_FEATURES and a binary `label` for train_subluxation_classifier().

    label=1 for segments explicitly listed in a 'Subluxation vorhanden' row's
    sublux_levels; every other segment (including all segments of a 'keine
    Subluxation' row) is label=0.
    """
    rows = []
    for _, r in feedback_df.iterrows():
        m = df4d[(df4d["id"] == r["image_id"]) & (df4d["rater_name"] == r["rater_name"])]
        if m.empty:
            continue
        xmd = m.iloc[0]["xray_measurement_data"]
        merged = _merged_vertebra_features(xmd)

        segments = []
        for a in xmd.get("angles", []):
            if a.get("cobbType") != "Cobb-segmental":
                continue
            ca = a.get("cobbAngle", {})
            vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
            if vu and vl:
                segments.append((vu, vl))
        all_segment_strs = [f"{vu}/{vl}" for vu, vl in segments]
        positives = (
            _parse_positive_segments(r["sublux_levels"], all_segment_strs)
            if r["sublux_label"] == "Subluxation vorhanden" else set()
        )

        for vu, vl in segments:
            seg_str = f"{vu}/{vl}"
            if vu not in merged.index or vl not in merged.index:
                continue
            row = {"image_id": r["image_id"], "rater_name": r["rater_name"], "segment": seg_str,
                   "label": 1 if seg_str in positives else 0}
            row.update(_segment_features(merged, vu, vl))
            rows.append(row)
    return pd.DataFrame(rows)


def train_subluxation_classifier(training_df: pd.DataFrame):
    """Fits the RandomForest validated by session-time cross-validation
    (AUC ~0.71) on all available labeled segments. Returns (model, scaler,
    yellow_threshold, red_threshold) — the two thresholds split predicted
    probabilities into green/yellow/red at the ~80th/~93rd percentile of the
    training set's own predictions (roughly matching the ~8% true positive
    rate), for consistent traffic-light coloring in the UI."""
    X = training_df[SUBLUXATION_MODEL_FEATURES]
    mask = X.notna().all(axis=1)
    X, y = X[mask], training_df.loc[mask, "label"]
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=0)
    model.fit(scaler.transform(X), y)
    proba = model.predict_proba(scaler.transform(X))[:, 1]
    yellow_threshold = float(np.quantile(proba, 0.80))
    red_threshold = float(np.quantile(proba, 0.93))
    return model, scaler, yellow_threshold, red_threshold


def compute_segment_predictions(xmd: dict, model, scaler) -> pd.DataFrame:
    """One row per Cobb-segmental segment in xmd: predicted subluxation
    probability plus the segment's upper/lower line dicts (for drawing).
    proba is NaN when a feature is unavailable (e.g. no calibration line, or
    too few nodes for the spline)."""
    merged = _merged_vertebra_features(xmd)

    rows = []
    for a in xmd.get("angles", []):
        if a.get("cobbType") != "Cobb-segmental":
            continue
        ca = a.get("cobbAngle", {})
        vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
        upper_line, lower_line = a.get("upperLine"), a.get("lowerLine")
        if not (vu and vl and upper_line and lower_line):
            continue
        if vu not in merged.index or vl not in merged.index:
            continue
        feats = _segment_features(merged, vu, vl)
        vec = pd.DataFrame([feats], columns=SUBLUXATION_MODEL_FEATURES)
        proba = np.nan if vec.isna().any().any() else float(model.predict_proba(scaler.transform(vec))[0, 1])
        rows.append({
            "vertebra_upper": vu, "vertebra_lower": vl,
            "upperLine": upper_line, "lowerLine": lower_line,
            "proba": proba,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subluxation v2 — smoothed-spline distance features
# ---------------------------------------------------------------------------

V2_FEATURES = [
    # All features are gradients (differences between adjacent vertebrae),
    # invariant to absolute position, global curvature, rotation, and scale.
    # *_delta_above is NaN for the topmost segment (no neighbour above);
    # imputed with 0 ("no change from above") at fit/predict time.
    "node_delta_above",  # node_dist(VU) − node_dist(VU_prev)
    "node_delta_below",  # node_dist(VU) − node_dist(VL)
    "spike_score",       # node_dist(VU) − mean(neighbours)
    "seg_delta_above",   # same for endplate line midpoints
    "seg_delta_below",
]

# Columns imputed with 0 for topmost segments (no neighbour above).
_V2_IMPUTE_ZERO = ["node_delta_above", "seg_delta_above"]


def _wrap90(a: float) -> float:
    """Fold angle difference into (−90, 90] to handle the periodicity of undirected lines."""
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    return a


def compute_smoothed_spline_distances(xmd: dict, sigma: float = 2.0, num_steps: int = 300) -> pd.DataFrame:
    """Per-vertebra distances from actual measurements to an end-pinned smoothed spline.

    node_dist_norm : |vertebra node → nearest point on smoothed spline| / calib_px
    seg_dist_norm  : |endplate line midpoint → nearest point on smoothed spline| / calib_px

    The smoothed spline is built from Gaussian-smoothed nodes with endpoints
    pinned to their original positions (smooth_nodes_pinned), giving a reference
    curve that captures the overall spinal shape without drifting at chain ends.
    Returns an empty DataFrame when there are too few or mismatched nodes.
    """
    nodes_raw = xmd.get("stateObject", {}).get("nodes", [])
    vr = xmd.get("stateObject", {}).get("vertebra_range", [0, 0])
    selected_vertebra = spine_labels[vr[0]: vr[1] + 1]
    cols = ["vertebra", "node_dist_px", "node_dist_norm",
            "seg_dist_px", "seg_dist_norm", "angle_deg"]

    if len(nodes_raw) < 3 or len(nodes_raw) != len(selected_vertebra):
        return pd.DataFrame(columns=cols)

    nodes_canvas = [(n["x"], n["y"]) for n in nodes_raw]
    smoothed = smooth_nodes_pinned(nodes_canvas, sigma=sigma)
    spline_smooth = interpolate_bspline(smoothed, num_steps=num_steps)
    if not spline_smooth:
        return pd.DataFrame(columns=cols)

    spline_arr = np.array(spline_smooth)
    calib_px = get_calibration_px(xmd)

    lines_df = build_cobb_segmental_lines_df(xmd)
    if not lines_df.empty:
        seg_info = lines_df.set_index("vertebra")[["mx", "my", "angle_deg"]]
    else:
        seg_info = pd.DataFrame(columns=["mx", "my", "angle_deg"])

    rows = []
    for label, node in zip(selected_vertebra, nodes_canvas):
        node_pt = np.array(node)
        node_dist_px = float(np.hypot(spline_arr[:, 0] - node_pt[0],
                                      spline_arr[:, 1] - node_pt[1]).min())

        seg_dist_px = np.nan
        angle_deg = np.nan
        if label in seg_info.index:
            info = seg_info.loc[label]
            seg_dist_px = float(np.hypot(spline_arr[:, 0] - info["mx"],
                                         spline_arr[:, 1] - info["my"]).min())
            angle_deg = float(info["angle_deg"])

        rows.append({
            "vertebra": label,
            "node_dist_px": node_dist_px,
            "node_dist_norm": node_dist_px / calib_px if calib_px else np.nan,
            "seg_dist_px": seg_dist_px,
            "seg_dist_norm": seg_dist_px / calib_px if calib_px else np.nan,
            "angle_deg": angle_deg,
        })
    return pd.DataFrame(rows, columns=cols)


def _v2_segment_features(dist_df: pd.DataFrame, vu: str, vl: str) -> dict:
    """Compute V2_FEATURES for segment VU/VL.

    dist_df must be indexed by vertebra label, rows in spine order (cranial→caudal),
    as returned by compute_smoothed_spline_distances(...).set_index('vertebra').

    All features are gradients (differences between adjacent vertebrae), not
    absolute distances. This makes them invariant to global spinal curvature:
    a curved spine where all nodes are equally far from the smoothed reference
    produces near-zero gradient features throughout.
    """
    def _get(col: str, label: str) -> float:
        return dist_df.at[label, col] if label in dist_df.index else np.nan

    nd_u   = _get("node_dist_norm", vu)
    nd_l   = _get("node_dist_norm", vl)
    sd_u   = _get("seg_dist_norm",  vu)
    sd_l   = _get("seg_dist_norm",  vl)
    ang_u  = _get("angle_deg",      vu)
    ang_l  = _get("angle_deg",      vl)

    # Vertebra immediately above VU in the measured chain (spine-ordered index)
    labels = list(dist_df.index)
    vu_pos = labels.index(vu) if vu in labels else -1
    vu_prev = labels[vu_pos - 1] if vu_pos > 0 else None
    nd_above  = _get("node_dist_norm", vu_prev) if vu_prev else np.nan
    sd_above  = _get("seg_dist_norm",  vu_prev) if vu_prev else np.nan
    ang_above = _get("angle_deg",      vu_prev) if vu_prev else np.nan

    node_nn = [v for v in (nd_above, nd_l)  if not np.isnan(v)]
    ang_nn  = [v for v in (ang_above, ang_l) if not np.isnan(v)]

    spike_score = nd_u - float(np.mean(node_nn)) if node_nn else np.nan
    angle_spike = _wrap90(ang_u - float(np.mean(ang_nn))) if ang_nn else np.nan

    return {
        "node_delta_above":  nd_u - nd_above,
        "node_delta_below":  nd_u - nd_l,
        "spike_score":       spike_score,
        "seg_delta_above":   sd_u - sd_above,
        "seg_delta_below":   sd_u - sd_l,
        "angle_delta_above": _wrap90(ang_u - ang_above),
        "angle_delta_below": _wrap90(ang_u - ang_l),
        "angle_spike":       angle_spike,
    }


def build_v2_training_table(feedback_df: pd.DataFrame, df4d: pd.DataFrame,
                            sigma: float = 2.0) -> pd.DataFrame:
    """One row per (labeled measurement × Cobb-segmental segment) with V2_FEATURES and binary label.

    Augmentation: for every original row a reflected copy is added with negated
    angle features (_V2_ANGLE_FEATURES). This makes the model invariant to
    left-right image orientation without requiring image-level preprocessing.
    The reflected copy keeps the same image_id so GroupKFold keeps originals
    and their mirrors in the same fold, preventing data leakage.
    """
    rows = []
    for _, r in feedback_df.iterrows():
        m = df4d[(df4d["id"] == r["image_id"]) & (df4d["rater_name"] == r["rater_name"])]
        if m.empty:
            continue
        xmd = m.iloc[0]["xray_measurement_data"]
        dist_df = compute_smoothed_spline_distances(xmd, sigma=sigma).set_index("vertebra")
        if dist_df.empty:
            continue

        segments = [(a["cobbAngle"]["vertebra_upper"], a["cobbAngle"]["vertebra_lower"])
                    for a in xmd.get("angles", [])
                    if a.get("cobbType") == "Cobb-segmental"
                    and a.get("cobbAngle", {}).get("vertebra_upper")
                    and a.get("cobbAngle", {}).get("vertebra_lower")]
        all_seg_strs = [f"{vu}/{vl}" for vu, vl in segments]
        positives = (
            _parse_positive_segments(r["sublux_levels"], all_seg_strs)
            if r["sublux_label"] == "Subluxation vorhanden" else set()
        )

        for vu, vl in segments:
            if vu not in dist_df.index or vl not in dist_df.index:
                continue
            feats = _v2_segment_features(dist_df, vu, vl)
            base = {"image_id": r["image_id"], "rater_name": r["rater_name"],
                    "segment": f"{vu}/{vl}", "label": 1 if f"{vu}/{vl}" in positives else 0,
                    **feats}
            rows.append({**base, "augmented": False})

    return pd.DataFrame(rows)


def train_v2_classifier(training_df: pd.DataFrame):
    """Logistic regression on V2_FEATURES with balanced class weights.

    Returns (model, scaler, yellow_threshold, red_threshold).
    Thresholds split predicted probabilities at the 80th/93rd percentile of
    training predictions — consistent with the v1 traffic-light convention.
    """
    from sklearn.linear_model import LogisticRegression
    X = training_df[V2_FEATURES].copy()
    X[_V2_IMPUTE_ZERO] = X[_V2_IMPUTE_ZERO].fillna(0.0)
    mask = X.notna().all(axis=1)
    X, y = X[mask], training_df.loc[mask, "label"]
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
    model.fit(scaler.transform(X), y)
    proba = model.predict_proba(scaler.transform(X))[:, 1]
    return model, scaler, float(np.quantile(proba, 0.80)), float(np.quantile(proba, 0.93))


def compute_v2_segment_predictions(xmd: dict, model, scaler, sigma: float = 2.0) -> pd.DataFrame:
    """One row per Cobb-segmental segment: V2_FEATURES + proba + line dicts for drawing."""
    dist_df = compute_smoothed_spline_distances(xmd, sigma=sigma)
    if dist_df.empty:
        return pd.DataFrame()
    dist_df = dist_df.set_index("vertebra")

    rows = []
    for a in xmd.get("angles", []):
        if a.get("cobbType") != "Cobb-segmental":
            continue
        ca = a.get("cobbAngle", {})
        vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
        upper_line, lower_line = a.get("upperLine"), a.get("lowerLine")
        if not (vu and vl and upper_line and lower_line):
            continue
        if vu not in dist_df.index or vl not in dist_df.index:
            continue
        feats = _v2_segment_features(dist_df, vu, vl)
        vec = pd.DataFrame([feats], columns=V2_FEATURES)
        proba = (float(model.predict_proba(scaler.transform(vec))[0, 1])
                 if not vec.isna().any().any() else np.nan)
        rows.append({"segment": f"{vu}/{vl}", "vertebra_upper": vu, "vertebra_lower": vl,
                     "upper_line": upper_line, "lower_line": lower_line, **feats, "proba": proba})
    return pd.DataFrame(rows)
