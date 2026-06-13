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

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from dotmap import DotMap
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import f as f_dist

from bspline_python import (
    interpolate_bspline,
    smooth_nodes,
    calculate_tangents,
    get_perpendicular_lines_at_indices,
    map_nodes_to_spline_and_keys,
    generate_COBB_angles_from_tangents,
    find_tangent_extrema_indices,
    Line,
    calculate_angle,
    spine_labels,
    spine_regions,
)

# Pre-built mapping: vertebra label -> spine region name
label_to_region: dict[str, str] = {}
for _region, (_start, _end) in spine_regions.items():
    for _i in range(_start, _end + 1):
        if _i < len(spine_labels):
            label_to_region[spine_labels[_i]] = _region


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
    """Cobb angles at spline inflection points using node-level tangent extrema.

    Node-level tangents (8-10 pts) detect Richtungswechsel without smoothing;
    perpendicular lines are still computed on the dense spline for accuracy.
    Angles below min_angle degrees are discarded; survivors are re-ranked so
    'inflection_r0' is always the first clinically significant angle from the top.

    smooth_sigma > 0 applies Gaussian smoothing to nodes before spline fitting.
    cobbType = 'inflection_r0', 'inflection_r1', … (rank after min_angle filter)
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

    # Detect inflection points from node-level tangents (no smoothing needed)
    node_tangents = calculate_tangents(nodes, [], length=50)
    node_angles = [t.angle for t in node_tangents]
    extrema_node_idx = find_tangent_extrema_indices(node_angles, smoothing_window=1)

    # Map node-level extrema to exact dense-spline positions
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

    node_tangents = calculate_tangents(nodes, [], length=50)
    node_angles = [t.angle for t in node_tangents]
    extrema_node_idx = find_tangent_extrema_indices(node_angles, smoothing_window=1)

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
        all_angles = list(xmd.get("angles", []))
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
    agg = (
        angles_df
        .dropna(subset=["angle"])
        .groupby(group_cols)
        .agg(
            n_raters=("rater_name", "nunique"),
            std=("angle", "std"),
            mean_angle=("angle", "mean"),
            img_type=("img_type", "first"),
        )
        .reset_index()
    )
    result = agg[agg["n_raters"] >= 2].copy()
    result["std"] = result["std"].fillna(0.0)
    return result.sort_values("std", ascending=False).reset_index(drop=True)
