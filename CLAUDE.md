# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python data analysis project for spine X-ray measurements. It fetches data from the **xray-annotator** web app via REST API, reconstructs B-Spline curves from manually placed vertebra nodes, computes Cobb angles, detects labeling errors, and visualizes results. Analysis is done in Jupyter Notebooks; reusable logic lives in Python modules.

## API Access

The backend is `https://xray-api.renecol.org`. All requests require a Bearer token obtained from https://xray-annotator.renecol.org (Account info → Copy token). Tokens are short-lived; users paste them directly into notebooks.

The primary endpoint is:
```
GET /api/imagedata/?detailed=True&complete=True
```

See `docs/api_imagedata_structure.md` for the full documented response schema.

## Running Notebooks

```bash
jupyter notebook          # start Jupyter, then open a .ipynb file
jupyter lab               # alternative UI
```

There are no build steps, test suites, or linters. Development is entirely notebook-driven.

## Key Files

| File | Purpose |
|---|---|
| `bspline_python.py` | Core library – all spline math, Cobb angle computation, plotting |
| `annotation_analysis.py` | Inter-rater reliability module – DataFrame builders, ICC computation, inflection-point angles, img_type helpers |
| `reconstructing_spline.ipynb` | Main analysis notebook: API fetch → data cleaning → label-error detection → angle reconstruction |
| `annotation_correctness.ipynb` | Inter-rater reliability study: node ICC, label-error correction, Cobb-angle ICC by technique, inflection-point method |
| `comparing_angles.ipynb` | Statistical comparison of annotator-computed vs. reconstructed Cobb angles |
| `user_activity.ipynb` | Annotation activity statistics per user/image |
| `spine_interactive_viewer.py` | `SpineMeasurementViewer` class for interactive ipywidgets-based browsing |
| `spline_optimization/spline_optimization.py` | `SplineOptimizer` – grid-search over spline parameters to minimize inter-measurement deviation |

## Architecture

### Data flow

```
API (/api/imagedata/)
  └─ raw JSON (792 images, up to 4 measure_datasets each)
       └─ reconstructing_spline.ipynb
            ├─ df4d  – one row per measurement, `xray_measurement_data` column holds nested dict
            ├─ label error detection via k-Means on node (x,y) positions
            │    └─ data/label_errors.json  (shift / empty classification)
            ├─ df4d_corrected  – label-shift-corrected copy
            ├─ df5c  – long format: one row per angle (raw + reconstructed angles combined)
            └─ data/*.pkl  – cached intermediates for downstream notebooks
```

### `bspline_python.py` – function groups

- **Spline:** `interpolate_bspline()` — cubic B-spline (x-axis) + PCHIP (y-axis) hybrid; mirrors the TypeScript implementation in xray-annotator
- **Geometry:** `calculate_tangents()`, `get_perpendicular_lines_at_indices()`, `make_perpendicular_line_at_line_center()`
- **Vertebra mapping:** `map_nodes_to_spline_and_keys()` — maps each node to the nearest spline point and assigns a spine label from `spine_labels`
- **Cobb angles:** `generate_COBB_angles_from_tangents()` — modes: `segmental`, `regional_S1`, `regional_L5`
- **Inflection detection:** `find_tangent_extrema_indices()` — local minima + maxima of tangent angles (used by `getInflectionCobbAngles`)
- **Plotting:** `plot_spline_and_lines()`, `plot_spline_and_lines_multiple()` (matplotlib, `invert_yaxis=True` because pixel y-axis is top-down)

### `annotation_analysis.py` – function groups

- **DataFrame builders:** `build_df4d()`, `build_nodes_df()`, `build_angles_df()`, `build_vertebra_range_df()`
- **Label-error pipeline:** `detect_label_errors()`, `apply_label_corrections()`
- **Angle computation:** `getReconstructedCobbAngles()`, `getInflectionCobbAngles(xmd, min_angle)` — inflection-point Cobb angles at node-level tangent extrema; `cobbType` = `inflection_r0`, `inflection_r1`, …
- **ICC helpers:** `compute_icc()` — two-way random ICC(2,1) via pingouin; `compute_icc_by_img_type(df, cobb_types, img_types)` — returns pivot DataFrame (index=img_type, columns=cobbType); `compute_std_by_label()`
- **Inflection filtering:** `filter_inflection_max_common_rank(df)` — keeps only ranks that all raters of each image share (max-common-rank strategy)

### Spine label system

26 labels indexed 0–25:
```python
spine_labels = ['C0'…'C7', 'T1'…'T12', 'L1'…'L5', 'S1']
```
`stateObject.vertebra_range` in the API response is a `[start_index, end_index]` pair into this array. Typical measurements run T10 (index 17) → S1 (index 25).

### Label-error correction pattern

Measurements where a user accidentally set nodes one or more vertebrae off are detected by:
1. k-Means clustering of all node `(x, y)` positions with `k = spine_length`
2. Assigning cluster centers → sorted spine labels by y-coordinate
3. Comparing assigned labels to original labels; majority-vote shift is applied via `correct_measurement_label_shift()`

### Data files (`data/`)

| File | Contents |
|---|---|
| `df4d_spine_measurements.pkl` | Raw flattened measurements DataFrame |
| `df4d_corrected_spine_measurements.pkl` | Label-corrected version |
| `df5c_with_reconstructed_angles.pkl` | Long-format angles (raw + reconstructed) |
| `df5c_corrected_with_reconstructed_angles.pkl` | Corrected version |
| `label_errors.json` | Per-measurement shift/empty error classification |
| `df_keys.json` | Image IDs, UUIDs, img_type, img URLs |

Pickle files are the primary inter-notebook cache format. Do not commit them (`.gitignore` excludes `data/` and `output/`).

## Dependencies

The project uses a conda/venv environment with: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `dotmap`, `requests`, `ipywidgets`, `jupyter`, `pingouin` (ICC-Berechnung; `pip install pingouin`).

`dotmap` is used throughout for attribute-style access on dicts (e.g. `DotMap(node).x`).
