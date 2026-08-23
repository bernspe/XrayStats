"""
Image-Group Vergleich — Cobb-Winkel-Vergleich zwischen img_types innerhalb
Patient.image_groups (z. B. Xray-LWS-ap vs. MRI-LWS-cor).

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run image_group_comparison_app.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

from bspline_python import spine_labels, convertMeasurementBorderToSpineRegion

warnings.filterwarnings("ignore")

API_URL = "https://xray-api.renecol.org"
CACHE_FILE = Path("data/image_groups_cache.json")

# Default pairs to compare within a group (x-axis type, y-axis type)
DEFAULT_PAIRS = [
    ("Xray-LWS-ap", "MRI-LWS-cor"),
    ("Xray-LWS-lat", "MRI-LWS-sag"),
    ("Xray-LWS-recl","Xray-LWS-incl"),
]

COBB_TYPES = ["Spline", "Cobb-regional", "Cobb-segmental"]

REGIONAL_REGIONS = ["MT", "TL", "U-LU", "L-LU", "LU"]
SEGMENTAL_REGIONS = ["T11/T12", "T12/L1", "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

# Segmental pairs as (upper, lower) for lookup
_SEG_PAIRS: dict[tuple[str, str], str] = {
    ("T11", "T12"): "T11/T12", ("T12", "L1"): "T12/L1", ("L1", "L2"): "L1/L2",
    ("L2", "L3"): "L2/L3", ("L3", "L4"): "L3/L4", ("L4", "L5"): "L4/L5",
    ("L5", "S1"): "L5/S1",
}


def angle_region(upper: str, lower: str, cobb_type: str) -> str | None:
    """Map (vertebra_upper, vertebra_lower, cobbType) to a display region label."""
    if "segmental" in cobb_type.lower():
        return _SEG_PAIRS.get((upper, lower), f"{upper}/{lower}")
    # Regional / Spline: detect LU (spans both U-LU [20,22] and L-LU [22,25])
    try:
        up_idx = spine_labels.index(upper)
        lo_idx = spine_labels.index(lower)
    except ValueError:
        return None
    if up_idx <= 20 and lo_idx >= 25:  # L1 or higher → S1: full lumbar lordosis
        return "LU"
    try:
        return convertMeasurementBorderToSpineRegion(upper, lower)
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Data fetch & cache
# ──────────────────────────────────────────────────────────────────────────────

def fetch_from_api(token: str) -> list[dict]:
    resp = requests.get(
        url=f"{API_URL}/api/imagedata/",
        headers={"Authorization": f"Bearer {token}"},
        params={"detailed": True, "complete": True},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def save_cache(data: list[dict]) -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)


def load_cache() -> list[dict] | None:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_angles_from_image(image: dict) -> list[dict]:
    """Returns list of {cobbType, region, vertebra_upper, vertebra_lower, angle} per rater."""
    rows = []
    for md in image.get("measure_dataset", []):
        xmd = (md.get("data") or {}).get("xray_measurement_data") or {}
        for a in xmd.get("angles", []):
            ca = a.get("cobbAngle", {})
            angle_val = ca.get("angle")
            if angle_val is None:
                continue
            ct = a.get("cobbType", "")
            upper = ca.get("vertebra_upper", "")
            lower = ca.get("vertebra_lower", "")
            rows.append({
                "cobbType": ct,
                "region": angle_region(upper, lower, ct),
                "vertebra_upper": upper,
                "vertebra_lower": lower,
                "angle": float(angle_val),
                "rater": (md.get("user") or {}).get("username", ""),
            })
    return rows


def images_by_uuid(data: list[dict]) -> dict[str, dict]:
    return {img["uuid"]: img for img in data}


def build_group_index(data: list[dict]) -> dict:
    """
    Returns {group_name: [{patient_id, patient_uuid, group_entry, images_in_group}]}

    image_groups on Patient is a JSON list. Supported formats:
      A) list of strings            ["GroupA", "GroupB"]
      B) list of {name, images}     [{"name": "G1", "images": ["uuid1","uuid2"]}, ...]
      C) list of {name, img_types}  [{"name": "G1", "img_types": ["Xray-LWS-ap","MRI-LWS-cor"]}, ...]
    """
    by_uuid = images_by_uuid(data)
    index: dict[str, list] = {}

    for img in data:
        rp = img.get("referring_patient") or {}
        groups = rp.get("image_groups") or []
        if not groups:
            continue

        patient_uuid = rp.get("uuid", "")
        patient_id = rp.get("patient_id", "")

        for entry in groups:
            # Normalise to {name, image_uuids, img_types}
            if isinstance(entry, str):
                group_name = entry
                image_uuids = []
                img_types = []
            elif isinstance(entry, dict):
                group_name = entry.get("name") or entry.get("group") or str(entry)
                image_uuids = entry.get("images") or entry.get("image_uuids") or []
                img_types = entry.get("img_types") or []
            else:
                continue

            index.setdefault(group_name, {})
            if patient_uuid not in index[group_name]:
                index[group_name][patient_uuid] = {
                    "patient_id": patient_id,
                    "patient_uuid": patient_uuid,
                    "image_uuids": set(image_uuids),
                    "img_types": set(img_types),
                    "images": {},  # img_type -> list of image dicts
                }
            # Add the current image to this patient's group entry if it belongs
            current_uuid = img.get("uuid", "")
            current_type = img.get("img_type", "")
            pat_entry = index[group_name][patient_uuid]
            # belongs if explicitly listed or img_type listed or no filter
            if (current_uuid in pat_entry["image_uuids"]
                    or current_type in pat_entry["img_types"]
                    or (not pat_entry["image_uuids"] and not pat_entry["img_types"])):
                pat_entry["images"].setdefault(current_type, []).append(img)

    # Convert inner dicts → lists for easier iteration
    return {g: list(pats.values()) for g, pats in index.items()}


def mean_angles_for_images(images: list[dict]) -> pd.DataFrame:
    """Mean angle + unique rater count across all images of the same type for one patient."""
    all_rows = []
    for img in images:
        all_rows.extend(extract_angles_from_image(img))
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).dropna(subset=["region"])
    grp = df.groupby(["cobbType", "region"])
    result = grp["angle"].mean().reset_index().rename(columns={"angle": "mean_angle"})
    result["n_raters"] = grp["rater"].nunique().values
    return result


def pair_angles(df_a: pd.DataFrame, df_b: pd.DataFrame,
                type_a: str, type_b: str,
                sel_cobb_type: str | None = None) -> pd.DataFrame:
    """Inner join on (cobbType, region), optionally filtered by cobbType."""
    if df_a.empty or df_b.empty:
        return pd.DataFrame()
    if sel_cobb_type:
        df_a = df_a[df_a["cobbType"] == sel_cobb_type]
        df_b = df_b[df_b["cobbType"] == sel_cobb_type]
    return df_a.merge(df_b, on=["cobbType", "region"],
                      suffixes=(f"_{type_a}", f"_{type_b}"))


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    r, p_r = stats.pearsonr(a, b)
    _, p_t = stats.ttest_rel(a, b)
    n = len(a)
    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_upper": mean_diff + 1.96 * std_diff,
        "loa_lower": mean_diff - 1.96 * std_diff,
        "pearson_r": float(r),
        "p_pearson": float(p_r),
        "p_ttest": float(p_t),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

# Stable region order for x-axis
_REGION_ORDER = REGIONAL_REGIONS + SEGMENTAL_REGIONS

COLOR_A = "#2171b5"
COLOR_B = "#e6550d"


def _region_sort_key(r: str) -> int:
    return _REGION_ORDER.index(r) if r in _REGION_ORDER else 999


def region_bar_plot(data_by_region: dict, type_a: str, type_b: str) -> plt.Figure:
    """Grouped bar chart: mean ± SD per region for each img_type."""
    regions = sorted(data_by_region.keys(), key=_region_sort_key)
    x = np.arange(len(regions))
    width = 0.35

    means_a = [np.mean(data_by_region[r]["a"]) for r in regions]
    sds_a   = [np.std(data_by_region[r]["a"], ddof=1) if len(data_by_region[r]["a"]) > 1 else 0 for r in regions]
    means_b = [np.mean(data_by_region[r]["b"]) for r in regions]
    sds_b   = [np.std(data_by_region[r]["b"], ddof=1) if len(data_by_region[r]["b"]) > 1 else 0 for r in regions]
    ns      = [len(data_by_region[r]["a"]) for r in regions]

    fig, ax = plt.subplots(figsize=(max(6, len(regions) * 1.4), 4))
    bars_a = ax.bar(x - width / 2, means_a, width, yerr=sds_a, capsize=4,
                    color=COLOR_A, alpha=0.8, label=type_a, error_kw={"lw": 1.2})
    bars_b = ax.bar(x + width / 2, means_b, width, yerr=sds_b, capsize=4,
                    color=COLOR_B, alpha=0.8, label=type_b, error_kw={"lw": 1.2})
    for xi, n in zip(x, ns):
        ax.text(xi, 0, f"n={n}", ha="center", va="bottom", fontsize=7, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Cobb-Winkel [°]")
    ax.set_title("Mittelwert ± SD je Region")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def bland_altman_by_region(data_by_region: dict, type_a: str, type_b: str) -> plt.Figure:
    """Bland-Altman, Punkte farbig nach Region."""
    regions = sorted(data_by_region.keys(), key=_region_sort_key)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(regions)))
    color_map = dict(zip(regions, colors))

    all_a, all_b, all_regions = [], [], []
    for r in regions:
        all_a.extend(data_by_region[r]["a"])
        all_b.extend(data_by_region[r]["b"])
        all_regions.extend([r] * len(data_by_region[r]["a"]))

    arr_a, arr_b = np.array(all_a), np.array(all_b)
    diff = arr_a - arr_b
    mean_vals = (arr_a + arr_b) / 2
    s = compute_stats(arr_a, arr_b)

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in regions:
        mask = np.array([x == r for x in all_regions])
        ax.scatter(mean_vals[mask], diff[mask], color=color_map[r],
                   label=r, alpha=0.8, s=50, edgecolors="none")

    ax.axhline(s["mean_diff"], color="#333", lw=1.5,
               label=f"Bias {s['mean_diff']:.2f}°")
    ax.axhline(s["loa_upper"], color="#333", lw=1, ls="--",
               label=f"+1.96 SD  {s['loa_upper']:.2f}°")
    ax.axhline(s["loa_lower"], color="#333", lw=1, ls="--",
               label=f"−1.96 SD  {s['loa_lower']:.2f}°")
    ax.axhline(0, color="#bbb", lw=0.7, ls=":")
    ax.set_xlabel(f"Mittelwert ({type_a}, {type_b}) [°]")
    ax.set_ylabel(f"Differenz ({type_a} − {type_b}) [°]")
    ax.set_title(f"Bland-Altman  (n={s['n']})")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Image Group Comparison", layout="wide")
st.title("Image-Group Vergleich")
st.caption("Cobb-Winkel-Vergleich zwischen img_types innerhalb Patient.image_groups")

# ── Sidebar: fetch / cache ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Daten")
    cached = load_cache()
    if cached:
        st.success(f"Cache vorhanden ({len(cached)} Bilder)")
    else:
        st.info("Kein Cache — Token eingeben und laden")

    token = st.text_input("Bearer Token", type="password",
                          help="xray-annotator.renecol.org → Account → Token kopieren")
    fetch_btn = st.button("Von API laden & cachen", disabled=not token.strip())

    if fetch_btn and token.strip():
        with st.spinner("Lade von API …"):
            try:
                raw = fetch_from_api(token.strip())
                save_cache(raw)
                st.success(f"{len(raw)} Bilder geladen und gecacht.")
                cached = raw
            except Exception as e:
                st.error(f"Fehler: {e}")

    st.divider()
    st.header("Vergleichspaare")
    st.caption("Welche img_types sollen innerhalb einer Group verglichen werden?")
    raw_pairs = st.text_area(
        "Paare (je Zeile: TypeA, TypeB)",
        value="\n".join(f"{a}, {b}" for a, b in DEFAULT_PAIRS),
    )
    comp_pairs = []
    for line in raw_pairs.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            comp_pairs.append(tuple(parts))

# ── Main ──────────────────────────────────────────────────────────────────────
if not cached:
    st.warning("Bitte Token eingeben und Daten laden.")
    st.stop()

data = cached

# Build group index
group_index = build_group_index(data)

# ── Tab 1: Alle image_groups auflisten ────────────────────────────────────────
tab_list, tab_compare, tab_debug = st.tabs(
    ["Gruppen-Übersicht", "Statistischer Vergleich", "Debug / Rohdaten"]
)

with tab_list:
    st.subheader("Verfügbare image_groups")
    if not group_index:
        st.info("Keine image_groups in den Daten gefunden.")
        # Show distribution of image_groups values
        groups_raw = []
        for img in data:
            rp = img.get("referring_patient") or {}
            for g in (rp.get("image_groups") or []):
                groups_raw.append(g)
        if groups_raw:
            st.write("Rohe image_groups Werte (Sample):", groups_raw[:20])
        else:
            st.write("Alle `referring_patient.image_groups` sind leer oder `[]`.")
    else:
        summary_rows = []
        for gname, pat_entries in group_index.items():
            n_patients = len(pat_entries)
            all_types = set()
            for pe in pat_entries:
                all_types |= set(pe["images"].keys())
            summary_rows.append({
                "Gruppe": gname,
                "Patienten": n_patients,
                "img_types": ", ".join(sorted(all_types)),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # Per-group detail
        for gname, pat_entries in group_index.items():
            with st.expander(f"Gruppe: {gname}"):
                rows = []
                for pe in pat_entries:
                    for itype, imgs in pe["images"].items():
                        for img in imgs:
                            n_md = len(img.get("measure_dataset", []))
                            rows.append({
                                "patient_id": pe["patient_id"],
                                "img_type": itype,
                                "image_uuid": img.get("uuid", "")[:8] + "…",
                                "acquisition_date": img.get("acquisition_date"),
                                "n_measurements": n_md,
                            })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)


with tab_compare:
    st.subheader("Statistischer Vergleich")
    st.caption(
        "Paare werden innerhalb jeder image_group gebildet — "
        "Bilder verschiedener Gruppen werden nie gemischt."
    )

    if not group_index:
        st.info("Keine image_groups mit Daten gefunden.")
    else:
        sel_cobb = st.selectbox("Cobb-Typ", COBB_TYPES)

        # Collect paired angles across ALL groups — group membership is the pairing key
        for type_a, type_b in comp_pairs:
            st.markdown(f"#### {type_a}  vs.  {type_b}")

            # {region: {"a": [vals], "b": [vals]}}
            data_by_region: dict[str, dict] = {}
            per_patient_rows = []

            for group_name, pat_entries in group_index.items():
                for pe in pat_entries:
                    imgs_a = pe["images"].get(type_a, [])
                    imgs_b = pe["images"].get(type_b, [])
                    if not imgs_a or not imgs_b:
                        continue

                    df_a = mean_angles_for_images(imgs_a)
                    df_b = mean_angles_for_images(imgs_b)
                    if df_a.empty or df_b.empty:
                        continue

                    paired = pair_angles(df_a, df_b, type_a, type_b,
                                         sel_cobb_type=sel_cobb)
                    if paired.empty:
                        continue

                    for _, row in paired.iterrows():
                        a_val = row[f"mean_angle_{type_a}"]
                        b_val = row[f"mean_angle_{type_b}"]
                        region = row["region"]
                        bucket = data_by_region.setdefault(region, {"a": [], "b": []})
                        bucket["a"].append(a_val)
                        bucket["b"].append(b_val)
                        per_patient_rows.append({
                            "patient_id": pe["patient_id"],
                            "image_group": group_name,
                            "cobbType": row["cobbType"],
                            "region": region,
                            type_a: round(a_val, 2),
                            f"n_raters_{type_a}": int(row.get(f"n_raters_{type_a}", 0)),
                            type_b: round(b_val, 2),
                            f"n_raters_{type_b}": int(row.get(f"n_raters_{type_b}", 0)),
                            "Differenz": round(a_val - b_val, 2),
                        })

            total_n = sum(len(v["a"]) for v in data_by_region.values())
            if total_n < 2:
                st.info(f"Zu wenig gepaarte Daten (n={total_n}) für {type_a} / {type_b} · {sel_cobb}.")
                continue

            # Overall stats (alle Regionen zusammen)
            all_a = np.array([v for r in data_by_region.values() for v in r["a"]])
            all_b = np.array([v for r in data_by_region.values() for v in r["b"]])
            s = compute_stats(all_a, all_b)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("n Paare gesamt", s["n"])
            c2.metric("Bias (A−B)", f"{s['mean_diff']:.2f}°")
            c3.metric("Pearson r", f"{s['pearson_r']:.3f}")
            c4.metric("p (paired t)", f"{s['p_ttest']:.4f}")

            col_bar, col_ba = st.columns([3, 2])
            with col_bar:
                st.pyplot(region_bar_plot(data_by_region, type_a, type_b))
            with col_ba:
                st.pyplot(bland_altman_by_region(data_by_region, type_a, type_b))

            # Per-region stats table
            region_rows = []
            for r, vals in sorted(data_by_region.items(), key=lambda x: _region_sort_key(x[0])):
                a_arr, b_arr = np.array(vals["a"]), np.array(vals["b"])
                n = len(a_arr)
                bias = float(np.mean(a_arr - b_arr))
                r_val = float(stats.pearsonr(a_arr, b_arr)[0]) if n >= 2 else float("nan")
                region_rows.append({
                    "Region": r, "n": n,
                    f"MW {type_a}": round(float(np.mean(a_arr)), 2),
                    f"SD {type_a}": round(float(np.std(a_arr, ddof=1)) if n > 1 else 0, 2),
                    f"MW {type_b}": round(float(np.mean(b_arr)), 2),
                    f"SD {type_b}": round(float(np.std(b_arr, ddof=1)) if n > 1 else 0, 2),
                    "Bias": round(bias, 2),
                    "r": round(r_val, 3) if not np.isnan(r_val) else "—",
                })
            st.dataframe(pd.DataFrame(region_rows), use_container_width=True, hide_index=True)

            with st.expander("Rohdaten"):
                st.dataframe(pd.DataFrame(per_patient_rows), use_container_width=True)

            st.divider()


with tab_debug:
    st.subheader("Rohdaten-Inspektion")
    st.write(f"**Gesamtbilder im Cache:** {len(data)}")

    # Show image_groups distribution
    all_groups_raw = []
    for img in data:
        rp = img.get("referring_patient") or {}
        groups = rp.get("image_groups") or []
        all_groups_raw.extend(groups)

    st.write(f"**Summe image_group-Einträge (alle Patienten):** {len(all_groups_raw)}")
    if all_groups_raw:
        st.write("**Sample (erste 10 Einträge):**")
        st.json(all_groups_raw[:10])

    # Patients with non-empty image_groups
    patients_with_groups = {
        (img.get("referring_patient") or {}).get("uuid")
        for img in data
        if (img.get("referring_patient") or {}).get("image_groups")
    } - {None}
    st.write(f"**Patienten mit image_groups ≠ []:** {len(patients_with_groups)}")

    # Raw group index structure
    if group_index:
        st.write("**Ermittelter group_index (Gruppen → Patienten):**")
        debug_index = {
            g: [{"patient_id": pe["patient_id"],
                 "img_types": list(pe["images"].keys())} for pe in pes]
            for g, pes in group_index.items()
        }
        st.json(debug_index)
