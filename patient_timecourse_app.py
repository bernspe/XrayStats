"""
Zeitlicher Winkelverlauf pro Patient — Streamlit App (Prototyp)

Prototyp fuer die geplante Patient.vue-Visualisierung im xray-annotator:
das echte Wirbelsaeulen-Icon aus xray-annotator (whole-spine-*-labels.svg)
dient als Basis. Die darin bereits vorhandenen Region-Klammern (MT/TL/U-LU/
L-LU/LU) und Segment-Marker (L1/L2 ... L5/S1) werden nach der KLINISCHEN
BEWERTUNG der Gesamtveraenderung seit dem Erstbefund eingefaerbt (gruen=besser,
rot=schlechter, Gradient) statt nach Absolutwert — die Bewertungsrichtung haengt
von Bildtyp und Region ab (siehe trend_direction()). Hover zeigt Delta
(absolut/prozentual) Endwert-Anfangswert als Zahl und hebt die zugehoerige
Linie in den Winkelverlauf-Charts hervor; Aenderungen unter der
Signifikanzschwelle (siehe REGIONAL_/SEGMENTAL_SIGNIFICANCE_DEG) gelten als
"stabil" (neutral).

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run patient_timecourse_app.py
"""
import warnings
warnings.filterwarnings("ignore")

import copy
import io
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from annotation_analysis import build_angles_df, is_lateral
from prepare_outlier_data import fetch_img_urls

DATA_DIR = "data"
IMG_TYPES = {"a.p.": "Xray-LWS-ap", "seitlich": "Xray-LWS-lat"}
REGIONAL_COBB_TYPES = ["Cobb-regional", "Spline"]  # zusammengefasst als "Regionale Messung"
ICON_MAX_HEIGHT_PX = 200

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
XRAY_ANNOTATOR_PUBLIC = Path("/Users/peterbernstein/Vue/xray-annotator/public")
SVG_FILES = {
    "Xray-LWS-ap": XRAY_ANNOTATOR_PUBLIC / "whole-spine-ap-labels2.svg",
    "Xray-LWS-lat": XRAY_ANNOTATOR_PUBLIC / "whole-spine-lateral-labels2.svg",
}

# Region/segment name -> element id inside the SVG (see docs/api_imagedata_structure.md §8
# and bspline_python.spine_regions for the underlying vertebra ranges). "LU" (whole lumbar
# lordosis) only exists as an element in the lateral SVG; harmless no-op lookup for ap.
REGION_IDS = {"MT": "mt", "TL": "tl", "U-LU": "u-lu", "L-LU": "l-lu", "LU": "lu"}
# Lateral films report the sagittal profile TK (T4-T12), TL (T10-L2, the
# thoracolumbar junction) and LL (L1-S1). ponytail: the lateral SVG has no tk/ll
# markers, so TK reuses the mt arc (T5-T11) and LL the lu arc (L1-S1, exact).
LATERAL_REGION_IDS = {"TK": "mt", "TL": "tl", "LL": "lu"}


def region_ids_for(img_type: str) -> dict:
    return LATERAL_REGION_IDS if is_lateral(img_type) else REGION_IDS

SEGMENT_IDS = {"L1/L2": "l1-l2", "L2/L3": "l2-l3", "L3/L4": "l3-l4", "L4/L5": "l4-l5", "L5/S1": "l5-s1"}

# Diverging good/bad gradient (not absolute severity): a change is colour-coded by whether
# it is clinically favourable, not by how large the angle itself is.
GOOD_RGB = (40, 167, 69)      # green
NEUTRAL_RGB = (240, 240, 240)  # no/insufficient/not-significant change
BAD_RGB = (215, 48, 39)       # red

# Below these |delta| thresholds a change counts as noise, not a real clinical change:
# coloured neutral ("stabil") rather than green/red. Regional Cobb angles are coarser
# measurements (larger inter-rater spread) than segmental ones, hence the higher bar.
REGIONAL_SIGNIFICANCE_DEG = 5.0
SEGMENTAL_SIGNIFICANCE_DEG = 3.0
TREND_SCALE_DEG = 5.0  # degrees *beyond* the significance threshold for full colour saturation
# ponytail: fixed thresholds/scale, not calibrated against real MDC (minimal detectable
# change) literature — tune if colours saturate too eagerly/rarely in practice.

LATERAL_BAD_ON_INCREASE = {"TK", "TL"}  # more kyphosis = worse


def trend_direction(img_type: str, label: str) -> int:
    """+1 if an *increase* in this label's angle is clinically favourable, else -1.

    a.p. (scoliosis): any increase is worse, everywhere.
    lateral (sagittal profile): more lumbar lordosis (LL and the lumbar segmental
    angles) is favourable; more kyphosis (TK, junctional TL) is not.
    """
    if img_type == "Xray-LWS-ap":
        return -1
    return -1 if label in LATERAL_BAD_ON_INCREASE else 1


def _lerp_rgb(a: tuple, b: tuple, f: float) -> tuple:
    return tuple(round(a[i] + (b[i] - a[i]) * f) for i in range(3))


def trend_color(delta: float | None, direction: int, significance: float,
                 scale: float = TREND_SCALE_DEG) -> str:
    if delta is None or abs(delta) < significance:
        rgb = NEUTRAL_RGB
    else:
        sign = 1.0 if delta * direction > 0 else -1.0
        magnitude = min(1.0, (abs(delta) - significance) / scale)
        goodness = sign * magnitude
        rgb = _lerp_rgb(NEUTRAL_RGB, BAD_RGB, -goodness) if goodness < 0 else _lerp_rgb(NEUTRAL_RGB, GOOD_RGB, goodness)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@st.cache_resource
def load_angles() -> pd.DataFrame:
    df4d = pd.read_pickle(f"{DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    # lateral films: TK/TL/LL spans instead of the a.p. apex regions
    angles = build_angles_df(df4d, lateral_spans=tuple(REGIONAL_COBB_TYPES))
    meta = df4d[["id", "patient_id", "acquisition_date"]].drop_duplicates("id")
    angles = angles.merge(meta, left_on="image_id", right_on="id", how="left")
    angles["acquisition_date"] = pd.to_datetime(angles["acquisition_date"], errors="coerce")
    return angles.dropna(subset=["acquisition_date", "patient_id"])


def patients_with_timecourse(angles: pd.DataFrame) -> list[str]:
    """Patients with >=2 distinct visit dates of measurable Cobb angles in at
    least one of ap/lat — i.e. where a Verlauf (trend) actually exists, not
    just a single baseline snapshot with nothing to compare it to."""
    qualifying = angles[
        angles["img_type"].isin(IMG_TYPES.values())
        & angles["cobbType"].isin([*REGIONAL_COBB_TYPES, "Cobb-segmental"])
    ]
    visit_counts = qualifying.groupby(["patient_id", "img_type"])["acquisition_date"].nunique()
    return sorted(visit_counts[visit_counts >= 2].index.get_level_values("patient_id").unique())


@st.cache_resource
def load_image_meta() -> pd.DataFrame:
    df4d = pd.read_pickle(f"{DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    meta = df4d[["id", "patient_id", "img_type", "acquisition_date"]].drop_duplicates("id").copy()
    meta["acquisition_date"] = pd.to_datetime(meta["acquisition_date"], errors="coerce")
    return meta.dropna(subset=["acquisition_date", "patient_id"])


@st.cache_resource
def load_img_urls() -> dict:
    """image_id -> signed GCS URL, refreshed independently of df4d via the sidebar
    token form (same data/img_urls.pkl cache as subluxation_app.py)."""
    with open(f"{DATA_DIR}/img_urls.pkl", "rb") as fh:
        return pickle.load(fh)


def _largest_run(mask: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    splits = np.flatnonzero(np.diff(idx) > 1)
    best = max(np.split(idx, splits + 1), key=len)
    return int(best[0]), int(best[-1])


def crop_black_border(img: Image.Image, threshold: float = 5.0) -> Image.Image:
    """Trim the black letterboxing baked into the exported x-ray scans.

    A plain bounding box of "any pixel above threshold" doesn't work here:
    these exports have a 1px-wide bright frame line at the very edges, which
    alone touches every row/column and blocks the box from shrinking. Instead
    take the largest *contiguous* run of columns/rows whose mean brightness
    clears the threshold — the actual film is one wide contiguous block, the
    border frame is a 1px sliver that can't compete on width.
    """
    arr = np.asarray(img.convert("L"), dtype=float)
    col_run = _largest_run(arr.mean(axis=0) > threshold)
    row_run = _largest_run(arr.mean(axis=1) > threshold)
    if not col_run or not row_run:
        return img
    return img.crop((col_run[0], row_run[0], col_run[1] + 1, row_run[1] + 1))


@st.cache_data(show_spinner=False)
def load_cropped_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return crop_black_border(Image.open(io.BytesIO(resp.content)))
    except Exception:
        return None


def fit_height(img: Image.Image, height: int) -> Image.Image:
    """Resize to an exact pixel height so st.image(..., width="content") renders it
    at that height regardless of the column's width (st.image has no height param)."""
    scale = height / img.height
    return img.resize((max(1, round(img.width * scale)), height))


def with_synthetic_lu(df_region: pd.DataFrame) -> pd.DataFrame:
    """Add LU (whole lumbar lordosis) rows = U-LU ∪ L-LU, relabelled — build_label_series's
    per-(image,rater,date) max-reduction then does max(U-LU,L-LU) per rater, same convention
    as stats_app.py's synthetic LU region."""
    lu_source = df_region[df_region["_label"].isin(["U-LU", "L-LU"])].copy()
    lu_source["_label"] = "LU"
    return pd.concat([df_region, lu_source], ignore_index=True)


def compute_label_stats(df_labeled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """label -> DataFrame[acquisition_date, angle, std], mean/std across raters per visit.

    Reduces multiple angles per (image, rater) to their max first — needed
    because a region can carry both a Cobb-regional and a Spline angle from
    the same rater on the same image, which would otherwise double-count
    (mirrors annotation_analysis.aggregate_angles_by_region). std is the
    inter-rater spread per visit. Shared source for the pictogram tooltip
    (build_label_series) and the Winkelverlauf line chart (build_wide_chart_df).
    """
    result = {}
    for label, sub in df_labeled.groupby("_label"):
        reduced = sub.groupby(["image_id", "rater_name", "acquisition_date"])["angle"].max().reset_index()
        stats = (
            reduced.groupby("acquisition_date")["angle"]
            .agg(angle="mean", std="std")
            .reset_index()
            .sort_values("acquisition_date")
        )
        stats["std"] = stats["std"].fillna(0.0)
        result[label] = stats
    return result


def build_label_series(stats_by_label: dict[str, pd.DataFrame]) -> dict:
    """label -> chronological [{date_label, angle, std}, ...], JSON-ready for the
    pictogram's hover tooltip (see render_spine_icon)."""
    result = {}
    for label, stats in stats_by_label.items():
        s = stats.copy()
        s["date_label"] = s["acquisition_date"].dt.strftime("%Y-%m-%d")
        result[label] = s[["date_label", "angle", "std"]].round(1).to_dict("records")
    return result


CHART_PALETTE = ["#4da3ff", "#ff8f39", "#3fbf6f", "#e0507a", "#9b6bff",
                  "#c9a227", "#2bb3b3", "#ff6161", "#7a7a7a", "#5ad1e6"]


def render_chart_panel(stats_by_label: dict[str, pd.DataFrame], order: list[str],
                        title: str, width: int = 380, height: int = 170) -> str:
    """Self-contained SVG line-chart panel with its own y-axis (regional and
    segmental angles differ enough in magnitude to squash one on a shared
    scale) and a mean ± std band per series. Lines/legend share class
    "xr-marker"+"chart-line"/data-key with the pictogram markers in
    render_spine_icon, so the same hover handlers show the tooltip and
    cross-highlight this line whichever element (pictogram, chart line, or
    legend swatch) triggered it.
    """
    series = [(label, stats_by_label[label]) for label in order
              if label in stats_by_label and not stats_by_label[label].empty]
    pad_l, pad_r, pad_t = 34, 8, 16
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - 34
    title_html = f'<text x="4" y="12" font-size="11" fill="#333" font-weight="600">{title}</text>'
    if not series:
        return (f'<svg width="{width}" height="{height}">{title_html}'
                 f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" font-size="11" fill="#999" '
                 f'text-anchor="middle">keine Daten</text></svg>')

    all_dates = pd.concat([s["acquisition_date"] for _, s in series])
    dmin, dmax = all_dates.min(), all_dates.max()
    span = (dmax - dmin).total_seconds() or 1.0

    def x_at(d) -> float:
        return pad_l + (pd.Timestamp(d) - dmin).total_seconds() / span * plot_w

    y_max = max((s["angle"] + s["std"]).max() for _, s in series)
    y_max = max(5.0, np.ceil(max(y_max, 1e-6) / 5.0) * 5.0)

    def y_at(v: float) -> float:
        return pad_t + plot_h - max(0.0, v) / y_max * plot_h

    parts = [f'<svg width="{width}" height="{height}">', title_html]
    for frac in (0.0, 0.5, 1.0):
        val = y_max * frac
        y = y_at(val)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
                      f'stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 4}" y="{y + 3:.1f}" font-size="9" fill="#888" '
                      f'text-anchor="end">{val:.0f}°</text>')
    for d in (dmin, dmax):
        parts.append(f'<text x="{x_at(d):.1f}" y="{height - 22}" font-size="9" fill="#888" '
                      f'text-anchor="middle">{d:%Y-%m-%d}</text>')

    legend_x = pad_l
    for i, (label, stats) in enumerate(series):
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        pts = stats.sort_values("acquisition_date")
        xs = [x_at(d) for d in pts["acquisition_date"]]
        line_pts = " ".join(f"{x:.1f},{y_at(a):.1f}" for x, a in zip(xs, pts["angle"]))
        band_top = [(x, y_at(a + s)) for x, a, s in zip(xs, pts["angle"], pts["std"])]
        band_bot = [(x, y_at(a - s)) for x, a, s in zip(xs, pts["angle"], pts["std"])][::-1]
        band_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in band_top + band_bot)
        dots = "".join(f'<circle cx="{x:.1f}" cy="{y_at(a):.1f}" r="2.5" fill="{color}"/>'
                        for x, a in zip(xs, pts["angle"]))
        parts.append(
            f'<g class="xr-marker chart-line" data-key="{label}">'
            f'<polygon points="{band_pts}" fill="{color}" fill-opacity="0.15"/>'
            f'<polyline points="{line_pts}" fill="none" stroke="{color}" stroke-width="2"/>'
            f'{dots}</g>'
        )
        parts.append(
            f'<g class="xr-marker" data-key="{label}" transform="translate({legend_x},{height - 6})">'
            f'<rect width="8" height="8" y="-8" fill="{color}"/>'
            f'<text x="11" y="0" font-size="9" fill="#555">{label}</text></g>'
        )
        legend_x += 11 + 6 * len(label) + 10
    parts.append("</svg>")
    return "".join(parts)


def render_spine_icon(svg_path: Path, label_series: dict, img_type: str,
                       chart_html: str = "", chart_height: int = 0) -> tuple[str, int, int]:
    """Inline the xray-annotator spine icon; colour its existing region/segment
    markers by whether the latest change is clinically favourable, hide markers
    without data, and wire up hover -> a compact numeric tooltip (value ± std, Δ).

    chart_html (from render_chart_panel) is placed alongside the icon in the
    same document, so hovering a pictogram marker, a chart line, or a chart
    legend swatch all share one JS hover handler — that's what lets a
    pictogram hover highlight the matching line chart without cross-iframe
    messaging.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    vb_w, vb_h = (float(v) for v in root.get("viewBox").split()[2:])
    icon_width_px = round(ICON_MAX_HEIGHT_PX * vb_w / vb_h)
    root.attrib.pop("width", None)
    root.attrib.pop("height", None)
    root.set("style", f"height:{ICON_MAX_HEIGHT_PX}px;width:auto;display:block;margin:0 auto")
    # at 200px height this icon renders only ~40px wide (tall/narrow viewBox) — enlarge the
    # invisible hover targets below so hover stays usable at that scale.
    parent_map = {child: parent for parent in root.iter() for child in parent}

    region_ids = region_ids_for(img_type)
    label_significance = {**{l: REGIONAL_SIGNIFICANCE_DEG for l in region_ids},
                           **{l: SEGMENTAL_SIGNIFICANCE_DEG for l in SEGMENT_IDS}}

    trend_dir = {}
    trend_sig = {}
    for label, eid in {**region_ids, **SEGMENT_IDS}.items():
        el = root.find(f".//*[@id='{eid}']")
        if el is None:
            continue
        records = label_series.get(label, [])
        tag = el.tag.split("}")[-1]
        # Region markers come in two shapes across the two SVGs: a <g> wrapping two
        # boundary-line <path> children (ap), or a single standalone <path> (lateral).
        boundary_lines = [c for c in el if c.tag.split("}")[-1] == "path"] if tag == "g" else [el]
        if not records:
            el.set("style", "display:none")
            continue
        delta = records[-1]["angle"] - records[0]["angle"] if len(records) > 1 else None
        direction = trend_direction(img_type, label)
        significance = label_significance[label]
        trend_dir[label] = direction
        trend_sig[label] = significance
        color = trend_color(delta, direction, significance)
        el.set("data-key", label)
        el.set("class", "xr-marker")
        if tag == "ellipse":
            el.set("style", f"fill:{color};fill-opacity:0.9;stroke:#333;stroke-width:0.5")
            # invisible enlarged duplicate makes the small dot easier to hover
            hit = copy.deepcopy(el)
            hit.attrib.pop("id", None)
            hit.set("rx", str(float(el.get("rx")) * 2.2))
            hit.set("ry", str(float(el.get("ry")) * 2.2))
            hit.set("style", "fill:#000;fill-opacity:0;stroke:none")
            parent = parent_map[el]
            parent.insert(list(parent).index(el), hit)
        else:  # region marker: one or two boundary-line <path>s (see boundary_lines above)
            for child in boundary_lines:
                child.set("style", f"fill:none;stroke:{color};stroke-width:0.7;stroke-opacity:0.95")
                # invisible wide duplicate makes the thin line easier to hover
                hit = copy.deepcopy(child)
                hit.attrib.pop("id", None)
                hit.set("style", "fill:none;stroke:#000;stroke-opacity:0;stroke-width:4")
                if child is el:
                    parent_map[el].insert(list(parent_map[el]).index(el), hit)
                else:
                    el.append(hit)

    svg_markup = ET.tostring(root, encoding="unicode")
    data_json = json.dumps(label_series)
    dir_json = json.dumps(trend_dir)
    sig_json = json.dumps(trend_sig)

    html = f"""
    <div style="position:relative;background:white;padding:12px;border-radius:8px;
                display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
      {svg_markup}
      <div style="display:flex;flex-direction:column;gap:8px">{chart_html}</div>
      <div id="xr-tooltip" style="position:fixed;display:none;background:#111;color:#fff;
           padding:6px 9px;border-radius:6px;font:12px sans-serif;pointer-events:none;
           z-index:9999;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.3)"></div>
    </div>
    <script>
    const XR_DATA = {data_json};
    const XR_DIR = {dir_json};
    const XR_SIG = {sig_json};
    function xrSetHighlight(key) {{
      document.querySelectorAll(".chart-line").forEach(el => {{
        el.style.opacity = (el.getAttribute("data-key") === key) ? "1" : "0.15";
      }});
    }}
    function xrClearHighlight() {{
      document.querySelectorAll(".chart-line").forEach(el => el.style.opacity = "");
    }}
    function xrShowTooltip(evt, key) {{
      const records = XR_DATA[key];
      if (!records || !records.length) return;
      xrSetHighlight(key);
      const tip = document.getElementById("xr-tooltip");
      const last = records[records.length - 1];
      const first = records.length > 1 ? records[0] : null;
      let deltaHtml = "Ausgangswert";
      if (first) {{
        const dAbs = (last.angle - first.angle);
        const dPct = first.angle !== 0 ? (dAbs / first.angle * 100) : null;
        const goodness = dAbs * (XR_DIR[key] || 0);
        const verdict = Math.abs(dAbs) < (XR_SIG[key] || 0) ? "stabil" : (goodness > 0 ? "Verbesserung" : "Verschlechterung");
        deltaHtml = (dAbs >= 0 ? "+" : "") + dAbs.toFixed(1) + "°" +
          (dPct !== null ? " (" + (dPct >= 0 ? "+" : "") + dPct.toFixed(0) + "%)" : "") +
          " — " + verdict;
      }}
      tip.innerHTML = "<b>" + key + "</b> " + last.date_label + ": " + last.angle + "° ± " + last.std +
        "°<br>Δ: " + deltaHtml;
      tip.style.display = "block";
      // keep the box inside the iframe's own viewport instead of letting it clip at the edge
      let left = evt.clientX + 12, top = evt.clientY + 12;
      if (left + tip.offsetWidth > window.innerWidth) left = evt.clientX - tip.offsetWidth - 12;
      if (top + tip.offsetHeight > window.innerHeight) top = evt.clientY - tip.offsetHeight - 12;
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = Math.max(0, top) + "px";
    }}
    document.querySelectorAll(".xr-marker").forEach(el => {{
      el.style.cursor = "pointer";
      el.addEventListener("mouseenter", e => xrShowTooltip(e, el.getAttribute("data-key")));
      el.addEventListener("mousemove", e => xrShowTooltip(e, el.getAttribute("data-key")));
      el.addEventListener("mouseleave", () => {{
        document.getElementById("xr-tooltip").style.display = "none";
        xrClearHighlight();
      }});
    }});
    </script>
    """
    return html, icon_width_px, max(ICON_MAX_HEIGHT_PX, chart_height)


def render_gradient_legend() -> str:
    return f"""
    <div style="max-width:220px;margin:8px auto;font:12px sans-serif;text-align:center">
      <div style="height:10px;border-radius:5px;background:linear-gradient(to right,
           #{'{:02x}{:02x}{:02x}'.format(*BAD_RGB)}, #{'{:02x}{:02x}{:02x}'.format(*NEUTRAL_RGB)},
           #{'{:02x}{:02x}{:02x}'.format(*GOOD_RGB)})"></div>
      <div style="display:flex;justify-content:space-between;color:#666;margin-top:2px">
        <span>Verschlechterung</span><span>stabil</span><span>Verbesserung</span>
      </div>
    </div>
    """


def show_xray(col, row, img_urls: dict, label: str) -> None:
    date_str = f"{label}: {row['acquisition_date']:%Y-%m-%d}"
    url = img_urls.get(row["id"])
    img = load_cropped_image(url) if url else None
    if img is not None:
        resized = fit_height(img, ICON_MAX_HEIGHT_PX)
        col.image(resized, caption=date_str, width=resized.width)
    else:
        col.caption(f"{date_str} (keine/ungültige Bild-URL im Cache)")


def render_patient_row(img_label: str, img_type: str, patient_id: str,
                        angles: pd.DataFrame, img_meta: pd.DataFrame, img_urls: dict) -> None:
    """First x-ray | last x-ray, then the pictogram + linked regional/segmental
    line charts (own y-axis each) below, all in one hover-linked component."""
    st.subheader(img_label)
    df_p = angles[(angles["img_type"] == img_type) & (angles["patient_id"] == patient_id)]
    patient_images = (
        img_meta[(img_meta["img_type"] == img_type) & (img_meta["patient_id"] == patient_id)]
        .sort_values("acquisition_date")
    )

    df_region = with_synthetic_lu(
        df_p[df_p["cobbType"].isin(REGIONAL_COBB_TYPES)].assign(_label=lambda d: d["region"])
    )
    df_segment = df_p[df_p["cobbType"] == "Cobb-segmental"].assign(
        _label=lambda d: d["vertebra_upper"] + "/" + d["vertebra_lower"]
    )
    stats_by_label = {**compute_label_stats(df_region), **compute_label_stats(df_segment)}

    cols = st.columns([1, 1, 3])
    if not patient_images.empty:
        first_row, last_row = patient_images.iloc[0], patient_images.iloc[-1]
        show_xray(cols[0], first_row, img_urls, "Erstbefund")
        if len(patient_images) > 1:
            show_xray(cols[1], last_row, img_urls, "Letztbefund")
    else:
        cols[0].caption("Keine Aufnahme vorhanden")

    # both chart panels stacked must fit the pictogram's own height (ICON_MAX_HEIGHT_PX)
    # so the whole row — x-rays, pictogram, charts — stays exactly one line tall.
    panel_gap = 8
    panel_height = (ICON_MAX_HEIGHT_PX - panel_gap) // 2
    chart_html = (
        render_chart_panel(stats_by_label, list(region_ids_for(img_type)), "Regionale Winkel",
                            width=220, height=panel_height)
        + render_chart_panel(stats_by_label, list(SEGMENT_IDS.keys()), "Segmentale Winkel",
                              width=220, height=panel_height)
    )
    icon_html, icon_width, combined_height = render_spine_icon(
        SVG_FILES[img_type], build_label_series(stats_by_label), img_type,
        chart_html=chart_html, chart_height=2 * panel_height + panel_gap,
    )
    with cols[2]:
        st.iframe(icon_html, width="stretch", height=combined_height + 24)


def main():
    st.set_page_config(page_title="Winkelverlauf", layout="wide")
    st.title("Zeitlicher Winkelverlauf pro Patient")

    angles = load_angles()
    img_meta = load_image_meta()
    try:
        img_urls = load_img_urls()
    except FileNotFoundError:
        img_urls = {}

    with st.sidebar:
        patients = patients_with_timecourse(angles)
        if not patients:
            st.warning("Keine Patienten mit rekonstruierbarem Verlauf (>=2 Aufnahmen) gefunden.")
            st.stop()

        if "patient_select" not in st.session_state or st.session_state.patient_select not in patients:
            st.session_state.patient_select = patients[0]

        def _advance_patient() -> None:
            # callback runs before the script reruns, so mutating the selectbox's
            # own keyed state here is allowed (doing it after the widget renders
            # in the normal script body is not — Streamlit raises on that).
            cur = patients.index(st.session_state.patient_select)
            if cur < len(patients) - 1:
                st.session_state.patient_select = patients[cur + 1]

        patient_id = st.selectbox("Patient", patients, key="patient_select")
        st.button("Nächster Patient ▶", disabled=(patients.index(patient_id) >= len(patients) - 1),
                  width="stretch", on_click=_advance_patient)

        st.divider()
        st.subheader("Bild-URLs auffrischen")
        st.caption("Signierte GCS-URLs laufen nach 24h ab.")
        token = st.text_input("Bearer Token", type="password",
                               help="xray-annotator.renecol.org → Account info → Copy token")
        if st.button("URLs auffrischen"):
            if not token.strip():
                st.error("Bitte Token eingeben.")
            else:
                with st.spinner("Bild-URLs werden von der API geholt …"):
                    try:
                        fresh_urls = fetch_img_urls(token.strip())
                        with open(f"{DATA_DIR}/img_urls.pkl", "wb") as fh:
                            pickle.dump(fresh_urls, fh)
                        load_img_urls.clear()
                        st.success(f"{len(fresh_urls)} URLs aufgefrischt.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fehlgeschlagen: {e}")

    st.caption(
        f"Patient {patient_id} — Farbe = klinische Bewertung der Veränderung seit dem Erstbefund (nicht Absolutwert), "
        "Hover auf Piktogramm zeigt Verlauf ± Streubreite. ⚠️ Bild-URLs sind nur 24h gültig — "
        "bei abgelaufenen/fehlenden URLs links Token eingeben und auffrischen."
    )

    render_patient_row("a.p.", "Xray-LWS-ap", patient_id, angles, img_meta, img_urls)
    render_patient_row("seitlich", "Xray-LWS-lat", patient_id, angles, img_meta, img_urls)
    st.markdown(render_gradient_legend(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
