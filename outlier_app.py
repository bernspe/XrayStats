"""
Annotation Precision Outlier Viewer — Streamlit App

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run outlier_app.py

Vorbedingung: prepare_outlier_data.py einmalig ausführen, um Pickle-Dateien zu erstellen.
"""

import io
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from bspline_python import interpolate_bspline, smooth_nodes
from annotation_analysis import get_inflection_overlay_data

# Konva-Stage hat feste Höhe 600px (KonvaImage.vue).
# Alle gespeicherten Node/CobbLine-Koordinaten sind in diesem Canvas-Raum.
KONVA_STAGE_HEIGHT = 600
DATA_DIR = "data"
COBB_LINE_TYPES = {"Cobb-regional", "Cobb-segmental"}
COLORS = plt.cm.tab10.colors


# ---------------------------------------------------------------------------
# Daten laden (gecacht für die Session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_data():
    df4d = pd.read_pickle(f"{DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    angles_df = pd.read_pickle(f"{DATA_DIR}/outlier_angles_df.pkl")
    outlier_df = pd.read_pickle(f"{DATA_DIR}/outlier_ranking_df.pkl")
    with open(f"{DATA_DIR}/img_urls.pkl", "rb") as fh:
        img_urls = pickle.load(fh)
    # Pre-build per-image ranking summary
    img_rank = (
        outlier_df
        .groupby(["cobbType", "image_id"])
        .agg(max_std=("std", "max"), img_type=("img_type", "first"))
        .reset_index()
    )
    return df4d, angles_df, outlier_df, img_urls, img_rank


@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


SMOOTH_MIN_ANGLE = 10.0  # Threshold für Inflection-Methode auf geglätteter Spline


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------

def build_figure(image_id: int, cobb: str, df4d, outlier_df, angles_df, img_urls,
                 smooth_sigma: float = 0.0):
    url = img_urls.get(image_id)
    xray = None
    if url:
        raw = fetch_image_bytes(url)
        if raw:
            xray = Image.open(io.BytesIO(raw)).convert("RGB")

    scale = (xray.height / KONVA_STAGE_HEIGHT) if xray else 1.0
    measurements = df4d[df4d["id"] == image_id]

    fig, (ax_img, ax_tbl) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # X-Ray
    if xray:
        ax_img.imshow(xray, origin="upper", aspect="equal")
    else:
        ax_img.set_facecolor("#222")
        ax_img.text(0.5, 0.5, "Bild nicht verfügbar\n(URL abgelaufen?)",
                    ha="center", va="center", color="white",
                    transform=ax_img.transAxes, fontsize=12)

    # Overlay pro Rater
    for i, (_, row) in enumerate(measurements.iterrows()):
        xmd = row.get("xray_measurement_data")
        if not isinstance(xmd, dict):
            continue
        nodes_raw = xmd.get("stateObject", {}).get("nodes", [])
        if len(nodes_raw) < 3:
            continue

        color = COLORS[i % len(COLORS)]
        rater = row.get("rater_name", f"Rater {i+1}")

        nodes_xy = [(n["x"] * scale, n["y"] * scale) for n in nodes_raw]
        spline = interpolate_bspline(nodes_xy, num_steps=20)

        if spline:
            ax_img.plot(
                [p[0] for p in spline], [p[1] for p in spline],
                "-", color=color, linewidth=2, label=rater, alpha=0.85, zorder=3,
            )

        # Geglättete Spline + Inflection-Perpendikularen
        if smooth_sigma > 0:
            overlay = get_inflection_overlay_data(xmd, min_angle=SMOOTH_MIN_ANGLE,
                                                  smooth_sigma=smooth_sigma)
            # Gestrichelte geglättete Spline
            nodes_smooth = smooth_nodes(nodes_xy, sigma=smooth_sigma)
            spline_smooth = interpolate_bspline(nodes_smooth, num_steps=20)
            if spline_smooth:
                ax_img.plot(
                    [p[0] for p in spline_smooth], [p[1] for p in spline_smooth],
                    "--", color=color, linewidth=1.5, alpha=0.55, zorder=2,
                )
            # Perpendikularen an den Inflektionspunkten
            for pl in overlay["perp_lines"]:
                ax_img.plot(
                    [pl["x1"] * scale, pl["x2"] * scale],
                    [pl["y1"] * scale, pl["y2"] * scale],
                    "-", color=color, linewidth=2, alpha=0.85, zorder=5,
                )

        ax_img.scatter(
            [xy[0] for xy in nodes_xy], [xy[1] for xy in nodes_xy],
            color=color, s=35, zorder=4, edgecolors="white", linewidths=0.5,
        )

        if cobb in COBB_LINE_TYPES:
            for cl in xmd.get("stateObject", {}).get("cobbLines", []):
                if cl.get("mode") == cobb:
                    ax_img.plot(
                        [cl["x1"] * scale, cl["x2"] * scale],
                        [cl["y1"] * scale, cl["y2"] * scale],
                        "-", color=color, linewidth=2.5, alpha=0.9, zorder=5,
                    )

    img_type = measurements["img_type"].iloc[0] if len(measurements) else ""
    ax_img.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax_img.set_title(f"ID {image_id}  ({img_type})", fontsize=12)
    ax_img.axis("off")

    # Tabelle
    ax_tbl.axis("off")
    ax_tbl.set_title(f"cobbType: {cobb}", fontsize=11)

    pair_std = (
        outlier_df[
            (outlier_df["image_id"] == image_id) &
            (outlier_df["cobbType"] == cobb)
        ][["vertebra_upper", "vertebra_lower", "mean_angle", "std", "n_raters"]]
        .sort_values("std", ascending=False)
    )
    angle_rows = (
        angles_df[
            (angles_df["image_id"] == image_id) &
            (angles_df["cobbType"] == cobb)
        ][["rater_name", "vertebra_upper", "vertebra_lower", "angle"]]
        .assign(angle=lambda d: d["angle"].round(1))
        .sort_values(["vertebra_upper", "rater_name"])
    )

    y, lh = 0.98, 0.047

    def txt(s, bold=False, gray=False):
        nonlocal y
        ax_tbl.text(0.02, y, s, fontsize=9,
                    fontweight="bold" if bold else "normal",
                    fontfamily="monospace",
                    color="gray" if gray else "black",
                    transform=ax_tbl.transAxes, va="top")
        y -= lh

    if not pair_std.empty:
        txt("Winkelpaare (σ absteigend):", bold=True)
        txt(f"{'Upper':>8}  {'Lower':>6}  {'Mean':>6}  {'σ':>5}  n", gray=True)
        for _, pr in pair_std.iterrows():
            txt(f"{pr['vertebra_upper']:>8}  {pr['vertebra_lower']:>6}  "
                f"{pr['mean_angle']:>5.1f}°  {pr['std']:>4.1f}°  {int(pr['n_raters'])}")
    y -= lh
    if not angle_rows.empty:
        txt("Einzelmessungen:", bold=True)
        txt(f"{'Rater':>12}  {'Upper':>8}  {'Lower':>6}  {'°':>6}", gray=True)
        for _, ar in angle_rows.iterrows():
            txt(f"{str(ar['rater_name']):>12}  {ar['vertebra_upper']:>8}  "
                f"{ar['vertebra_lower']:>6}  {ar['angle']:>5.1f}°")
    else:
        ax_tbl.text(0.5, 0.5, f"Keine Daten\nfür cobbType={cobb}",
                    ha="center", va="center", transform=ax_tbl.transAxes,
                    fontsize=11, color="gray")

    # Geglättete Inflection-Winkel (σ + min_angle=10°, immer angezeigt wenn Smoothing aktiv)
    if smooth_sigma > 0:
        y -= lh
        txt(f"Geglättet Inflection (σ={smooth_sigma}, min={SMOOTH_MIN_ANGLE:.0f}°):", bold=True)
        all_smoothed = []
        for i, (_, row) in enumerate(measurements.iterrows()):
            xmd = row.get("xray_measurement_data")
            if not isinstance(xmd, dict):
                continue
            rater = row.get("rater_name", f"Rater {i+1}")
            overlay = get_inflection_overlay_data(xmd, min_angle=SMOOTH_MIN_ANGLE,
                                                  smooth_sigma=smooth_sigma)
            for a in overlay["angles"]:
                ca = a["cobbAngle"]
                all_smoothed.append({
                    "rater_name":     rater,
                    "cobbType":       a["cobbType"],
                    "vertebra_upper": ca["vertebra_upper"],
                    "vertebra_lower": ca["vertebra_lower"],
                    "angle":          ca["angle"],
                })

        if all_smoothed:
            txt(f"{'Rater':>12}  {'Rank':>3}  {'Upper':>8}  {'Lower':>6}  {'°':>6}", gray=True)
            for a in sorted(all_smoothed, key=lambda x: (x["cobbType"], x["rater_name"])):
                rank = a["cobbType"].replace("inflection_", "")
                txt(f"{str(a['rater_name']):>12}  {rank:>3}  {a['vertebra_upper']:>8}  "
                    f"{a['vertebra_lower']:>6}  {a['angle']:>5.1f}°")
        else:
            txt("Keine Inflection-Winkel ≥ 10° gefunden", gray=True)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Annotation Precision Outliers",
        layout="wide",
        page_icon="🩻",
    )
    st.title("🩻 Annotation Precision Outliers")

    try:
        df4d, angles_df, outlier_df, img_urls, img_rank = load_data()
    except FileNotFoundError as e:
        st.error(
            f"Pickle-Dateien nicht gefunden: `{e}`\n\n"
            "Bitte zuerst ausführen:\n"
            "```\npython prepare_outlier_data.py --token YOUR_TOKEN\n```"
        )
        return

    # Sidebar
    with st.sidebar:
        st.header("Filter")

        cobb_types = sorted(outlier_df["cobbType"].unique())
        default_cobb = "Cobb-regional" if "Cobb-regional" in cobb_types else cobb_types[0]
        cobb = st.selectbox("cobbType", cobb_types,
                            index=cobb_types.index(default_cobb))

        n_top = st.slider("Top N Bilder", min_value=5, max_value=200,
                          value=30, step=5)

        st.divider()
        st.subheader("Smoothing")
        show_smoothing = st.checkbox("Geglättete Spline anzeigen", value=False)
        smooth_sigma = 0.0
        if show_smoothing:
            smooth_sigma = st.slider(
                "Gauss-Sigma", min_value=0.5, max_value=3.0, value=1.0, step=0.5,
                help="Stärke der Gauss-Glättung der Spline-Nodes. "
                     "Gestrichelte Linie = geglättete Kurve.",
            )

        st.divider()
        st.caption(
            f"**{df4d['id'].nunique()}** Bilder · "
            f"**{len(df4d)}** Messungen · "
            f"**{len(outlier_df)}** Outlier-Gruppen"
        )

    # Bild-Ranking für gewählten cobbType
    ranked = (
        img_rank[img_rank["cobbType"] == cobb]
        .sort_values("max_std", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )

    if ranked.empty:
        st.warning(f"Keine Bilder mit ≥2 Ratern für **{cobb}**.")
        return

    labels = [
        f"{r['img_type']:18s}  ID {r['image_id']}  σ_max={r['max_std']:.1f}°"
        for _, r in ranked.iterrows()
    ]
    selected = st.selectbox("Bild (sortiert nach max. σ)", labels)
    image_id = int(ranked.iloc[labels.index(selected)]["image_id"])

    # Figure
    with st.spinner("Bild laden …"):
        fig = build_figure(image_id, cobb, df4d, outlier_df, angles_df, img_urls,
                           smooth_sigma=smooth_sigma)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
