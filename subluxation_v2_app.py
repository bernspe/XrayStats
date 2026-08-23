"""
Subluxation Detector v2 — Streamlit App

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run subluxation_v2_app.py

Detects vertebral subluxation via an end-pinned smoothed-spline distance model
(compute_smoothed_spline_distances / train_v2_classifier) and displays:
  - X-ray with raw spline, smoothed reference spline, deviation arrows
  - Segmental lines color-coded by predicted subluxation probability
  - Per-vertebra feature table
"""

import io
import json
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from annotation_analysis import (
    build_v2_training_table,
    compute_smoothed_spline_distances,
    compute_v2_segment_predictions,
    train_v2_classifier,
    spine_labels,
)
from bspline_python import interpolate_bspline, smooth_nodes_pinned
from prepare_outlier_data import fetch_img_urls

DATA_DIR = "data"
MODEL_FILE = f"{DATA_DIR}/subluxation_v2_model.pkl"
FEEDBACK_FILE = f"{DATA_DIR}/subluxation_v2_feedback.csv"
NAV_STATE_FILE = f"{DATA_DIR}/subluxation_v2_nav_state.json"
KONVA_STAGE_HEIGHT = 600

_TRAFFIC = {
    "green": "#4CAF50",
    "yellow": "#FFC107",
    "red": "#F44336",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_data():
    df4d = pd.read_pickle(f"{DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    with open(f"{DATA_DIR}/img_urls.pkl", "rb") as fh:
        img_urls = pickle.load(fh)
    return df4d, img_urls


@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


@st.cache_data(show_spinner="Messungen werden geladen …")
def build_measurement_index(_df4d: pd.DataFrame) -> pd.DataFrame:
    """One row per measurement with ≥1 Cobb-segmental segment AND usable node/spline data."""
    from annotation_analysis import build_cobb_segmental_lines_df
    rows = []
    for _, row in _df4d.iterrows():
        xmd = row.get("xray_measurement_data")
        if not isinstance(xmd, dict):
            continue
        if build_cobb_segmental_lines_df(xmd).empty:
            continue
        nodes = xmd.get("stateObject", {}).get("nodes", [])
        vr = xmd.get("stateObject", {}).get("vertebra_range", [0, 0])
        n_vertebrae = vr[1] - vr[0] + 1
        if len(nodes) < 3 or len(nodes) != n_vertebrae:
            continue
        rows.append({"image_id": row["id"], "patient_id": row["patient_id"],
                     "img_type": row["img_type"], "rater_name": row["rater_name"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as fh:
            return pickle.load(fh)
    return None


def save_model(bundle) -> None:
    with open(MODEL_FILE, "wb") as fh:
        pickle.dump(bundle, fh)


# ---------------------------------------------------------------------------
# Navigation state
# ---------------------------------------------------------------------------

def load_nav() -> dict | None:
    try:
        with open(NAV_STATE_FILE) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_nav(image_id: int, rater_name: str) -> None:
    with open(NAV_STATE_FILE, "w") as fh:
        json.dump({"image_id": image_id, "rater_name": rater_name}, fh)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _seg_color(proba: float, yellow_thr: float, red_thr: float) -> str:
    if np.isnan(proba):
        return "#888888"
    if proba >= red_thr:
        return _TRAFFIC["red"]
    if proba >= yellow_thr:
        return _TRAFFIC["yellow"]
    return _TRAFFIC["green"]


def build_figure(image_id: int, xmd: dict, img_urls: dict,
                 model_bundle, sigma: float,
                 gt_segments: set | None = None) -> plt.Figure:
    url = img_urls.get(image_id)
    xray = None
    if url:
        raw = fetch_image_bytes(url)
        if raw:
            xray = Image.open(io.BytesIO(raw)).convert("RGB")
    scale = (xray.height / KONVA_STAGE_HEIGHT) if xray else 1.0

    fig, ax = plt.subplots(figsize=(9, 9))
    if xray:
        ax.imshow(xray, origin="upper", aspect="equal")
    else:
        ax.set_facecolor("#111")
        ax.text(0.5, 0.5, "Bild nicht verfügbar", ha="center", va="center",
                color="white", transform=ax.transAxes, fontsize=12)

    nodes_raw = xmd.get("stateObject", {}).get("nodes", [])
    vr = xmd.get("stateObject", {}).get("vertebra_range", [0, 0])
    selected_vertebra = spine_labels[vr[0]: vr[1] + 1]
    nodes_canvas = [(n["x"], n["y"]) for n in nodes_raw]

    has_spline = len(nodes_canvas) >= 3 and len(nodes_canvas) == len(selected_vertebra)

    if has_spline:
        # Raw spline (gray dashed)
        raw_spline = interpolate_bspline(nodes_canvas, num_steps=200)
        if raw_spline:
            rx, ry = zip(*raw_spline)
            ax.plot([x * scale for x in rx], [y * scale for y in ry],
                    "-", color="#888888", linewidth=1.2, alpha=0.6, zorder=4, label="Spline (roh)")

        # Smoothed reference spline (white solid)
        smoothed = smooth_nodes_pinned(nodes_canvas, sigma=sigma)
        smooth_spline = interpolate_bspline(smoothed, num_steps=300)
        if smooth_spline:
            sx, sy = zip(*smooth_spline)
            ax.plot([x * scale for x in sx], [y * scale for y in sy],
                    "-", color="white", linewidth=2.0, alpha=0.85, zorder=5,
                    label="Referenz (geglättet)")

            # Deviation arrows: node → nearest point on smooth spline
            spline_arr = np.array(smooth_spline)
            for label, node in zip(selected_vertebra, nodes_canvas):
                node_px = np.array([node[0] * scale, node[1] * scale])
                spline_px = spline_arr * scale
                idx = int(np.hypot(spline_px[:, 0] - node_px[0],
                                   spline_px[:, 1] - node_px[1]).argmin())
                target = spline_px[idx]
                arrow_len = float(np.hypot(*(target - node_px)))
                if arrow_len > 3:
                    ax.annotate("", xy=target, xytext=node_px,
                                arrowprops=dict(arrowstyle="->", color="#FFD700",
                                                lw=1.2, alpha=0.8),
                                zorder=7)

        # Nodes (small dots)
        for node in nodes_canvas:
            ax.plot(node[0] * scale, node[1] * scale, "o",
                    color="#FFD700", markersize=4, zorder=8)

    # Segmental lines — color-coded by model prediction
    if model_bundle:
        model, scaler, y_thr, r_thr = model_bundle
        pred_df = compute_v2_segment_predictions(xmd, model, scaler, sigma=sigma)
        for _, seg in pred_df.iterrows():
            color = _seg_color(seg["proba"], y_thr, r_thr)
            for line_dict in (seg["upper_line"], seg["lower_line"]):
                ax.plot([line_dict["x1"] * scale, line_dict["x2"] * scale],
                        [line_dict["y1"] * scale, line_dict["y2"] * scale],
                        "-", color=color, linewidth=2.5, alpha=0.9, zorder=6)
            # Label the segment at midpoint of upper line
            ul = seg["upper_line"]
            mx = ((ul["x1"] + ul["x2"]) / 2) * scale
            my = ((ul["y1"] + ul["y2"]) / 2) * scale
            prob_txt = f"{seg['proba']:.0%}" if not np.isnan(seg["proba"]) else "?"
            is_gt = gt_segments and seg["segment"] in gt_segments
            label_txt = f"★ {seg['segment']} {prob_txt}" if is_gt else f"{seg['segment']} {prob_txt}"
            ax.text(mx + 8, my, label_txt,
                    color=color, fontsize=7.5, va="center", zorder=9, fontweight="bold" if is_gt else "normal",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"))
    else:
        # No model: draw lines in neutral color
        for a in xmd.get("angles", []):
            if a.get("cobbType") != "Cobb-segmental":
                continue
            for line_key in ("upperLine", "lowerLine"):
                ln = a.get(line_key)
                if ln:
                    ax.plot([ln["x1"] * scale, ln["x2"] * scale],
                            [ln["y1"] * scale, ln["y2"] * scale],
                            "--", color="#00E5FF", linewidth=1.8, alpha=0.8, zorder=6)

    legend_patches = [
        mpatches.Patch(color=_TRAFFIC["green"], label="unauffällig"),
        mpatches.Patch(color=_TRAFFIC["yellow"], label="verdächtig"),
        mpatches.Patch(color=_TRAFFIC["red"], label="Subluxation"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_title(f"ID {image_id}  σ={sigma}", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Subluxation Detector v2", layout="wide",
                       page_icon="🦴", initial_sidebar_state="expanded")
    st.title("🦴 Subluxation Detector v2")
    st.caption("Geglättete-Spline-Distanz-Modell · Logistische Regression auf End-Pinned-Referenzkurve")

    try:
        df4d, img_urls = load_data()
    except FileNotFoundError as e:
        st.error(f"Pickle-Dateien nicht gefunden: `{e}`")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("Bild-URLs auffrischen")
        token = st.text_input("Bearer Token", type="password")
        if st.button("URLs auffrischen"):
            if not token.strip():
                st.error("Token fehlt.")
            else:
                with st.spinner("Hole URLs …"):
                    try:
                        fresh = fetch_img_urls(token.strip())
                        with open(f"{DATA_DIR}/img_urls.pkl", "wb") as fh:
                            pickle.dump(fresh, fh)
                        load_data.clear()
                        st.success(f"{len(fresh)} URLs aufgefrischt.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        st.divider()

        st.subheader("Modell")
        sigma = st.slider("Glättungs-Sigma (σ)", min_value=0.5, max_value=6.0,
                          value=2.0, step=0.5,
                          help="Stärke der Gaussian-Glättung für die Referenz-Spline")

        model_bundle = load_model()
        if model_bundle:
            st.success("Modell geladen ✓")
        else:
            st.warning("Kein Modell vorhanden — bitte trainieren.")

        if st.button("Modell trainieren"):
            if not os.path.exists(FEEDBACK_FILE):
                st.error(f"{FEEDBACK_FILE} nicht gefunden.")
            else:
                feedback_df = pd.read_csv(FEEDBACK_FILE)
                with st.spinner("Trainiere …"):
                    train_df = build_v2_training_table(feedback_df, df4d, sigma=sigma)
                if train_df.empty or train_df["label"].sum() == 0:
                    st.error("Keine positiven Labels im Feedback gefunden.")
                else:
                    model_bundle = train_v2_classifier(train_df)
                    save_model(model_bundle)
                    pos = int(train_df["label"].sum())
                    total = len(train_df)
                    st.success(f"Trainiert · {pos}/{total} positive Segmente")
                    st.rerun()

        st.divider()
        st.subheader("Filter")
        index_df = build_measurement_index(df4d)
        img_types = sorted(index_df["img_type"].dropna().unique())
        sel_types = st.multiselect("img_type", img_types, default=img_types)
        only_sublux = st.checkbox("Nur Fälle mit Subluxation (Ground Truth)",
                                  help="Filtert auf Messungen, die im Feedback als "
                                       "'Subluxation vorhanden' markiert wurden.")

    # --- Measurement index ---
    ranked = index_df[index_df["img_type"].isin(sel_types)].copy()
    if only_sublux and os.path.exists(FEEDBACK_FILE):
        fb = pd.read_csv(FEEDBACK_FILE)
        sublux_pairs = set(
            zip(fb.loc[fb["sublux_label"] == "Subluxation vorhanden", "image_id"],
                fb.loc[fb["sublux_label"] == "Subluxation vorhanden", "rater_name"])
        )
        ranked = ranked[ranked.apply(
            lambda r: (r["image_id"], r["rater_name"]) in sublux_pairs, axis=1
        )]
    ranked = ranked.sort_values(["image_id", "rater_name"]).reset_index(drop=True)
    if ranked.empty:
        st.warning("Keine Messungen für die gewählten Filter.")
        return

    st.sidebar.caption(f"**{len(ranked)}** Messungen · **{ranked['image_id'].nunique()}** Bilder")

    # --- Navigation ---
    if "nav_idx" not in st.session_state:
        last = load_nav()
        idx = 0
        if last:
            matches = ranked.index[
                (ranked["image_id"] == last.get("image_id")) &
                (ranked["rater_name"] == last.get("rater_name"))
            ]
            if len(matches):
                idx = int(matches[0])
        st.session_state.nav_idx = idx

    idx = min(max(st.session_state.nav_idx, 0), len(ranked) - 1)
    col_prev, col_pos, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("◀ Zurück", disabled=(idx <= 0), use_container_width=True):
            idx -= 1
    with col_next:
        if st.button("Weiter ▶", disabled=(idx >= len(ranked) - 1), use_container_width=True):
            idx += 1
    idx = min(max(idx, 0), len(ranked) - 1)  # re-clamp after button adjustments
    st.session_state.nav_idx = idx

    sel = ranked.iloc[idx]
    image_id = int(sel["image_id"])
    rater_name = sel["rater_name"]
    with col_pos:
        st.markdown(
            f"<div style='text-align:center'>{idx + 1} / {len(ranked)}  ·  "
            f"{sel['img_type']}  [{sel['patient_id']}]  ID {image_id}  {rater_name}</div>",
            unsafe_allow_html=True,
        )
    save_nav(image_id, rater_name)

    measurement = df4d[(df4d["id"] == image_id) & (df4d["rater_name"] == rater_name)].iloc[0]
    xmd = measurement["xray_measurement_data"]

    # Load GT segments for this measurement from feedback (if available)
    gt_segments: set = set()
    if os.path.exists(FEEDBACK_FILE):
        fb_all = pd.read_csv(FEEDBACK_FILE)
        fb_row = fb_all[
            (fb_all["image_id"] == image_id) &
            (fb_all["rater_name"] == rater_name) &
            (fb_all["sublux_label"] == "Subluxation vorhanden")
        ]
        if not fb_row.empty:
            levels_str = fb_row.iloc[-1]["sublux_levels"]
            if isinstance(levels_str, str) and levels_str.strip():
                gt_segments = set(levels_str.split(";"))

    # --- Main content ---
    col_img, col_table = st.columns([2, 1])
    with col_img:
        fig = build_figure(image_id, xmd, img_urls, model_bundle, sigma, gt_segments)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_table:
        dist_df = compute_smoothed_spline_distances(xmd, sigma=sigma)
        if dist_df.empty:
            st.info("Keine Spline-Daten (zu wenig Knoten?).")
        else:
            display = dist_df[["vertebra", "node_dist_norm", "seg_dist_norm"]].copy()
            display.columns = ["Vertebra", "node_dist", "seg_dist"]
            display = display.set_index("Vertebra").round(3)

            # Highlight rows if model available
            if model_bundle:
                model, scaler, y_thr, r_thr = model_bundle
                pred_df = compute_v2_segment_predictions(xmd, model, scaler, sigma=sigma)
                if not pred_df.empty:
                    st.subheader("Segment-Vorhersagen")
                    seg_display = pred_df[["segment", "node_delta_above", "node_delta_below",
                                          "spike_score", "proba"]].copy()
                    seg_display["GT"] = seg_display["segment"].apply(
                        lambda s: "★" if s in gt_segments else "")
                    seg_display.columns = ["Segment", "Δ_oben", "Δ_unten", "spike", "Prob", "GT"]
                    seg_display = seg_display.set_index("Segment").round(3)

                    def _color_prob(val):
                        if pd.isna(val):
                            return ""
                        if val >= r_thr:
                            return "background-color: #5a1a1a"
                        if val >= y_thr:
                            return "background-color: #4a3a00"
                        return "background-color: #1a3a1a"

                    st.dataframe(seg_display.style.applymap(_color_prob, subset=["Prob"]),
                                 use_container_width=True)

            st.subheader("Vertebra-Distanzen")
            st.dataframe(display, use_container_width=True)


if __name__ == "__main__":
    main()
