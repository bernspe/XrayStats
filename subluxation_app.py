"""
Subluxation Line Viewer — Streamlit App

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run subluxation_app.py

Plain browsing + ground-truth-labeling tool for Cobb-segmental measurements:
shows the raw annotated upper/lower lines per segment and records subluxation
labels (data/subluxation_feedback.csv) for whatever future analysis.

NOTE: an earlier version of this app color-coded lines by a trained model's
predicted subluxation probability (RandomForest on spline/z-score features,
session-validated CV AUC ~0.71). That model showed no real predictive
quality once used on actual images and was retired on 2026-07-08 — see
project memory 'subluxation-scoring-calibration' for the full writeup. The
underlying feature-engineering code is still in annotation_analysis.py
(compute_subluxation_scores, compute_spline_line_features,
compute_spline_smoothing_deviation, build_segment_training_table, etc.) in
case a future attempt wants to reference it, but nothing here calls it.
"""

import csv
import io
import json
import os
import pickle
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from annotation_analysis import spine_labels, build_cobb_segmental_lines_df
from prepare_outlier_data import fetch_img_urls

_SPINE_IDX = {lbl: i for i, lbl in enumerate(spine_labels)}

DATA_DIR = "data"
FEEDBACK_FILE = f"{DATA_DIR}/subluxation_feedback.csv"
MODEL_VERDICT_FILE = f"{DATA_DIR}/subluxation_model_verdicts.csv"
NAV_STATE_FILE = f"{DATA_DIR}/subluxation_nav_state.json"
KONVA_STAGE_HEIGHT = 600
FEEDBACK_FIELDS = ["timestamp", "image_id", "patient_id", "rater_name", "img_type",
                   "max_score", "verdict", "comment", "sublux_label", "sublux_levels"]


def segment_options(xmd: dict) -> list[str]:
    """Segment labels ('L4/L5') straight from the Cobb-segmental angle entries."""
    segs = []
    for a in xmd.get("angles", []):
        if a.get("cobbType") != "Cobb-segmental":
            continue
        ca = a.get("cobbAngle", {})
        vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
        if vu and vl:
            segs.append((_SPINE_IDX.get(vu, 99), f"{vu}/{vl}"))
    return [s for _, s in sorted(segs)]


def load_last_position() -> dict | None:
    try:
        with open(NAV_STATE_FILE) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_last_position(image_id: int, rater_name: str) -> None:
    with open(NAV_STATE_FILE, "w") as fh:
        json.dump({"image_id": image_id, "rater_name": rater_name}, fh)


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


@st.cache_data(show_spinner="Messungen werden eingelesen …")
def build_ranking(_df4d: pd.DataFrame) -> pd.DataFrame:
    """One row per measurement with >=1 Cobb-segmental line."""
    rows = []
    for _, row in _df4d.iterrows():
        xmd = row.get("xray_measurement_data")
        if not isinstance(xmd, dict):
            continue
        if build_cobb_segmental_lines_df(xmd).empty:
            continue
        rows.append({
            "image_id": row["id"],
            "patient_id": row["patient_id"],
            "img_type": row["img_type"],
            "rater_name": row["rater_name"],
        })
    return pd.DataFrame(rows)


def _draw_raw_segmental_lines(ax, xmd: dict, scale: float) -> None:
    """Every upperLine/lowerLine straight from the API, one pair per
    Cobb-segmental angle entry. Adjacent entries share a line for their
    common vertebra; if the two dashed lines visibly diverge instead of
    overlapping, that's inconsistent original annotation."""
    for a in xmd.get("angles", []):
        if a.get("cobbType") != "Cobb-segmental":
            continue
        for line_key, color in (("upperLine", "#00E5FF"), ("lowerLine", "#FF00C8")):
            ln = a.get(line_key)
            if not ln:
                continue
            ax.plot([ln["x1"] * scale, ln["x2"] * scale],
                    [ln["y1"] * scale, ln["y2"] * scale],
                    "--", color=color, linewidth=2.0, alpha=0.9, zorder=6)


def build_figure(image_id: int, xmd: dict, img_urls: dict):
    url = img_urls.get(image_id)
    xray = None
    if url:
        raw = fetch_image_bytes(url)
        if raw:
            xray = Image.open(io.BytesIO(raw)).convert("RGB")
    scale = (xray.height / KONVA_STAGE_HEIGHT) if xray else 1.0

    fig, ax = plt.subplots(figsize=(9, 8))
    if xray:
        ax.imshow(xray, origin="upper", aspect="equal")
    else:
        ax.set_facecolor("#222")
        ax.text(0.5, 0.5, "Bild nicht verfügbar\n(URL abgelaufen?)",
                ha="center", va="center", color="white",
                transform=ax.transAxes, fontsize=12)

    _draw_raw_segmental_lines(ax, xmd, scale)
    legend_handles = [
        plt.Line2D([0], [0], color=c, lw=1.5, ls="--", label=label)
        for c, label in (("#00E5FF", "upperLine"), ("#FF00C8", "lowerLine"))
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.8)
    ax.set_title(f"ID {image_id}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def save_feedback(record: dict) -> None:
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEEDBACK_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def main():
    st.set_page_config(page_title="Subluxation Line Viewer", layout="wide", page_icon="🦴",
                       initial_sidebar_state="expanded")
    st.title("🦴 Subluxation Line Viewer")
    st.caption("Reiner Browser + Ground-Truth-Labeling. Die automatische Modell-Bewertung "
               "wurde am 2026-07-08 verworfen (keine erkennbare Vorhersagequalität in der Praxis).")

    try:
        df4d, img_urls = load_data()
    except FileNotFoundError as e:
        st.error(f"Pickle-Dateien nicht gefunden: `{e}`")
        return

    with st.sidebar:
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
                        load_data.clear()
                        st.success(f"{len(fresh_urls)} URLs aufgefrischt.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fehlgeschlagen: {e}")
        st.divider()

    ranking = build_ranking(df4d)
    if ranking.empty:
        st.warning("Keine Messungen mit auswertbaren Cobb-segmental-Segmenten gefunden.")
        return

    feedback_df = pd.read_csv(FEEDBACK_FILE) if os.path.exists(FEEDBACK_FILE) else pd.DataFrame(
        columns=FEEDBACK_FIELDS)
    model_verdicts_df = pd.read_csv(MODEL_VERDICT_FILE) if os.path.exists(MODEL_VERDICT_FILE) else pd.DataFrame(
        columns=["image_id", "rater_name"])
    seen = (set(zip(feedback_df["image_id"], feedback_df["rater_name"])) |
            set(zip(model_verdicts_df["image_id"], model_verdicts_df["rater_name"])))
    if seen:
        ranking = ranking[~ranking.apply(lambda r: (r["image_id"], r["rater_name"]) in seen, axis=1)]
    if ranking.empty:
        st.success("Alle Messungen mit auswertbaren Segmenten wurden bereits bewertet. 🎉")
        return

    with st.sidebar:
        st.header("Filter")
        img_types = sorted(ranking["img_type"].dropna().unique())
        sel_types = st.multiselect("img_type", img_types, default=img_types)
        st.divider()
        st.caption(f"**{len(ranking)}** unbewertete Messungen  ·  **{ranking['image_id'].nunique()}** Bilder")

    ranked = (ranking[ranking["img_type"].isin(sel_types)]
              .sort_values(["image_id", "rater_name"]).reset_index(drop=True))
    if ranked.empty:
        st.warning("Keine Messungen für die gewählten Filter.")
        return

    if "nav_idx" not in st.session_state:
        last = load_last_position()
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

    col_prev, col_pos, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Zurück", disabled=(idx <= 0), use_container_width=True):
            idx -= 1
    with col_next:
        if st.button("Weiter ▶", disabled=(idx >= len(ranked) - 1), use_container_width=True):
            idx += 1
    st.session_state.nav_idx = idx

    sel_row = ranked.iloc[idx]
    image_id = int(sel_row["image_id"])
    rater_name = sel_row["rater_name"]
    with col_pos:
        st.markdown(
            f"<div style='text-align:center'>{idx + 1} / {len(ranked)}  ·  "
            f"{sel_row['img_type']}  [{sel_row['patient_id']}]  ID {image_id}  "
            f"{rater_name}</div>",
            unsafe_allow_html=True,
        )

    save_last_position(image_id, rater_name)

    measurement = df4d[(df4d["id"] == image_id) & (df4d["rater_name"] == rater_name)].iloc[0]
    xmd = measurement["xray_measurement_data"]

    key_label = f"sublux_label_{image_id}_{rater_name}"
    key_levels = f"sublux_levels_{image_id}_{rater_name}"
    key_comment = f"comment_{image_id}_{rater_name}"

    def _save_and_advance(sublux_label: str, sublux_levels: list[str], comment: str) -> None:
        save_feedback({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_id": image_id,
            "patient_id": sel_row["patient_id"],
            "rater_name": rater_name,
            "img_type": sel_row["img_type"],
            "max_score": "",
            "verdict": "",
            "comment": comment,
            "sublux_label": sublux_label,
            "sublux_levels": ";".join(sublux_levels),
        })
        # Don't increment: this image just got filtered out of `ranked` (see
        # "seen" filtering above), so everything after it shifts down by one —
        # staying at the same numeric index already lands on the next image.
        st.session_state.nav_idx = idx
        st.rerun()

    if st.button("✅ Keine Subluxation & Weiter", use_container_width=True):
        _save_and_advance("keine Subluxation", [], "")

    col_img, col_table = st.columns([2, 1])
    with col_img:
        fig = build_figure(image_id, xmd, img_urls)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col_table:
        lines_df = build_cobb_segmental_lines_df(xmd)
        st.dataframe(
            lines_df[["vertebra", "length", "angle_deg"]].round(1).set_index("vertebra"),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Ground-Truth-Details")
    st.radio(
        "Subluxation im Bild?",
        ["keine Subluxation", "Subluxation vorhanden", "unklar"],
        horizontal=True,
        key=key_label,
    )
    st.multiselect(
        "Betroffene Segmente (optional)",
        options=segment_options(xmd),
        key=key_levels,
    )
    st.text_input("Kommentar (optional)", key=key_comment)
    if st.button("💾 Speichern & Weiter", use_container_width=True):
        _save_and_advance(
            st.session_state.get(key_label, "keine Subluxation"),
            st.session_state.get(key_levels, []),
            st.session_state.get(key_comment, ""),
        )

    if os.path.exists(FEEDBACK_FILE):
        st.divider()
        st.caption("Bisheriges Feedback")
        st.dataframe(pd.read_csv(FEEDBACK_FILE).tail(20), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
