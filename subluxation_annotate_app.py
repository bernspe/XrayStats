"""
Subluxation Annotate App — Streamlit

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run subluxation_annotate_app.py

Klick-basiertes Annotierungswerkzeug:
  - X-ray mit Segmentlinien und klickbaren Markern
  - Klick auf Marker → Segment als pathologisch togglen (grün ↔ rot)
  - Buttons als Fallback / Tastatur-Alternative
  - Speichert nach data/subluxation_v2_feedback.csv
"""

import csv
import io
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

from annotation_analysis import build_cobb_segmental_lines_df, spine_labels
from prepare_outlier_data import fetch_img_urls

DATA_DIR = "data"
FEEDBACK_FILE = f"{DATA_DIR}/subluxation_v2_feedback.csv"
NAV_FILE = f"{DATA_DIR}/subluxation_annotate_nav.json"
KONVA_STAGE_HEIGHT = 600
FIELDS = ["timestamp", "image_id", "patient_id", "rater_name",
          "img_type", "sublux_label", "sublux_levels"]
_SPINE_IDX = {lbl: i for i, lbl in enumerate(spine_labels)}


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
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


@st.cache_data(show_spinner="Messungen werden geladen …")
def build_index(_df4d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in _df4d.iterrows():
        xmd = row.get("xray_measurement_data")
        if not isinstance(xmd, dict):
            continue
        if build_cobb_segmental_lines_df(xmd).empty:
            continue
        nodes = xmd.get("stateObject", {}).get("nodes", [])
        vr = xmd.get("stateObject", {}).get("vertebra_range", [0, 0])
        if len(nodes) < 3 or len(nodes) != (vr[1] - vr[0] + 1):
            continue
        rows.append({"image_id": row["id"], "patient_id": row["patient_id"],
                     "img_type": row["img_type"], "rater_name": row["rater_name"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_nav() -> dict | None:
    try:
        with open(NAV_FILE) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_nav(image_id: int, rater_name: str) -> None:
    with open(NAV_FILE, "w") as fh:
        json.dump({"image_id": image_id, "rater_name": rater_name}, fh)


def load_feedback() -> pd.DataFrame:
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=FIELDS)


def append_feedback(record: dict) -> None:
    exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(record)


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def get_segments(xmd: dict) -> list[str]:
    """Sorted list of 'VU/VL' segment strings from Cobb-segmental angles."""
    segs = []
    for a in xmd.get("angles", []):
        if a.get("cobbType") != "Cobb-segmental":
            continue
        ca = a.get("cobbAngle", {})
        vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
        if vu and vl:
            segs.append((_SPINE_IDX.get(vu, 99), f"{vu}/{vl}"))
    return [s for _, s in sorted(segs)]


def get_segment_lines(xmd: dict) -> list[dict]:
    """List of {seg, vu, vl, upper_line, lower_line} dicts, spine-sorted."""
    result = []
    for a in xmd.get("angles", []):
        if a.get("cobbType") != "Cobb-segmental":
            continue
        ca = a.get("cobbAngle", {})
        vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
        ul, ll = a.get("upperLine"), a.get("lowerLine")
        if vu and vl and ul and ll:
            result.append({
                "seg": f"{vu}/{vl}", "vu": vu, "vl": vl,
                "spine_idx": _SPINE_IDX.get(vu, 99),
                "upper_line": ul, "lower_line": ll,
            })
    return sorted(result, key=lambda x: x["spine_idx"])


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------

def build_figure(xmd: dict, img_bytes: bytes | None,
                 selected: set[str], scale: float) -> go.Figure:
    fig = go.Figure()

    img_w, img_h = 500, 800
    if img_bytes:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_arr = np.array(img)
        img_w, img_h = img.width, img.height
        fig.add_trace(go.Image(z=img_arr, hoverinfo="skip"))

    for entry in get_segment_lines(xmd):
        seg = entry["seg"]
        ul, ll = entry["upper_line"], entry["lower_line"]
        color = "#EF5350" if seg in selected else "#66BB6A"

        # Endplate lines (non-interactive, visual only)
        for line in (ul, ll):
            fig.add_trace(go.Scatter(
                x=[line["x1"] * scale, line["x2"] * scale],
                y=[line["y1"] * scale, line["y2"] * scale],
                mode="lines",
                line=dict(color=color, width=3.5),
                hoverinfo="skip",
                showlegend=False,
            ))

        # Clickable marker at right end of upper line, offset further right
        mx = max(ul["x1"], ul["x2"]) * scale + 30
        my = (ul["y1"] + ul["y2"]) / 2 * scale
        fig.add_trace(go.Scatter(
            x=[mx], y=[my],
            mode="markers+text",
            marker=dict(size=28, color=color, opacity=0.9,
                        symbol="circle",
                        line=dict(color="white", width=2)),
            text=[f"  {seg}"],
            textposition="middle right",
            textfont=dict(color="white", size=13, family="Arial Black"),
            name=seg,
            customdata=[seg],  # single string per point
            hovertemplate=f"<b>{seg}</b>  {'✗ pathologisch' if seg in selected else '✓ normal'}"
                          f"<br><i>Klicken zum Umschalten</i><extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        xaxis=dict(visible=False, range=[0, img_w], fixedrange=True),
        yaxis=dict(visible=False, range=[img_h, 0], fixedrange=True),
        dragmode=False,
        height=680,
        clickmode="event",
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Subluxation Annotieren", layout="wide",
                       page_icon="✏️", initial_sidebar_state="collapsed")

    try:
        df4d, img_urls = load_data()
    except FileNotFoundError as e:
        st.error(f"Pickle-Dateien nicht gefunden: {e}")
        return

    # Sidebar
    with st.sidebar:
        rater_name = st.text_input("Rater-Name", value="Peter")
        show_annotated = st.checkbox("Bereits annotierte Bilder anzeigen")
        img_types_all = sorted(build_index(df4d)["img_type"].dropna().unique())
        sel_types = st.multiselect("img_type", img_types_all, default=img_types_all)
        st.divider()
        st.subheader("URLs auffrischen")
        token = st.text_input("Bearer Token", type="password")
        if st.button("URLs auffrischen"):
            if token.strip():
                with st.spinner("…"):
                    try:
                        fresh = fetch_img_urls(token.strip())
                        with open(f"{DATA_DIR}/img_urls.pkl", "wb") as fh:
                            pickle.dump(fresh, fh)
                        load_data.clear()
                        st.success(f"{len(fresh)} URLs aufgefrischt.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    index_df = build_index(df4d)
    feedback_df = load_feedback()

    ranked = index_df[index_df["img_type"].isin(sel_types)].copy()
    if not show_annotated and not feedback_df.empty:
        done = set(zip(feedback_df["image_id"], feedback_df["rater_name"]))
        ranked = ranked[~ranked.apply(
            lambda r: (r["image_id"], r["rater_name"]) in done, axis=1)]
    ranked = ranked.sort_values(["image_id", "rater_name"]).reset_index(drop=True)

    if ranked.empty:
        st.success("✅ Alle Messungen annotiert!")
        if not feedback_df.empty:
            st.dataframe(feedback_df.tail(30), use_container_width=True, hide_index=True)
        return

    # Progress
    total = len(index_df[index_df["img_type"].isin(sel_types)])
    done_count = total - len(ranked)
    st.progress(done_count / total if total else 0,
                text=f"{done_count} / {total} annotiert")

    # Navigation state
    if "nav_idx" not in st.session_state:
        last = load_nav()
        idx = 0
        if last:
            m = ranked.index[(ranked["image_id"] == last.get("image_id")) &
                             (ranked["rater_name"] == last.get("rater_name"))]
            if len(m):
                idx = int(m[0])
        st.session_state.nav_idx = idx

    c_prev, c_info, c_next = st.columns([1, 8, 1])
    with c_prev:
        if st.button("◀", use_container_width=True,
                     disabled=(st.session_state.nav_idx <= 0)):
            st.session_state.nav_idx -= 1
    with c_next:
        if st.button("▶", use_container_width=True,
                     disabled=(st.session_state.nav_idx >= len(ranked) - 1)):
            st.session_state.nav_idx += 1
    st.session_state.nav_idx = min(max(st.session_state.nav_idx, 0), len(ranked) - 1)
    idx = st.session_state.nav_idx

    sel_row = ranked.iloc[idx]
    image_id = int(sel_row["image_id"])
    with c_info:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"<b>{idx + 1} / {len(ranked)}</b>  ·  {sel_row['img_type']}  "
            f"[{sel_row['patient_id']}]  ID {image_id}  {sel_row['rater_name']}"
            f"</div>", unsafe_allow_html=True)
    save_nav(image_id, sel_row["rater_name"])

    measurement = df4d[
        (df4d["id"] == image_id) & (df4d["rater_name"] == sel_row["rater_name"])
    ].iloc[0]
    xmd = measurement["xray_measurement_data"]

    url = img_urls.get(image_id)
    img_bytes = fetch_image_bytes(url) if url else None
    scale = 1.0
    if img_bytes:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        scale = img.height / KONVA_STAGE_HEIGHT

    # Per-image selection state
    sel_key = f"sel_{image_id}_{sel_row['rater_name']}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set()
    selected: set = st.session_state[sel_key]

    segments = get_segments(xmd)

    # Plotly chart with on_select
    chart_key = f"chart_{image_id}_{sel_row['rater_name']}"
    fig = build_figure(xmd, img_bytes, selected, scale)
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=chart_key,
    )

    # Toggle clicked segment
    if event and event.selection and event.selection.points:
        for pt in event.selection.points:
            seg = pt.get("customdata")
            if seg and isinstance(seg, str):
                if seg in selected:
                    selected.discard(seg)
                else:
                    selected.add(seg)
        st.rerun()

    # Segment toggle buttons (backup / keyboard-friendly)
    if segments:
        cols = st.columns(len(segments))
        for col, seg in zip(cols, segments):
            is_sel = seg in selected
            if col.button(
                f"🔴 {seg}" if is_sel else f"🟢 {seg}",
                key=f"segbtn_{seg}_{idx}",
                use_container_width=True,
            ):
                if is_sel:
                    selected.discard(seg)
                else:
                    selected.add(seg)
                st.rerun()

    # Action buttons
    st.divider()

    def _save_and_advance(levels: list[str]) -> None:
        # rater_name = original measurement rater (used for df4d join in training)
        # annotator  = person doing the subluxation labeling (sidebar)
        append_feedback({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_id": image_id,
            "patient_id": sel_row["patient_id"],
            "rater_name": sel_row["rater_name"],
            "img_type": sel_row["img_type"],
            "sublux_label": "Subluxation vorhanden" if levels else "keine Subluxation",
            "sublux_levels": ";".join(levels),
        })
        st.session_state[sel_key] = set()
        st.session_state.nav_idx = min(idx + 1, len(ranked) - 1)
        st.rerun()

    c_ok, c_save = st.columns(2)
    with c_ok:
        if st.button("✅  Keine Subluxation → Weiter", use_container_width=True):
            _save_and_advance([])
    with c_save:
        save_label = (f"💾  Speichern & Weiter  ({', '.join(sorted(selected))})"
                      if selected else "💾  Speichern & Weiter")
        if st.button(save_label, use_container_width=True,
                     type="primary", disabled=not selected):
            _save_and_advance(sorted(selected))


if __name__ == "__main__":
    main()
