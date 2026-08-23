"""
Annotator Timing Analysis — Messdauer je Annotator
Δt = last_modified - created_at pro MeasureData

Start:
    cd /Users/peterbernstein/Python/XrayStats
    streamlit run annotation_timing_app.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

API_URL   = "https://xray-api.renecol.org"
CACHE_FILE = Path("data/annotation_timing_cache.json")

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _fetch_raw(token: str) -> list[dict]:
    resp = requests.get(
        f"{API_URL}/api/imagedata/",
        params={"detailed": "True", "complete": "True"},
        headers={"Authorization": f"Bearer {token}"},
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


def raw_to_df(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for img in raw:
        img_type = img.get("img_type") or "Unknown"
        for md in img.get("measure_dataset", []):
            u = md.get("user") or {}
            name = f"{u.get('first_name', '')} {u.get('last_name', '')}".strip() or u.get("username", "Unknown")
            rows.append({
                "annotator":     name,
                "img_type":      img_type,
                "created_at":    md.get("created_at"),
                "last_modified": md.get("last_modified"),
            })
    df = pd.DataFrame(rows)
    df["created_at"]    = pd.to_datetime(df["created_at"],    utc=True)
    df["last_modified"] = pd.to_datetime(df["last_modified"], utc=True)
    df["duration_s"] = (df["last_modified"] - df["created_at"]).dt.total_seconds()
    # ponytail: drop negative/zero durations (same-second saves = incomplete sessions)
    return df[df["duration_s"] > 0].copy()


def ci95_bootstrap(x: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    medians = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def fmt_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s" if m else f"{sec}s"


def fmt_hms(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Annotator Timing")
st.caption("Δt = last_modified − created_at pro Messung")

cached = load_cache()

with st.sidebar:
    st.header("Daten")
    if cached:
        st.success(f"Cache vorhanden ({len(cached)} Bilder)")
    else:
        st.warning("Kein Cache — bitte laden.")

    token = st.text_input("Bearer Token", type="password")
    fetch_btn = st.button("Von API laden & cachen", disabled=not token.strip())
    if fetch_btn:
        with st.spinner("Lade von API …"):
            cached = _fetch_raw(token)
            save_cache(cached)
        st.success(f"{len(cached)} Bilder gecacht.")

if not cached:
    st.info("Bitte Token eingeben und 'Von API laden & cachen' klicken.")
    st.stop()

df = raw_to_df(cached)

annotators = sorted(df["annotator"].unique())
selected = st.multiselect("Annotatoren", annotators, default=annotators)
df = df[df["annotator"].isin(selected)]

img_types = sorted(df["img_type"].unique())
selected_types = st.multiselect("Bildtypen (img_type)", img_types, default=img_types)
df = df[df["img_type"].isin(selected_types)]

# percentile cap for display & CI sum
pct_cap = st.slider("Obere Perzentile abschneiden (Anzeige + CI-Summe)", 90, 100, 99)
cap = np.percentile(df["duration_s"], pct_cap)
df_plot = df[df["duration_s"] <= cap].copy()

st.markdown(f"**{len(df):,} Messungen** gesamt · {len(df_plot):,} nach Cutoff ({fmt_seconds(cap)})")

# ── Statistiktabelle ──────────────────────────────────────────────────────────

def make_stat_row(label_dict: dict, x: np.ndarray, cap: float) -> dict:
    if len(x) == 0:
        return {**label_dict, "N": 0, "Median": "—", "95% CI low": "—",
                "95% CI high": "—", "Summe im CI": "—", "Gesamt-Summe": "—"}
    med = np.median(x)
    lo, hi = ci95_bootstrap(x)
    ci_mask = (x >= lo) & (x <= hi) & (x <= cap)
    return {
        **label_dict,
        "N":            len(x),
        "Median":       fmt_seconds(med),
        "95% CI low":   fmt_seconds(lo),
        "95% CI high":  fmt_seconds(hi),
        "Summe im CI":  fmt_hms(x[ci_mask].sum()),
        "Gesamt-Summe": fmt_hms(x.sum()),
    }

stat_rows = []
for (ann, itype), grp in df.groupby(["annotator", "img_type"]):
    stat_rows.append(make_stat_row({"Annotator": ann, "img_type": itype}, grp["duration_s"].values, cap))

# Subtotals per annotator
for ann, grp in df.groupby("annotator"):
    stat_rows.append(make_stat_row({"Annotator": ann, "img_type": "∑ Annotator"}, grp["duration_s"].values, cap))

# Total row
stat_rows.append(make_stat_row({"Annotator": "TOTAL", "img_type": ""}, df["duration_s"].values, cap))

display_cols = ["Annotator", "img_type", "N", "Median", "95% CI low", "95% CI high", "Summe im CI", "Gesamt-Summe"]
stat_df = pd.DataFrame(stat_rows)[display_cols]

st.subheader("Statistik je Annotator und Bildtyp")
st.dataframe(stat_df.set_index(["Annotator", "img_type"]), use_container_width=True)

# ── Zeitverlauf ───────────────────────────────────────────────────────────────

st.subheader("Messdauer über Messzeitpunkt")

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
color_map  = {ann:   COLORS[i % len(COLORS)]  for i, ann   in enumerate(sorted(selected))}
marker_map = {itype: MARKERS[i % len(MARKERS)] for i, itype in enumerate(sorted(selected_types))}

fig_t, ax_t = plt.subplots(figsize=(11, 4.5), constrained_layout=True)

_labeled_ann   = set()
_labeled_itype = set()

for ann in selected:
    t0 = df[df["annotator"] == ann]["created_at"].min()
    for itype in selected_types:
        grp = df_plot[(df_plot["annotator"] == ann) & (df_plot["img_type"] == itype)].sort_values("created_at")
        if grp.empty:
            continue
        days = (grp["created_at"] - t0).dt.total_seconds() / 86400
        # legend: one entry per annotator (color) + one per img_type (marker shape)
        lbl_ann   = ann   if ann   not in _labeled_ann   else None
        lbl_itype = itype if itype not in _labeled_itype else None
        ax_t.scatter(days, grp["duration_s"] / 60,
                     s=22, alpha=0.55,
                     color=color_map[ann], marker=marker_map[itype],
                     label=lbl_ann)
        if lbl_itype:
            # invisible dummy for img_type legend entry
            ax_t.scatter([], [], s=22, color="gray", marker=marker_map[itype], label=f"[{itype}]")
        _labeled_ann.add(ann)
        _labeled_itype.add(itype)

    # rolling median über alle img_types zusammen je Annotator
    grp_all = df_plot[df_plot["annotator"] == ann].sort_values("created_at")
    if len(grp_all) >= 7:
        t0 = df[df["annotator"] == ann]["created_at"].min()
        days_all = (grp_all["created_at"] - t0).dt.total_seconds() / 86400
        roll = grp_all["duration_s"].rolling(7, center=True).median() / 60
        ax_t.plot(days_all, roll, color=color_map[ann], linewidth=1.4, alpha=0.85)

ax_t.set_xlabel("Tage seit erster Messung (t₀ je Annotator)")
ax_t.set_ylabel("Messdauer (min)")
ax_t.set_title("Messdauer über Zeit — alle Annotatoren (relativ zu t₀)")
ax_t.legend(fontsize=7, markerscale=1.4, ncol=2)
ax_t.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))

st.pyplot(fig_t)

# ── Histogramm ────────────────────────────────────────────────────────────────

st.subheader("Verteilung der Messdauer")

n_ann = len(selected)
fig, axes = plt.subplots(
    n_ann, 1, figsize=(9, 3.2 * n_ann), squeeze=False, constrained_layout=True
)

type_colors = {itype: COLORS[i % len(COLORS)] for i, itype in enumerate(sorted(selected_types))}

for ax, ann in zip(axes[:, 0], selected):
    full_grp = df[df["annotator"] == ann]["duration_s"].values
    if len(full_grp) == 0:
        ax.set_visible(False)
        continue

    med = np.median(full_grp)
    lo, hi = ci95_bootstrap(full_grp)

    # shared bin edges across all img_types for this annotator
    capped = df_plot[df_plot["annotator"] == ann]["duration_s"].values
    if len(capped) == 0:
        ax.set_visible(False)
        continue
    bins = np.linspace(capped.min(), capped.max(), 41)

    # stacked histogram
    type_data = [
        df_plot[(df_plot["annotator"] == ann) & (df_plot["img_type"] == it)]["duration_s"].values / 60
        for it in selected_types
    ]
    ax.hist(type_data, bins=bins / 60, stacked=True,
            color=[type_colors[it] for it in selected_types],
            label=selected_types, edgecolor="white", linewidth=0.3)

    ax.axvline(med / 60, color="#DD4444", linewidth=1.8, label=f"Median {fmt_seconds(med)}")
    ax.axvspan(lo / 60, min(hi, cap) / 60, color="#DD4444", alpha=0.15,
               label=f"95% CI [{fmt_seconds(lo)}, {fmt_seconds(hi)}]")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("Messdauer (min)")
    ax.set_ylabel("Anzahl")
    ax.set_title(f"{ann}  (N={len(full_grp)})")
    ax.legend(fontsize=7, ncol=2)

st.pyplot(fig)
