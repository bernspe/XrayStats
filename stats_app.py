"""
Cobb-Winkel Statistik — Streamlit App

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run stats_app.py
"""
import warnings
warnings.filterwarnings("ignore")

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from annotation_analysis import (
    compute_icc, build_angles_df,
    compute_icc_by_segment,
    REGIONAL_COBB_TYPES, SEGMENTAL_COBB_TYPES, LUMBAR_SEGMENT_PAIRS,
)
from bspline_python import convertMeasurementBorderToSpineRegion
import annotation_analysis as aa

DATA_DIR = "data"

DIMS = {
    "cobbType":    "Cobb-Typ",
    "img_type":    "Bildtyp",
    "rater_name":  "User/Rater",
    "main_region": "Hauptregion (a.p.: CT/PT/MT/TL/U-LU/L-LU, seitlich: TK/TL/LL)",
}
ICC_BREAKDOWN_OPTIONS = {"(keine)": None, "Bildtyp": "img_type"}
SIGMA_OPTIONS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
REGION_ORDER = ["CT", "PT", "MT", "TK", "TL", "U-LU", "L-LU", "LU", "LL"]
# Lateral films: an angle whose borders sit on a sagittal span is that span — the
# same tolerances lateral_regional_angles uses for a single drawn pair.
_LATERAL_BORDERS = {"TK": (aa._TK_UPPER, aa._TK_LOWER), "TL": (aa._TL_UPPER, aa._TL_LOWER),
                    "LL": (aa._LL_UPPER, aa._LL_LOWER)}
COMPARE_COBBS = ["Cobb-regional", "Cobb-segmental", "Spline", "reconstructed_regional", "reconstructed_segmental"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _main_region(row) -> str:
    if aa.is_lateral(row.get("img_type")):
        for name, (upper, lower) in _LATERAL_BORDERS.items():
            if row["vertebra_upper"] in upper and row["vertebra_lower"] in lower:
                return name
    # ponytail: lateral angles off the spans (reconstructed inflection segments) keep their a.p. region
    # a.p.: the apex region build_angles_df computed from the nodes; rows without it
    # (an outlier_angles_df.pkl older than prepare_outlier_data.py's apex rule) fall back to the midpoint
    if isinstance(row.get("region"), str):
        return row["region"]
    try:
        return convertMeasurementBorderToSpineRegion(
            row["vertebra_upper"].strip(), row["vertebra_lower"].strip()
        ) or "?"
    except AttributeError:
        return "?"


def apply_filters(df, cobbs, img, raters, regions) -> pd.DataFrame:
    return df[
        df["cobbType"].isin(cobbs) &
        df["img_type"].isin(img) &
        df["rater_name"].isin(raters) &
        df["main_region"].isin(regions)
    ]


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_data() -> pd.DataFrame:
    """Sigma=0 fast path: pre-built pkl."""
    df = pd.read_pickle(f"{DATA_DIR}/outlier_angles_df.pkl")
    df["main_region"] = df.apply(_main_region, axis=1)
    return df


@st.cache_data(show_spinner=False)
def _build_smoothed(sigma: float) -> pd.DataFrame:
    """Recompute angles with Gaussian node smoothing. Cached per sigma value."""
    df4d = pd.read_pickle(f"{DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    df = build_angles_df(df4d, include_reconstructed=True, smooth_sigma=sigma)
    df["main_region"] = df.apply(_main_region, axis=1)
    return df


def get_angles(sigma: float) -> pd.DataFrame:
    return load_data() if sigma == 0.0 else _build_smoothed(sigma)


# ── Boxplot ───────────────────────────────────────────────────────────────────

def make_boxplot(df: pd.DataFrame, x_var: str, hue_var: str | None) -> plt.Figure:
    colors = plt.cm.tab10.colors
    if hue_var:
        x_vals = sorted(df[x_var].dropna().unique())
        h_vals = sorted(df[hue_var].dropna().unique())
        n_x, n_h = len(x_vals), len(h_vals)
        width = 0.8 / n_h
        fig, ax = plt.subplots(figsize=(max(8, n_x * n_h * 0.9 + 2), 5))
        for j, hv in enumerate(h_vals):
            positions = [i + (j - n_h / 2 + 0.5) * width for i in range(n_x)]
            data = [
                df[(df[x_var] == xv) & (df[hue_var] == hv)]["angle"].dropna().values
                for xv in x_vals
            ]
            bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                            patch_artist=True, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[j % len(colors)])
                patch.set_alpha(0.7)
            ax.plot([], [], color=colors[j % len(colors)], label=str(hv), linewidth=4)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_vals, rotation=40, ha="right")
        ax.legend(title=DIMS[hue_var], loc="upper right", fontsize=8)
    else:
        groups = df.groupby(x_var)["angle"].apply(list)
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.3 + 2), 5))
        bp = ax.boxplot(groups.values, labels=groups.index, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        ax.tick_params(axis="x", rotation=40)
    ax.set_xlabel(DIMS[x_var])
    ax.set_ylabel("Cobb-Winkel (°)")
    ax.set_title(f"Winkelverteilung nach {DIMS[x_var]}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ── ICC ───────────────────────────────────────────────────────────────────────

def compute_icc_table(df: pd.DataFrame, breakdown_col: str | None) -> pd.DataFrame:
    """ICC(2,1) + medianer Inter-Rater-Std pro (cobbType, Region), optional nach breakdown_col.

    Aggregiert zuerst auf max pro (image, rater, cobbType, Region), damit Rater mit
    mehreren Winkeln in derselben Region exakt eine Beobachtung pro Zelle liefern.
    """
    agg = (
        df.dropna(subset=["angle", "main_region"])
        .groupby(["image_id", "img_type", "rater_name", "cobbType", "main_region"],
                 as_index=False)["angle"].max()
    )
    # Synthetic LU = max(U-LU, L-LU) per rater/image/cobbType
    lu = (
        agg[agg["main_region"].isin({"U-LU", "L-LU"})]
        .groupby(["image_id", "img_type", "rater_name", "cobbType"], as_index=False)["angle"].max()
    )
    lu["main_region"] = "LU"
    agg = pd.concat([agg, lu], ignore_index=True)
    group_keys = ["cobbType", "main_region"] + ([breakdown_col] if breakdown_col else [])
    rows = []
    for keys, sub in agg.groupby(group_keys):
        keys = (keys,) if isinstance(keys, str) else tuple(keys)
        res = compute_icc(sub, "image_id", "rater_name", "angle")
        res["median_std"] = round(sub.groupby("image_id")["angle"].std().median(), 2)
        row = dict(zip(group_keys, keys))
        row.update(res)
        rows.append(row)
    result = pd.DataFrame(rows)
    _rank = {r: i for i, r in enumerate(REGION_ORDER)}
    result["_r"] = result["main_region"].map(_rank).fillna(len(REGION_ORDER))
    return result.sort_values(["cobbType", "_r"]).drop(columns=["_r"]).reset_index(drop=True)


SEGMENT_ORDER = ["T12/L1", "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def _icc_color(val: float) -> str:
    if np.isnan(val):
        return "color: gray"
    if val >= 0.75:
        return "color: #2ecc71; font-weight:600"
    if val >= 0.50:
        return "color: #f39c12; font-weight:600"
    return "color: #e74c3c; font-weight:600"


def make_icc_chart(
    icc_df: pd.DataFrame,
    x_col: str,
    x_order: list[str],
) -> plt.Figure:
    """Gruppenbalken: ICC (links) + Median-Std-Marker (rechts) pro Region/Segment.

    Ein Subplot pro img_type, Farbe = cobbType.
    Fehlende Werte erzeugen keinen Balken (statt versteckter Punkte).
    """
    img_types = sorted(icc_df["img_type"].dropna().unique()) if "img_type" in icc_df.columns else ["–"]
    cobb_types = sorted(icc_df["cobbType"].dropna().unique())
    n_ct = len(cobb_types)
    bar_w = 0.8 / max(n_ct, 1)
    colors = plt.cm.tab10.colors
    cmap = {ct: colors[i % len(colors)] for i, ct in enumerate(cobb_types)}

    n = len(img_types)
    fig, axes = plt.subplots(1, n, figsize=(max(5, n * 5.5), 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, img_type in zip(axes, img_types):
        sub = icc_df[icc_df["img_type"] == img_type] if "img_type" in icc_df.columns else icc_df
        xs_present = [x for x in x_order if x in sub[x_col].values]
        if not xs_present:
            ax.set_title(img_type, fontweight="bold")
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center", transform=ax.transAxes)
            continue

        ax2 = ax.twinx()
        legend_added: set[str] = set()

        for j, ct in enumerate(cobb_types):
            ct_sub = sub[sub["cobbType"] == ct].set_index(x_col).reindex(xs_present)
            color = cmap[ct]
            offset = (j - n_ct / 2 + 0.5) * bar_w

            for i, x in enumerate(xs_present):
                icc_val = ct_sub.at[x, "ICC"] if x in ct_sub.index else np.nan
                std_val = (ct_sub.at[x, "median_std"]
                           if "median_std" in ct_sub.columns and x in ct_sub.index
                           else np.nan)
                pos = i + offset
                if not (isinstance(icc_val, float) and np.isnan(icc_val)):
                    lbl = ct if ct not in legend_added else ""
                    ax.bar(pos, icc_val, width=bar_w * 0.9, color=color,
                           alpha=0.82, label=lbl, zorder=3)
                    legend_added.add(ct)
                if not (isinstance(std_val, float) and np.isnan(std_val)):
                    ax2.plot(pos, std_val, "D", color=color,
                             markersize=8, alpha=0.85, zorder=4,
                             markeredgecolor="black", markeredgewidth=0.8)

        ax.set_xticks(range(len(xs_present)))
        ax.set_xticklabels(xs_present, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.75, color="#aaa", linestyle=":", linewidth=1)
        ax.axhline(0.50, color="#aaa", linestyle=":", linewidth=1)
        ax.set_ylabel("ICC(2,1)", fontsize=9)
        ax2.set_ylabel("Median Std (°)", fontsize=9, color="#888")
        ax2.tick_params(axis="y", labelcolor="#888")
        ax.set_title(img_type, fontweight="bold")
        ax.grid(axis="y", alpha=0.15, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="cobbType", loc="upper center",
                   bbox_to_anchor=(0.5, 1.12), ncol=min(len(labels), 4),
                   fontsize=8, framealpha=0.9)
    fig.text(0.5, -0.03, "█  ICC(2,1) (links)   ◆  Median Std° (rechts)",
             ha="center", fontsize=8, color="#666")
    plt.tight_layout()
    return fig


# ── Korrelation ───────────────────────────────────────────────────────────────

def make_corr_heatmap(corr: pd.DataFrame) -> plt.Figure:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4, n * 1.0)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < 0.8 else "white")
    ax.set_title("Pearson-Korrelation zwischen Cobb-Typen")
    plt.tight_layout()
    return fig


# ── Sigma-Vergleich ───────────────────────────────────────────────────────────

def make_sigma_table(dfs: dict, cobbs: list, groupby: list | None = None) -> pd.DataFrame:
    """Vergleichstabelle: Zeilen = cobbType [+ groupby], Spalten = σ-Werte (Ø / Std / Median)."""
    idx_cols = groupby if groupby else []
    parts = {}
    for sigma, df in sorted(dfs.items()):
        sub = df[df["cobbType"].isin(cobbs)]
        grp = sub.groupby(["cobbType"] + idx_cols)["angle"]
        agg = grp.agg(mean="mean", std="std", median="median").round(2)
        agg.columns = [f"σ={sigma} Ø", f"σ={sigma} Std", f"σ={sigma} Md"]
        parts[sigma] = agg
    return pd.concat(parts.values(), axis=1).reset_index()


def make_sigma_boxplot(dfs: dict, cobbs: list) -> plt.Figure:
    """Ein Subplot pro cobbType, Boxen pro σ. Spline dient als Baseline."""
    sigmas = sorted(dfs.keys())
    n = len(cobbs)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sigmas)))

    fig, axes = plt.subplots(1, n, figsize=(max(5, n * 3.5), 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cobb in zip(axes, cobbs):
        data = [dfs[s][dfs[s]["cobbType"] == cobb]["angle"].dropna().values for s in sigmas]
        labels = [f"σ={s}" for s in sigmas]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)
        for patch, color in zip(bp["boxes"], cmap):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)
        ax.set_title(cobb, fontsize=9, fontweight="bold")
        ax.set_ylabel("Cobb-Winkel (°)" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        if cobb in ("Cobb-regional", "Cobb-segmental", "Spline"):
            ax.set_xlabel("(API-gespeichert, σ-invariant)", fontsize=7, color="gray")

    fig.suptitle("Einfluss der Gauss-Glättung auf Cobb-Winkel", fontsize=11)
    plt.tight_layout()
    return fig


def make_sigma_icc(dfs: dict, cobbs: list) -> pd.DataFrame:
    """ICC pro (σ, cobbType, Region/Segment) — konsistent mit dem ICC-Tab.

    Regional: compute_icc_table (max pro Region, ICC pro Region).
    Segmental: compute_icc_by_segment (ICC pro Lumbar-Segment-Paar).
    Kein Pooling über Regionen/Segmente — vermeidet den Vergleich verschiedener Bereiche.
    """
    rows = []
    reg_cobbs = [ct for ct in cobbs if ct in REGIONAL_COBB_TYPES]
    seg_cobbs = [ct for ct in cobbs if ct in SEGMENTAL_COBB_TYPES]

    for sigma, df in sorted(dfs.items()):
        if reg_cobbs:
            icc_reg = compute_icc_table(df[df["cobbType"].isin(reg_cobbs)],
                                        breakdown_col="img_type")
            for _, r in icc_reg.dropna(subset=["ICC"]).iterrows():
                rows.append({
                    "σ": sigma,
                    "img_type": r.get("img_type", ""),
                    "cobbType": r["cobbType"],
                    "Bereich": r["main_region"],
                    "ICC": round(r["ICC"], 3),
                    "CI95 [untere, obere]": (
                        f"[{r['CI95_lower']:.3f}, {r['CI95_upper']:.3f}]"
                        if not np.isnan(r["CI95_lower"]) else "—"
                    ),
                    "Median Std (°)": r["median_std"],
                    "n Bilder": r["n"],
                })

        if seg_cobbs:
            seg_df = df[
                df["cobbType"].isin(seg_cobbs) &
                df.apply(
                    lambda r: (r["vertebra_upper"], r["vertebra_lower"]) in LUMBAR_SEGMENT_PAIRS,
                    axis=1,
                )
            ]
            active_img_types = seg_df["img_type"].dropna().unique().tolist()
            if active_img_types:
                icc_seg = compute_icc_by_segment(seg_df, seg_cobbs, active_img_types)
                for _, r in icc_seg.dropna(subset=["ICC"]).iterrows():
                    rows.append({
                        "σ": sigma,
                        "img_type": r.get("img_type", ""),
                        "cobbType": r["cobbType"],
                        "Bereich": f"{r['vertebra_upper']}/{r['vertebra_lower']}",
                        "ICC": round(r["ICC"], 3),
                        "CI95 [untere, obere]": (
                            f"[{r['CI95_lower']:.3f}, {r['CI95_upper']:.3f}]"
                            if not np.isnan(r["CI95_lower"]) else "—"
                        ),
                        "Median Std (°)": r["median_std"],
                        "n Bilder": r["n"],
                    })

    return pd.DataFrame(rows)


def make_sigma_icc_chart(
    icc_df: pd.DataFrame,
    x_order: list[str],
    x_label: str,
) -> plt.Figure:
    """3D-Balkendiagramm: X = Region/Segment, Y = σ, Z = ICC, Farbe = cobbType.

    Ein Subplot pro img_type. Erweiterung des ICC-Panel-Charts um die σ-Dimension.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    img_types = sorted(icc_df["img_type"].dropna().unique()) if "img_type" in icc_df.columns else ["–"]
    cobb_types = sorted(icc_df["cobbType"].dropna().unique())
    sigma_vals = sorted(icc_df["σ"].unique())
    n_ct = len(cobb_types)
    dx = 0.8 / max(n_ct, 1)   # bar width along x (region axis)
    dy = 0.55                   # bar depth along y (sigma axis)

    colors = plt.cm.tab10.colors
    cmap = {ct: colors[i % len(colors)] for i, ct in enumerate(cobb_types)}

    n = len(img_types)
    fig = plt.figure(figsize=(max(7, n * 7), 5.5))

    for idx, img_type in enumerate(img_types):
        ax = fig.add_subplot(1, n, idx + 1, projection="3d")
        sub = icc_df[icc_df["img_type"] == img_type] if "img_type" in icc_df.columns else icc_df
        xs_present = [x for x in x_order if x in sub["Bereich"].values]
        if not xs_present:
            ax.set_title(img_type, fontweight="bold")
            continue

        legend_added: set[str] = set()
        for j, ct in enumerate(cobb_types):
            ct_sub = sub[sub["cobbType"] == ct]
            color = cmap[ct]
            offset = (j - n_ct / 2 + 0.5) * dx
            for xi, x in enumerate(xs_present):
                for yi, sigma in enumerate(sigma_vals):
                    row = ct_sub[(ct_sub["Bereich"] == x) & (ct_sub["σ"] == sigma)]
                    if row.empty:
                        continue
                    icc_val = row.iloc[0]["ICC"]
                    if np.isnan(icc_val):
                        continue
                    lbl = ct if ct not in legend_added else ""
                    ax.bar3d(
                        xi + offset - dx * 0.45,  # x base
                        yi - dy / 2,              # y base
                        0,                        # z base
                        dx * 0.9, dy, icc_val,
                        color=color, alpha=0.80, label=lbl, zsort="average",
                    )
                    legend_added.add(ct)

        # Reference planes at ICC = 0.75 and 0.50
        xg, yg = np.meshgrid([-0.6, len(xs_present) - 0.4], [-0.6, len(sigma_vals) - 0.4])
        for ref, col in [(0.75, "#2ecc71"), (0.50, "#f39c12")]:
            ax.plot_surface(xg, yg, np.full_like(xg, ref),
                            alpha=0.10, color=col, zorder=0)

        ax.set_xticks(range(len(xs_present)))
        ax.set_xticklabels(xs_present, rotation=18, ha="right", fontsize=7)
        ax.set_yticks(range(len(sigma_vals)))
        ax.set_yticklabels([f"σ={s}" for s in sigma_vals], fontsize=7)
        ax.set_zlim(0, 1.05)
        ax.set_zlabel("ICC(2,1)", fontsize=8, labelpad=6)
        ax.set_xlabel(x_label, fontsize=8, labelpad=8)
        ax.set_ylabel("Gauss-σ", fontsize=8, labelpad=8)
        ax.set_title(img_type, fontweight="bold")
        ax.view_init(elev=22, azim=-50)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap[ct], alpha=0.80) for ct in cobb_types]
    fig.legend(handles, cobb_types, title="cobbType", loc="upper center",
               bbox_to_anchor=(0.5, 1.06), ncol=min(len(cobb_types), 4),
               fontsize=8, framealpha=0.9)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Winkel-Statistik", layout="wide", page_icon="📐")
    st.title("📐 Cobb-Winkel Statistik")

    try:
        _ = load_data()
    except FileNotFoundError as e:
        st.error(f"Pickle-Datei fehlt: `{e}`\n\nBitte zuerst `prepare_outlier_data.py` ausführen.")
        return

    df0 = load_data()  # sigma=0, used for filter option lists

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Unabhängige Variablen")
        groupby_vars = st.multiselect(
            "Gruppierung nach",
            options=list(DIMS.keys()),
            default=["cobbType"],
            format_func=lambda x: DIMS[x],
            help="Für Tab 'Statistik'. ICC und Korrelation haben eigene Logik.",
        )

        st.divider()
        st.subheader("Gauss-Glättung")
        sigma = st.select_slider(
            "Sigma (0 = keine Glättung)",
            options=SIGMA_OPTIONS,
            value=0.0,
            help="Betrifft reconstructed_* und inflection_*. Spline/Cobb-regional/segmental bleiben unverändert.",
        )
        if sigma > 0:
            st.caption("⚠ Erste Berechnung pro σ dauert ~15 s.")

        st.divider()
        st.subheader("Filter")
        sel_cobbs = st.multiselect(
            "cobbType",
            sorted(df0["cobbType"].dropna().unique()),
            default=sorted(df0["cobbType"].dropna().unique()),
        )
        sel_img = st.multiselect(
            "Bildtyp",
            sorted(df0["img_type"].dropna().unique()),
            default=sorted(df0["img_type"].dropna().unique()),
        )
        sel_raters = st.multiselect(
            "User/Rater",
            sorted(df0["rater_name"].dropna().unique()),
            default=sorted(df0["rater_name"].dropna().unique()),
        )
        sel_regions = st.multiselect(
            "Hauptregion",
            sorted(df0["main_region"].dropna().unique()),
            default=sorted(df0["main_region"].dropna().unique()),
        )

        st.divider()
        st.caption(
            f"**{df0['image_id'].nunique()}** Bilder · "
            f"**{len(df0)}** Messungen gesamt"
        )

    # Aktiver Datensatz (sigma-abhängig)
    with st.spinner(f"Daten für σ={sigma} laden …") if sigma > 0 else st.empty():
        active_df = get_angles(sigma)

    filtered = apply_filters(active_df, sel_cobbs, sel_img, sel_raters, sel_regions)

    if filtered.empty:
        st.warning("Keine Daten nach aktuellem Filter.")
        return

    st.caption(f"**{len(filtered)}** Winkelwerte · σ = {sigma}")

    tab_stats, tab_icc, tab_corr, tab_sigma = st.tabs(
        ["📊 Statistik", "📏 ICC (Inter-Rater)", "🔗 Korrelation", "📈 Sigma-Vergleich"]
    )

    # ── Tab 1: Statistik ──────────────────────────────────────────────────────
    with tab_stats:
        if not groupby_vars:
            st.info("Bitte mindestens eine Gruppierungsvariable in der Sidebar wählen.")
        else:
            stats = (
                filtered.groupby(groupby_vars)["angle"]
                .agg(n="count", Mittelwert="mean", Std="std", Median="median",
                     Min="min", Max="max")
                .round(2)
                .reset_index()
            )
            st.subheader("Deskriptive Statistik")
            st.dataframe(stats, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ CSV exportieren",
                data=stats.to_csv(index=False).encode(),
                file_name="cobb_statistik.csv",
                mime="text/csv",
            )

            st.divider()
            plot_vars = groupby_vars[:2]
            x_var, hue_var = plot_vars[0], plot_vars[1] if len(plot_vars) == 2 else None
            n_combos = filtered[x_var].nunique() * (filtered[hue_var].nunique() if hue_var else 1)
            if n_combos > 80:
                st.warning(f"Zu viele Gruppen ({n_combos}) für Boxplot.")
            else:
                st.subheader("Boxplot")
                fig = make_boxplot(filtered, x_var, hue_var)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                st.pyplot(fig, use_container_width=True)
                st.download_button(
                    "⬇ Boxplot als PNG",
                    data=buf.getvalue(),
                    file_name="cobb_boxplot.png",
                    mime="image/png",
                )
                plt.close(fig)

    # ── Tab 2: ICC ────────────────────────────────────────────────────────────
    with tab_icc:
        st.caption(
            "ICC(2,1) · Zwei-Wege-ANOVA für unbalancierte Designs (Shrout & Fleiss 1979). "
            "Grün ≥ 0.75 · Gelb ≥ 0.50 · Rot < 0.50"
        )
        breakdown_label = st.selectbox(
            "Aufschlüsseln nach",
            options=list(ICC_BREAKDOWN_OPTIONS.keys()),
            index=0,
            key="icc_breakdown",
        )
        breakdown_col = ICC_BREAKDOWN_OPTIONS[breakdown_label]

        def _icc_display(df: pd.DataFrame) -> pd.DataFrame:
            d = df.dropna(subset=["ICC"]).rename(columns={
                "main_region": "Region",
                "vertebra_upper": "Oben", "vertebra_lower": "Unten",
                "CI95_lower": "CI95 untere", "CI95_upper": "CI95 obere",
                "n": "n Bilder", "median_std": "Median Std (°)",
            })
            for col in ["ICC", "CI95 untere", "CI95 obere"]:
                if col in d.columns:
                    d[col] = d[col].round(3)
            return d

        # ── Regional ─────────────────────────────────────────────────────────
        reg_filtered = filtered[filtered["cobbType"].isin(REGIONAL_COBB_TYPES)]
        if not reg_filtered.empty:
            st.subheader("Regional")
            st.caption(
                "Verglichene Typen: **Cobb-regional**, **Spline**, **reconstructed_regional**. "
                "Aggregation: max-Winkel pro (Rater · Bild · Spine-Region) vor ICC."
            )
            with st.spinner("Regional ICC berechnen …"):
                icc_reg = compute_icc_table(reg_filtered, breakdown_col)
                # Chart always per img_type regardless of table breakdown
                icc_reg_chart = (icc_reg if breakdown_col == "img_type"
                                 else compute_icc_table(reg_filtered, "img_type"))

            lu_regions = [r for r in ["U-LU", "L-LU", "LU"]
                          if r in icc_reg_chart["main_region"].values]
            sel_lu = st.multiselect(
                "Lumbar-Regionen einblenden",
                options=lu_regions,
                default=lu_regions,
                key="icc_lu_regions",
            )
            non_lu = [r for r in REGION_ORDER if r not in {"U-LU", "L-LU", "LU"}]
            chart_regions = non_lu + sel_lu  # preserves spine order
            chart_df = icc_reg_chart[icc_reg_chart["main_region"].isin(chart_regions)]
            fig = make_icc_chart(chart_df, "main_region", chart_regions)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇ Regional ICC Chart als PNG", data=buf.getvalue(),
                               file_name="icc_regional_chart.png", mime="image/png",
                               key="dl_icc_reg_chart")
            plt.close(fig)

            st.divider()
            d = _icc_display(icc_reg)
            st.dataframe(d.style.map(_icc_color, subset=["ICC"]),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇ Regional ICC als CSV",
                               data=d.to_csv(index=False).encode(),
                               file_name="icc_regional.csv", mime="text/csv",
                               key="dl_icc_reg")

        # ── Segmental ─────────────────────────────────────────────────────────
        seg_filtered = filtered[
            filtered["cobbType"].isin(SEGMENTAL_COBB_TYPES) &
            filtered.apply(
                lambda r: (r["vertebra_upper"], r["vertebra_lower"]) in LUMBAR_SEGMENT_PAIRS,
                axis=1,
            )
        ]
        if not seg_filtered.empty:
            st.subheader("Segmental")
            st.caption(
                "Verglichene Typen: **Cobb-segmental**, **reconstructed_segmental**. "
                "Gruppierung nach exaktem Vertebra-Paar (z.B. L1/L2) — keine Region-Aggregation."
            )
            active_img_types = seg_filtered["img_type"].dropna().unique().tolist()
            seg_cobb_types = [ct for ct in SEGMENTAL_COBB_TYPES
                              if ct in seg_filtered["cobbType"].unique()]
            with st.spinner("Segmental ICC berechnen …"):
                icc_seg = compute_icc_by_segment(seg_filtered, seg_cobb_types, active_img_types)
            if breakdown_col and breakdown_col in icc_seg.columns:
                icc_seg = icc_seg[icc_seg[breakdown_col].isin(filtered[breakdown_col].unique())]

            icc_seg_chart = icc_seg.copy()
            icc_seg_chart["segment"] = (icc_seg_chart["vertebra_upper"] + "/"
                                        + icc_seg_chart["vertebra_lower"])
            fig = make_icc_chart(icc_seg_chart, "segment", SEGMENT_ORDER)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇ Segmental ICC Chart als PNG", data=buf.getvalue(),
                               file_name="icc_segmental_chart.png", mime="image/png",
                               key="dl_icc_seg_chart")
            plt.close(fig)

            st.divider()
            d = _icc_display(icc_seg)
            st.dataframe(d.style.map(_icc_color, subset=["ICC"]),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇ Segmental ICC als CSV",
                               data=d.to_csv(index=False).encode(),
                               file_name="icc_segmental.csv", mime="text/csv",
                               key="dl_icc_seg")

        if reg_filtered.empty and seg_filtered.empty:
            st.info("Keine regionalen oder segmentalen cobbTypes im aktuellen Filter.")

    # ── Tab 3: Korrelation ────────────────────────────────────────────────────
    with tab_corr:
        st.subheader("Korrelation zwischen Cobb-Typen")
        st.caption(
            "Pearson r — Pivot: (Bild × Rater × Region) als Zeilen, cobbType als Spalten. "
            "Nur Paare mit ≥ 10 gemeinsamen Beobachtungen."
        )
        pivot = filtered.pivot_table(
            index=["image_id", "rater_name", "vertebra_upper", "vertebra_lower"],
            columns="cobbType",
            values="angle",
            aggfunc="first",
        )
        corr = pivot.corr(min_periods=10)

        if corr.empty or corr.isna().all().all():
            st.warning("Nicht genug überlappende Beobachtungen für Korrelationsmatrix.")
        else:
            fig = make_corr_heatmap(corr)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            st.pyplot(fig, use_container_width=True)
            st.download_button(
                "⬇ Heatmap als PNG",
                data=buf.getvalue(),
                file_name="cobb_korrelation.png",
                mime="image/png",
                key="dl_corr",
            )
            plt.close(fig)
            st.divider()
            st.caption("Numerische Korrelationsmatrix")
            st.dataframe(corr.round(3), use_container_width=True)

    # ── Tab 4: Sigma-Vergleich ────────────────────────────────────────────────
    with tab_sigma:
        st.subheader("Einfluss der Gauss-Glättung")
        st.caption(
            "**Cobb-regional / Cobb-segmental** = Ground-Truth (API, σ-invariant).  "
            "**Spline** = vom User gemessen, wird ab σ > 0 neu berechnet.  "
            "**reconstructed_*** = vollautomatisch aus geglättetem Spline."
        )

        col_a, col_b = st.columns([2, 3])
        with col_a:
            compare_sigmas = st.multiselect(
                "σ-Werte vergleichen",
                options=SIGMA_OPTIONS,
                default=[0.0, 1.0, 2.0],
                key="compare_sigmas",
            )
        with col_b:
            compare_cobbs = st.multiselect(
                "cobbTypes",
                options=COMPARE_COBBS,
                default=COMPARE_COBBS,
                key="compare_cobbs",
            )

        if not compare_sigmas or not compare_cobbs:
            st.info("Bitte mindestens einen σ-Wert und einen cobbType wählen.")
        else:
            # Lade / berechne alle benötigten σ-Datensätze
            with st.spinner("Sigma-Datensätze laden …"):
                sigma_dfs = {}
                for s in compare_sigmas:
                    df_s = _build_smoothed(s)  # immer gleiche Pipeline → Spline bleibt invariant
                    # Filter: img_type, rater, region — aber cobbType aus compare_cobbs
                    sigma_dfs[s] = apply_filters(
                        df_s, compare_cobbs, sel_img, sel_raters, sel_regions
                    )

            # Vergleichstabelle
            st.subheader("Deskriptive Statistik nach σ")
            cmp_table = make_sigma_table(sigma_dfs, compare_cobbs)
            st.dataframe(cmp_table, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Vergleich als CSV",
                data=cmp_table.to_csv(index=False).encode(),
                file_name="cobb_sigma_vergleich.csv",
                mime="text/csv",
                key="dl_sigma_tbl",
            )

            st.divider()

            # Boxplot
            st.subheader("Boxplot nach σ")
            n_combos = len(compare_cobbs) * len(compare_sigmas)
            if n_combos > 30:
                st.warning("Zu viele Kombinationen für übersichtlichen Boxplot.")
            else:
                fig = make_sigma_boxplot(sigma_dfs, compare_cobbs)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
                st.pyplot(fig, use_container_width=True)
                st.download_button(
                    "⬇ Boxplot als PNG",
                    data=buf.getvalue(),
                    file_name="cobb_sigma_boxplot.png",
                    mime="image/png",
                    key="dl_sigma_box",
                )
                plt.close(fig)

            st.divider()

            # ICC-Vergleich über σ
            st.subheader("ICC nach σ")
            with st.spinner("ICC berechnen …"):
                icc_sigma = make_sigma_icc(sigma_dfs, compare_cobbs)

            reg_mask = icc_sigma["Bereich"].isin(REGION_ORDER)
            icc_reg = icc_sigma[reg_mask]
            icc_seg = icc_sigma[~reg_mask]

            if not icc_reg.empty:
                fig_reg = make_sigma_icc_chart(icc_reg, REGION_ORDER, "Region")
                st.pyplot(fig_reg, use_container_width=True)
                plt.close(fig_reg)

            if not icc_seg.empty:
                fig_seg = make_sigma_icc_chart(icc_seg, SEGMENT_ORDER, "Segment")
                st.pyplot(fig_seg, use_container_width=True)
                plt.close(fig_seg)

            st.dataframe(
                icc_sigma.style.map(_icc_color, subset=["ICC"]),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇ ICC-Vergleich als CSV",
                data=icc_sigma.to_csv(index=False).encode(),
                file_name="cobb_sigma_icc.csv",
                mime="text/csv",
                key="dl_sigma_icc",
            )


if __name__ == "__main__":
    main()
