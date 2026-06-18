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

from annotation_analysis import compute_icc, build_angles_df
from bspline_python import convertMeasurementBorderToSpineRegion

DATA_DIR = "data"

DIMS = {
    "cobbType":    "Cobb-Typ",
    "img_type":    "Bildtyp",
    "rater_name":  "User/Rater",
    "main_region": "Hauptregion (CT/PT/MT/TL/LU)",
}
ICC_BREAKDOWN_OPTIONS = {"(keine)": None, "Bildtyp": "img_type", "Hauptregion": "main_region"}
SIGMA_OPTIONS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
COMPARE_COBBS = ["Cobb-regional", "Cobb-segmental", "Spline", "reconstructed_regional", "reconstructed_segmental"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _main_region(row) -> str:
    try:
        return convertMeasurementBorderToSpineRegion(
            row["vertebra_upper"].strip(), row["vertebra_lower"].strip()
        )
    except (ValueError, AttributeError):
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
    """ICC(2,1) + medianer Inter-Rater-Std pro cobbType, optional nach breakdown_col."""
    rows = []
    group_keys = ["cobbType"] + ([breakdown_col] if breakdown_col else [])
    for keys, sub in df.groupby(group_keys):
        keys = (keys,) if isinstance(keys, str) else keys
        res = compute_icc(sub, "image_id", "rater_name", "angle")
        res["median_std"] = round(sub.groupby("image_id")["angle"].std().median(), 2)
        row = dict(zip(group_keys, keys))
        row.update(res)
        rows.append(row)
    return pd.DataFrame(rows)


def _icc_color(val: float) -> str:
    if np.isnan(val):
        return "color: gray"
    if val >= 0.75:
        return "color: #2ecc71; font-weight:600"
    if val >= 0.50:
        return "color: #f39c12; font-weight:600"
    return "color: #e74c3c; font-weight:600"


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
    """ICC + Median-Std pro cobbType × σ als Vergleichstabelle."""
    rows = []
    for sigma, df in sorted(dfs.items()):
        sub = df[df["cobbType"].isin(cobbs)]
        for ct, grp in sub.groupby("cobbType"):
            res = compute_icc(grp, "image_id", "rater_name", "angle")
            rows.append({
                "σ": sigma,
                "cobbType": ct,
                "ICC": round(res["ICC"], 3) if not np.isnan(res["ICC"]) else np.nan,
                "CI95 [untere, obere]": (
                    f"[{res['CI95_lower']:.3f}, {res['CI95_upper']:.3f}]"
                    if not np.isnan(res["CI95_lower"]) else "—"
                ),
                "Median Std (°)": round(grp.groupby("image_id")["angle"].std().median(), 2),
                "n Bilder": res["n"],
            })
    return pd.DataFrame(rows)


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
        st.subheader("ICC(2,1) zwischen Ratern — pro Cobb-Typ")
        st.caption(
            "Zwei-Wege-ANOVA ICC für unbalancierte Designs (Shrout & Fleiss 1979). "
            "Grün ≥ 0.75 · Gelb ≥ 0.50 · Rot < 0.50"
        )
        breakdown_label = st.selectbox(
            "Aufschlüsseln nach",
            options=list(ICC_BREAKDOWN_OPTIONS.keys()),
            index=0,
            key="icc_breakdown",
        )
        breakdown_col = ICC_BREAKDOWN_OPTIONS[breakdown_label]

        with st.spinner("ICC berechnen …"):
            icc_df = compute_icc_table(filtered, breakdown_col)

        icc_display = icc_df.rename(columns={
            "ICC": "ICC", "CI95_lower": "CI95 untere", "CI95_upper": "CI95 obere",
            "n": "n Bilder", "median_std": "Median Std (°)",
        })
        for col in ["ICC", "CI95 untere", "CI95 obere"]:
            icc_display[col] = icc_display[col].round(3)

        st.dataframe(
            icc_display.style.map(_icc_color, subset=["ICC"]),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇ ICC-Tabelle als CSV",
            data=icc_display.to_csv(index=False).encode(),
            file_name="cobb_icc.csv",
            mime="text/csv",
            key="dl_icc",
        )

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
