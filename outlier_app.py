"""
Annotation Precision Outlier Viewer — Streamlit App

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run outlier_app.py

Vorbedingung: prepare_outlier_data.py einmalig ausführen, um Pickle-Dateien zu erstellen.
"""

import base64
import io
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from bspline_python import (
    interpolate_bspline,
    smooth_nodes,
    spine_labels,
    NodeItem,
    get_perpendicular_lines_at_indices,
    map_nodes_to_spline_and_keys,
)
from annotation_analysis import get_inflection_overlay_data

# Konva-Stage hat feste Höhe 600px (KonvaImage.vue).
# Alle gespeicherten Node/CobbLine-Koordinaten sind in diesem Canvas-Raum.
KONVA_STAGE_HEIGHT = 600
DATA_DIR = "data"
COLORS = plt.cm.tab10.colors
EXTRA_COBBS_LINESTYLES = ["--", "-.", ":"]
PERP_LENGTH = 200  # Länge der Perpendikularen in Konva-Canvas-Pixeln


def _draw_perp_lines(ax, xmd: dict, cobb_type: str, scale: float,
                     color, ls: str, lw: float, alpha: float, zorder: int,
                     smooth_sigma: float = 0.0) -> None:
    """Draw perpendicular lines for *cobb_type* onto *ax*.

    Handles Cobb-regional/segmental (from stateObject.cobbLines),
    Spline (from angles[].upperLine/lowerLine), and inflection_r* types
    (computed via get_inflection_overlay_data).
    """
    if cobb_type in ("Cobb-regional", "Cobb-segmental"):
        for cl in xmd.get("stateObject", {}).get("cobbLines", []):
            if cl.get("mode") == cobb_type:
                ax.plot([cl["x1"] * scale, cl["x2"] * scale],
                        [cl["y1"] * scale, cl["y2"] * scale],
                        ls, color=color, linewidth=lw, alpha=alpha, zorder=zorder)
    elif cobb_type == "Spline":
        # Perpendikularen aus dem Spline neu berechnen (einheitliche Länge wie inflection).
        # Die gespeicherten upperLine/lowerLine sind zu kurz und inkonsistent.
        nodes_raw = xmd.get("stateObject", {}).get("nodes", [])
        if len(nodes_raw) < 3:
            return
        nodes_canvas = [(n["x"], n["y"]) for n in nodes_raw]
        spline = interpolate_bspline(nodes_canvas, num_steps=200)
        if not spline:
            return
        vr = xmd.get("stateObject", {}).get("vertebra_range", [0, 0])
        selected_vertebra = spine_labels[vr[0]: vr[1] + 1]
        nodes_items = [NodeItem(x=x, y=y) for x, y in nodes_canvas]
        mapped = map_nodes_to_spline_and_keys(nodes_items, selected_vertebra, spline)
        lbl_to_idx = {m.label: m.spline_idx for m in mapped}
        for a in xmd.get("angles", []):
            if a.get("cobbType") != "Spline":
                continue
            ca = a.get("cobbAngle", {})
            vu, vl = ca.get("vertebra_upper"), ca.get("vertebra_lower")
            if vu not in lbl_to_idx or vl not in lbl_to_idx:
                continue
            perps = get_perpendicular_lines_at_indices(
                spline, length=PERP_LENGTH, indices=[lbl_to_idx[vu], lbl_to_idx[vl]]
            )
            for pl in perps:
                ax.plot([pl["x1"] * scale, pl["x2"] * scale],
                        [pl["y1"] * scale, pl["y2"] * scale],
                        ls, color=color, linewidth=lw, alpha=alpha, zorder=zorder)
    elif cobb_type.startswith("inflection_r"):
        threshold = float(cobb_type.split("@")[1]) if "@" in cobb_type else 10.0
        overlay = get_inflection_overlay_data(xmd, min_angle=threshold,
                                              smooth_sigma=smooth_sigma)
        for pl in overlay.get("perp_lines", []):
            ax.plot([pl["x1"] * scale, pl["x2"] * scale],
                    [pl["y1"] * scale, pl["y2"] * scale],
                    ls, color=color, linewidth=lw, alpha=alpha, zorder=zorder)


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
    # Patient-ID pro Bild (eine Messung pro Bild genügt)
    patient_map = df4d.groupby("id")["patient_id"].first()
    img_rank["patient_id"] = img_rank["image_id"].map(patient_map).fillna("?")
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

def build_figure(image_id: int, cobb: str, df4d, img_urls,
                 smooth_sigma: float = 0.0, extra_cobbs: list = [],
                 show_primary: bool = True, visible_raters=None):
    url = img_urls.get(image_id)
    xray = None
    if url:
        raw = fetch_image_bytes(url)
        if raw:
            xray = Image.open(io.BytesIO(raw)).convert("RGB")

    scale = (xray.height / KONVA_STAGE_HEIGHT) if xray else 1.0
    measurements = df4d[df4d["id"] == image_id]

    fig, ax_img = plt.subplots(1, 1, figsize=(10, 9))

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

        if visible_raters is not None and rater not in visible_raters:
            continue

        nodes_xy = [(n["x"] * scale, n["y"] * scale) for n in nodes_raw]
        spline = interpolate_bspline(nodes_xy, num_steps=20)

        if spline:
            ax_img.plot(
                [p[0] for p in spline], [p[1] for p in spline],
                "-", color=color, linewidth=2, label=rater, alpha=0.85, zorder=3,
            )

        # Geglättete Spline + Inflection-Perpendikularen
        if smooth_sigma > 0:
            # min_angle=0: alle Extrema-Grenzen zeichnen, unabhängig vom Winkelwert
            overlay = get_inflection_overlay_data(xmd, min_angle=0.0,
                                                  smooth_sigma=smooth_sigma)
            # Gestrichelte geglättete Spline
            nodes_smooth = smooth_nodes(nodes_xy, sigma=smooth_sigma)
            spline_smooth = interpolate_bspline(nodes_smooth, num_steps=20)
            if spline_smooth:
                ax_img.plot(
                    [p[0] for p in spline_smooth], [p[1] for p in spline_smooth],
                    "--", color="#FFEE00", linewidth=3.5, alpha=0.92, zorder=6,
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

        # Perpendikularen für primären cobbType (optional ausblendbar)
        if show_primary:
            _draw_perp_lines(ax_img, xmd, cobb, scale, color, "-", 2.5, 0.9, 5, smooth_sigma)

        # Perpendikularen für zusätzliche cobbTypes (Raterfarbe, anderer Linienstil)
        for j, extra_cobb in enumerate(extra_cobbs):
            xls = EXTRA_COBBS_LINESTYLES[j % len(EXTRA_COBBS_LINESTYLES)]
            _draw_perp_lines(ax_img, xmd, extra_cobb, scale, color, xls, 2.0, 0.85,
                             4 + j, smooth_sigma)

    # Dummy-Linien für extra-cobbType-Legende (grau, Linienstil zeigt Methode)
    for j, extra_cobb in enumerate(extra_cobbs):
        xls = EXTRA_COBBS_LINESTYLES[j % len(EXTRA_COBBS_LINESTYLES)]
        ax_img.plot([], [], xls, color="lightgray", linewidth=2, label=f"⟂ {extra_cobb}")

    img_type = measurements["img_type"].iloc[0] if len(measurements) else ""
    ax_img.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax_img.set_title(f"ID {image_id}  ({img_type})", fontsize=12)
    ax_img.axis("off")

    plt.tight_layout()
    return fig


def build_angle_table(image_id: int, cobbs_to_show: list, angles_df, df4d,
                      visible_raters=None):
    """Styled DataFrame: blocks nach Wirbelpaar (kranial→kaudal), Spalten = cobbTypes.

    Jede Rater-Zeile erhält die gleiche Hintergrundfarbe wie das Overlay im Bild.
    """
    if not cobbs_to_show:
        return None

    # Rater-Reihenfolge + Farben (identisch zur Overlay-Zuweisung in build_figure)
    measurements = df4d[df4d["id"] == image_id]
    rater_order = [
        row.get("rater_name", f"Rater {i+1}")
        for i, (_, row) in enumerate(measurements.iterrows())
    ]
    rater_color_hex = {}
    for i, rater in enumerate(rater_order):
        c = COLORS[i % len(COLORS)]
        rater_color_hex[rater] = (
            f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
        )

    # Winkel für alle cobbTypes sammeln
    _spine_idx = {lbl: i for i, lbl in enumerate(spine_labels)}
    parts = []
    for ct in cobbs_to_show:
        sub = angles_df[
            (angles_df["image_id"] == image_id) &
            (angles_df["cobbType"] == ct)
        ][["vertebra_upper", "vertebra_lower", "rater_name", "angle"]].copy()
        if not sub.empty:
            sub["cobbType"] = ct
            parts.append(sub)

    if not parts:
        return None

    combined = pd.concat(parts, ignore_index=True)
    if visible_raters is not None:
        combined = combined[combined["rater_name"].isin(visible_raters)]
    if combined.empty:
        return None

    combined["_u_idx"] = combined["vertebra_upper"].map(
        lambda v: _spine_idx.get(v, 99)
    )
    combined["_l_idx"] = combined["vertebra_lower"].map(
        lambda v: _spine_idx.get(v, 99)
    )
    _rater_rank = {r: i for i, r in enumerate(rater_order)}
    combined["_r_rank"] = combined["rater_name"].map(
        lambda r: _rater_rank.get(r, 99)
    )

    # Pivot: Zeilen = (Wirbelpaar, Rater), Spalten = cobbType
    pivot = combined.pivot_table(
        index=["_u_idx", "_l_idx", "vertebra_upper", "vertebra_lower",
               "_r_rank", "rater_name"],
        columns="cobbType",
        values="angle",
        aggfunc="first",
    ).round(1)

    pivot = pivot.sort_index(level=["_u_idx", "_l_idx", "_r_rank"])
    pivot = pivot.reset_index()

    # Fehlende cobbType-Spalten ergänzen
    for ct in cobbs_to_show:
        if ct not in pivot.columns:
            pivot[ct] = float("nan")

    # Anzeigespalten auswählen
    display_df = pivot[
        ["vertebra_upper", "vertebra_lower", "rater_name"] + cobbs_to_show
    ].copy()

    # Blockanzeige: Wirbelpaar nur in der ersten Zeile des Blocks zeigen
    prev_pair = None
    for i in range(len(display_df)):
        curr_pair = (
            display_df.iloc[i]["vertebra_upper"],
            display_df.iloc[i]["vertebra_lower"],
        )
        if curr_pair == prev_pair:
            display_df.iat[i, display_df.columns.get_loc("vertebra_upper")] = ""
            display_df.iat[i, display_df.columns.get_loc("vertebra_lower")] = ""
        else:
            prev_pair = curr_pair

    # Rater-Namen für Farbzuordnung merken (vor Umbenennung)
    rater_names_list = pivot["rater_name"].tolist()

    display_df = display_df.rename(columns={
        "vertebra_upper": "Oben",
        "vertebra_lower": "Unten",
        "rater_name": "Rater",
    })

    # Styling: Hintergrundfarbe nach Rater
    def _row_style(row):
        idx = row.name
        rater = rater_names_list[idx] if idx < len(rater_names_list) else ""
        hex_c = rater_color_hex.get(rater, "#888888")
        r_v = int(hex_c[1:3], 16)
        g_v = int(hex_c[3:5], 16)
        b_v = int(hex_c[5:7], 16)
        bg = f"rgba({r_v},{g_v},{b_v},0.18)"
        return [f"background-color: {bg}"] * len(row)

    fmt = {ct: lambda x: f"{x:.1f}°" if pd.notna(x) else "—" for ct in cobbs_to_show
           if ct in display_df.columns}

    styled = (
        display_df.style
        .apply(_row_style, axis=1)
        .format(fmt, na_rep="—")
        .set_properties(**{"text-align": "center"})
        .set_properties(subset=["Oben", "Unten", "Rater"],
                        **{"text-align": "left", "font-weight": "600"})
    )
    return styled


def compute_smooth_inflection_df(image_id: int, df4d, smooth_sigma: float,
                                  min_angle: float = SMOOTH_MIN_ANGLE,
                                  visible_raters=None) -> pd.DataFrame:
    """Berechnet Inflection-Winkel auf geglätteten Nodes on-the-fly.

    Gibt einen DataFrame in angles_df-Format zurück (image_id, cobbType,
    rater_name, vertebra_upper, vertebra_lower, angle).
    cobbType-Namen: 'inflection_r0', 'inflection_r1', …  (ohne @-Suffix).
    """
    measurements = df4d[df4d["id"] == image_id]
    rows = []
    for i, (_, row) in enumerate(measurements.iterrows()):
        xmd = row.get("xray_measurement_data")
        if not isinstance(xmd, dict):
            continue
        rater = row.get("rater_name", f"Rater {i+1}")
        if visible_raters is not None and rater not in visible_raters:
            continue
        overlay = get_inflection_overlay_data(xmd, min_angle=min_angle,
                                              smooth_sigma=smooth_sigma)
        for a in overlay["angles"]:
            ca = a["cobbAngle"]
            rows.append({
                "image_id":       image_id,
                "cobbType":       a["cobbType"],
                "rater_name":     rater,
                "vertebra_upper": ca["vertebra_upper"],
                "vertebra_lower": ca["vertebra_lower"],
                "angle":          float(ca["angle"]),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["image_id", "cobbType", "rater_name",
                 "vertebra_upper", "vertebra_lower", "angle"]
    )


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

        sort_ascending = st.checkbox(
            "Aufsteigend sortieren  (std=0 → Outlier)",
            value=False,
            help="Standard: höchste Abweichung zuerst. "
                 "Aktiviert: niedrigste Abweichung zuerst.",
        )

        show_primary = st.checkbox(
            f"Primär ({cobb}) einblenden", value=True,
            help="Perpendikularen des primären cobbTypes im Bild anzeigen.",
        )

        extra_cobb_options = [ct for ct in cobb_types if ct != cobb]
        extra_cobbs = st.multiselect(
            "Weitere cobbTypes einblenden",
            options=extra_cobb_options,
            default=[],
            help="Perpendikularen dieser Messmethoden werden farblich "
                 "hervorgehoben eingezeichnet.",
        )

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
        .sort_values("max_std", ascending=sort_ascending)
        .head(n_top)
        .reset_index(drop=True)
    )

    if ranked.empty:
        st.warning(f"Keine Bilder mit ≥2 Ratern für **{cobb}**.")
        return

    sort_label = "aufsteigend (std=0 → Outlier)" if sort_ascending else "absteigend (Outlier → std=0)"
    labels = [
        f"{r['img_type']:18s}  [{r['patient_id']}]  ID {r['image_id']}  σ_max={r['max_std']:.1f}°"
        for _, r in ranked.iterrows()
    ]
    selected = st.selectbox(f"Bild  ·  sortiert {sort_label}", labels)
    sel_row = ranked.iloc[labels.index(selected)]
    image_id = int(sel_row["image_id"])
    patient_id = sel_row["patient_id"]

    # Patient-ID + Rater-Filter (bild-spezifisch, außerhalb der Sidebar)
    col_pid, col_raters = st.columns([1, 3])
    with col_pid:
        st.metric("Patient-ID", patient_id)
    with col_raters:
        measurements_for_img = df4d[df4d["id"] == image_id]
        all_raters = [
            row.get("rater_name", f"Rater {i+1}")
            for i, (_, row) in enumerate(measurements_for_img.iterrows())
        ]
        visible_raters = st.multiselect(
            "Rater anzeigen",
            options=all_raters,
            default=all_raters,
            key=f"raters_{image_id}",
        )
    if not visible_raters:
        visible_raters = all_raters  # Fallback: alle zeigen wenn keiner gewählt

    # Figure (nur Bild mit Overlay)
    with st.spinner("Bild laden …"):
        fig = build_figure(image_id, cobb, df4d, img_urls,
                           smooth_sigma=smooth_sigma, extra_cobbs=extra_cobbs,
                           show_primary=show_primary, visible_raters=visible_raters)

    # Export-Bytes vor plt.close() erzeugen
    buf_jpg = io.BytesIO()
    fig.savefig(buf_jpg, format="jpeg", dpi=150, bbox_inches="tight")
    jpg_bytes = buf_jpg.getvalue()

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=100, bbox_inches="tight")
    png_b64 = base64.b64encode(buf_png.getvalue()).decode()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Export-Buttons
    fname = f"outlier_{image_id}_{cobb.replace('-', '_').replace('@', '_')}"
    col_dl, col_cb = st.columns([1, 1])
    with col_dl:
        st.download_button(
            label="⬇ Als JPG speichern",
            data=jpg_bytes,
            file_name=f"{fname}.jpg",
            mime="image/jpeg",
        )
    with col_cb:
        clipboard_html = f"""
<script>
async function copyToClipboard() {{
    const btn = document.getElementById('cbtn');
    try {{
        const b64 = '{png_b64}';
        const bin = atob(b64);
        const arr = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
        const blob = new Blob([arr], {{type: 'image/png'}});
        await navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]);
        btn.textContent = '✓ Kopiert!';
        setTimeout(() => btn.textContent = '📋 In Zwischenablage kopieren', 2000);
    }} catch(e) {{
        btn.textContent = '✗ ' + e.message;
        setTimeout(() => btn.textContent = '📋 In Zwischenablage kopieren', 3000);
    }}
}}
</script>
<button id="cbtn" onclick="copyToClipboard()"
    style="background:#262730;color:white;border:1px solid #555;
           padding:7px 18px;border-radius:6px;cursor:pointer;font-size:14px;width:100%">
    📋 In Zwischenablage kopieren
</button>
"""
        st.components.v1.html(clipboard_html, height=46)

    # Winkeltabelle unterhalb des Bildes
    cobbs_in_table = ([cobb] if show_primary else []) + extra_cobbs
    if cobbs_in_table:
        st.divider()
        styled_table = build_angle_table(image_id, cobbs_in_table, angles_df, df4d,
                                         visible_raters=visible_raters)
        if styled_table is not None:
            st.dataframe(styled_table, use_container_width=True, hide_index=True)
        else:
            st.caption("Keine Winkeldaten für die gewählten cobbTypes.")

    # Geglättete Inflection-Winkel (on-the-fly berechnet, nicht in angles_df)
    if smooth_sigma > 0:
        smooth_df = compute_smooth_inflection_df(image_id, df4d, smooth_sigma,
                                                 visible_raters=visible_raters)
        if not smooth_df.empty:
            smooth_cobbs = sorted(smooth_df["cobbType"].unique())
            st.divider()
            st.caption(
                f"Geglättete Inflection-Winkel  ·  σ = {smooth_sigma}  ·  "
                f"min ≥ {SMOOTH_MIN_ANGLE:.0f}°"
            )
            styled_smooth = build_angle_table(image_id, smooth_cobbs, smooth_df, df4d,
                                              visible_raters=visible_raters)
            if styled_smooth is not None:
                st.dataframe(styled_smooth, use_container_width=True, hide_index=True)
        else:
            if cobbs_in_table:
                st.caption(
                    f"Geglättete Inflection: keine Winkel ≥ {SMOOTH_MIN_ANGLE:.0f}° "
                    f"bei σ = {smooth_sigma}."
                )


if __name__ == "__main__":
    main()
