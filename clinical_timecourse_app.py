"""
Klinischer Timecourse — Streamlit App

Analysen auf ClinicalEntryData und ClinicalBaseData:
  - Sankey: erste → letzte Therapie pro Patient
  - Sankey: Therapie → Komplikation
  - Symptome × MRT × Therapie (Parallel Categories + Heatmap)

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run clinical_timecourse_app.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

DATA_DIR = "data"
API_BASE = "https://xray-api.renecol.org/api"


# ---------------------------------------------------------------------------
# Fetch & cache
# ---------------------------------------------------------------------------

def _flatten(records):
    """Merge top-level API fields with the nested data sub-dict."""
    rows = []
    for r in records:
        row = {k: v for k, v in r.items() if k != "data"}
        row.update(r.get("data") or {})
        rows.append(row)
    return rows


def _get(token, endpoint):
    resp = requests.get(
        f"{API_BASE}/{endpoint}/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_and_cache(token):
    df_entries = pd.DataFrame(_flatten(_get(token, "clinicalentrydata")))
    df_base = pd.DataFrame(_flatten(_get(token, "clinicalbasedata")))
    if "referring_date" in df_entries.columns:
        df_entries["referring_date"] = pd.to_datetime(df_entries["referring_date"], errors="coerce")
    df_entries.to_pickle(f"{DATA_DIR}/clinical_entries.pkl")
    df_base.to_pickle(f"{DATA_DIR}/clinical_base.pkl")
    return df_entries, df_base


@st.cache_resource
def load_data():
    df_entries = pd.read_pickle(f"{DATA_DIR}/clinical_entries.pkl")
    if "referring_date" in df_entries.columns:
        df_entries["referring_date"] = pd.to_datetime(df_entries["referring_date"], errors="coerce")
    df_base = pd.read_pickle(f"{DATA_DIR}/clinical_base.pkl")
    return df_entries, df_base


# ---------------------------------------------------------------------------
# Sankey helper
# ---------------------------------------------------------------------------

def _sankey(source_labels, target_labels, values, title):
    """Separate left/right node sets so same-label self-flows render correctly.

    Patients with only one event have first == last therapy → they appear on
    both the left and right side of the diagram as two distinct nodes with the
    same label, connected by a flow.
    """
    src_nodes = sorted(set(source_labels))
    tgt_nodes = sorted(set(target_labels))
    src_idx = {v: i for i, v in enumerate(src_nodes)}
    tgt_idx = {v: i + len(src_nodes) for i, v in enumerate(tgt_nodes)}

    flow: dict[tuple, int] = {}
    for s, t, v in zip(source_labels, target_labels, values):
        key = (src_idx[s], tgt_idx[t])
        flow[key] = flow.get(key, 0) + v

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=src_nodes + tgt_nodes, pad=15, thickness=20),
        link=dict(
            source=[k[0] for k in flow],
            target=[k[1] for k in flow],
            value=list(flow.values()),
        ),
    ))
    fig.update_layout(title_text=title, font_size=12, height=460)
    return fig


# ---------------------------------------------------------------------------
# Analysis figures
# ---------------------------------------------------------------------------

def fig_therapy_flow(df: pd.DataFrame) -> go.Figure:
    """Sankey: first → last therapy per patient (single-event patients included)."""
    agg = (
        df.dropna(subset=["therapy"])
        .sort_values("referring_date")
        .groupby("patient")["therapy"]
        .agg(first="first", last="last")
        .reset_index()
    )
    return _sankey(
        agg["first"].tolist(),
        agg["last"].tolist(),
        [1] * len(agg),
        "Therapieverlauf: Erste → Letzte Therapie pro Patient",
    )


def _patient_complications(df: pd.DataFrame) -> pd.DataFrame:
    """Per patient: union of all complications across all their entries.

    Returns a long DataFrame with columns [patient, complication].
    Patients with no complications get a single row with complication='Keine'.
    """
    df_c = df.copy()
    df_c["complication"] = df_c["complication"].apply(
        lambda x: x if isinstance(x, list) and x else []
    )
    df_c = df_c.explode("complication")
    # empty list → NaN after explode; drop those, then deduplicate per patient
    df_c = df_c[df_c["complication"].notna() & (df_c["complication"] != "")]
    has_comp = df_c.groupby("patient")["complication"].apply(list).reset_index()
    has_comp = has_comp.explode("complication").drop_duplicates()

    all_patients = pd.DataFrame({"patient": df["patient"].unique()})
    merged = all_patients.merge(has_comp, on="patient", how="left")
    merged["complication"] = merged["complication"].fillna("Keine")
    return merged


def fig_therapy_complication(df: pd.DataFrame) -> go.Figure:
    """Sankey: therapy → complications of the same ClinicalEntry (exploded).

    Each entry contributes its own therapy → complication links.
    Entries without therapy are skipped; entries without complications → 'Keine'.
    """
    df_exp = df.dropna(subset=["therapy"]).copy()
    df_exp["complication"] = df_exp["complication"].apply(
        lambda x: x if isinstance(x, list) and x else ["Keine"]
    )
    df_exp = df_exp.explode("complication")
    counts = df_exp.groupby(["therapy", "complication"]).size().reset_index(name="n")
    return _sankey(
        counts["therapy"].tolist(),
        counts["complication"].tolist(),
        counts["n"].tolist(),
        "Therapie → Komplikation",
    )


def fig_complication_frequency(df: pd.DataFrame):
    """Bar chart: number of patients per complication type (excluding 'Keine')."""
    patient_comps = _patient_complications(df)
    counts = (
        patient_comps
        .groupby("complication")["patient"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="Patienten")
    )
    if counts.empty:
        return px.bar(title="Keine Komplikationen dokumentiert")
    return px.bar(
        counts,
        x="complication",
        y="Patienten",
        title="Komplikationshäufigkeit (eindeutige Patienten)",
        labels={"complication": "Komplikation", "Patienten": "Anzahl Patienten"},
    )


def fig_parallel_categories(df: pd.DataFrame):
    df_pc = df.dropna(subset=["therapy"]).copy()
    df_pc["Symptom"] = df_pc["spinal_symptoms"].apply(
        lambda x: x[0] if isinstance(x, list) and x else "Keine"
    )
    df_pc["MRT-Merkmal"] = df_pc["mri_characteristics"].apply(
        lambda x: x[0] if isinstance(x, list) and x else "Keine"
    )
    df_pc["Therapiesegment"] = df_pc["therapy_specification"].apply(
        lambda x: x[0] if isinstance(x, list) and x else "Keine"
    )
    return px.parallel_categories(
        df_pc,
        dimensions=["Symptom", "MRT-Merkmal", "Therapiesegment", "therapy"],
        labels={"therapy": "Therapie"},
        title="Symptome / MRT-Merkmal / Therapiesegment / Therapie",
    )


def fig_heatmap(df: pd.DataFrame):
    df_h = df.copy()
    for col in ("spinal_symptoms", "mri_characteristics"):
        df_h[col] = df_h[col].apply(lambda x: x if isinstance(x, list) else [])
    df_h = df_h.explode("spinal_symptoms").explode("mri_characteristics")
    df_h = df_h[(df_h["spinal_symptoms"].notna()) & (df_h["spinal_symptoms"] != "")]
    df_h = df_h[(df_h["mri_characteristics"].notna()) & (df_h["mri_characteristics"] != "")]
    pivot = (
        df_h.groupby(["spinal_symptoms", "mri_characteristics"])
        .size()
        .unstack(fill_value=0)
    )
    return px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Symptome × MRT-Merkmale (Häufigkeit)",
        labels={"x": "MRT-Merkmal", "y": "Symptom", "color": "Anzahl"},
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def _sidebar(df_entries, df_base):
    """Handles data reload and base-data filters. Returns filtered df_entries."""
    st.sidebar.header("Daten")
    if df_entries is not None:
        st.sidebar.metric("Klinische Ereignisse", len(df_entries))
    if df_base is not None:
        st.sidebar.metric("Patienten (Stammdaten)", len(df_base))
    st.sidebar.divider()
    st.sidebar.subheader("Neu laden")
    st.sidebar.caption("Token aus xray-annotator.renecol.org → Account info → Copy token")
    token = st.sidebar.text_input("Bearer Token", type="password")
    if st.sidebar.button("Laden", use_container_width=True):
        if not token.strip():
            st.sidebar.error("Bitte Token eingeben.")
        else:
            with st.spinner("Lade Daten von API …"):
                try:
                    fetch_and_cache(token.strip())
                    load_data.clear()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Fehler: {e}")

    if df_entries is None or df_base is None:
        return df_entries

    st.sidebar.divider()
    st.sidebar.subheader("Filter (Stammdaten)")

    # Sex
    sex_options = sorted(df_base["sex"].dropna().unique().tolist())
    sel_sex = st.sidebar.multiselect("Geschlecht", sex_options, default=sex_options)

    # Diagnoses (array field → explode to unique values)
    all_diagnoses = sorted({
        d for diags in df_base["diagnoses"].dropna()
        for d in (diags if isinstance(diags, list) else [])
    })
    sel_diagnoses = st.sidebar.multiselect("Diagnosen", all_diagnoses, default=all_diagnoses)

    # Build patient mask on df_base, then restrict df_entries
    mask = pd.Series(True, index=df_base.index)
    if len(sel_sex) < len(sex_options):
        mask &= df_base["sex"].isin(sel_sex)
    if len(sel_diagnoses) < len(all_diagnoses):
        mask &= df_base["diagnoses"].apply(
            lambda x: isinstance(x, list) and any(d in x for d in sel_diagnoses)
        )

    filtered_patients = set(df_base.loc[mask, "patient"].tolist())
    return df_entries[df_entries["patient"].isin(filtered_patients)]


def main():
    st.set_page_config(page_title="Klinischer Timecourse", layout="wide")
    st.title("Klinischer Timecourse — Patientenstatistiken")

    try:
        df_entries, df_base = load_data()
    except FileNotFoundError:
        df_entries, df_base = None, None

    df_entries = _sidebar(df_entries, df_base)

    if df_entries is None:
        st.info("Noch keine Daten vorhanden. Bitte Token eingeben und laden.")
        return

    # Drop entries without therapy globally — applies to all figures
    df_entries = df_entries[
        df_entries["therapy"].notna() & (df_entries["therapy"].str.strip() != "")
    ]

    tab1, tab2, tab3 = st.tabs(
        ["Therapieverlauf", "Therapie × Komplikation", "Symptome / MRT / Therapie"]
    )

    with tab1:
        st.plotly_chart(fig_therapy_flow(df_entries), use_container_width=True)

    with tab2:
        st.plotly_chart(fig_therapy_complication(df_entries), use_container_width=True)
        st.plotly_chart(fig_complication_frequency(df_entries), use_container_width=True)

    with tab3:
        sub1, sub2 = st.tabs(["Parallel Categories", "Heatmap"])
        with sub1:
            st.plotly_chart(fig_parallel_categories(df_entries), use_container_width=True)
        with sub2:
            st.plotly_chart(fig_heatmap(df_entries), use_container_width=True)


if __name__ == "__main__":
    main()
