"""
ICFx + XrayAPI — Wissenschaftliche Analyse
SF36 · WHODAS · ICF · Klinische Daten

Start:
    cd /Users/peterbernstein/Python/XrayStats
    streamlit run icfx_app.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Konstanten ────────────────────────────────────────────────────────────────

XRAY_API = "https://xray-api.renecol.org/"
ICFX_API = "https://static.94.87.140.128.clients.your-server.de/icfxapi/"
RES_BASE = "http://icfx-ressources.renecol.org/"

# ── Auth helpers ──────────────────────────────────────────────────────────────

def get_icfx_token(xray_token: str) -> str:
    """Exchange xray token for ICFx institution token."""
    resp = requests.get(
        XRAY_API + "api/icfxtoken/",
        headers={"Authorization": f"Bearer {xray_token}"},
    )
    resp.raise_for_status()
    return resp.json()["token"]


# ── Data loading (cached per token) ──────────────────────────────────────────

@st.cache_data(show_spinner="Lade ICFx-Daten …")
def load_icfx_data(icfx_token: str) -> pd.DataFrame:
    r = requests.get(
        ICFX_API + "icfdata/",
        headers={"Authorization": f"Bearer {icfx_token}"},
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()).drop(columns=["owner_institution"], errors="ignore")
    df = df[df["owner"] == df["creator"]].copy()
    if "created" in df.columns:
        df["created"] = pd.to_datetime(df["created"])
        df = df.sort_values("created")
    df = pd.concat([df, df["data"].apply(pd.Series)], axis=1)
    return df


def deduplicate_icfx(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """One row per patient (owner). strategy: 'latest' | 'most_complete'"""
    if strategy == "latest":
        # sort is already chronological; last() takes final row per owner
        return df.groupby("owner", sort=False).last().reset_index()
    else:  # most_complete: row with fewest NaN instrument columns
        instrument_cols = [c for c in ("sf36", "whodas", "icf", "env") if c in df.columns]
        df = df.copy()
        df["_completeness"] = df[instrument_cols].notna().sum(axis=1)
        idx = df.groupby("owner")["_completeness"].idxmax()
        return df.loc[idx].drop(columns="_completeness").reset_index(drop=True)


@st.cache_data(show_spinner="Lade xray_api Patienten …")
def load_xray_patients(xray_token: str) -> pd.DataFrame:
    """
    Step 1+2: fetch all patients, keep only those with an ICFx account.
    hasICFxEntry is the ICFx UUID (string) or 0 when not set.
    """
    resp = requests.get(
        XRAY_API + "api/patients/",
        headers={"Authorization": f"Bearer {xray_token}"},
    )
    resp.raise_for_status()
    rows = []
    for p in resp.json():
        icfx_uuid = p.get("hasICFxEntry") or ""
        if not icfx_uuid:
            continue  # step 2: skip patients without ICFx account
        base = p.get("clinicalbasedata") or {}
        data = base.get("data") or {}
        entries = p.get("clinicalentrydata") or []
        last_entry_data = entries[-1]["data"] if entries else {}
        rows.append({
            "patient_uuid":    p["uuid"],
            "patient_id":      p.get("patient_id", ""),
            "icfx_uuid":       icfx_uuid,
            "birthdate":       data.get("birthdate"),
            "sex":             data.get("sex"),
            "diagnoses":       ", ".join(data.get("diagnoses") or []),
            "height":          data.get("height"),
            "weight":          data.get("weight"),
            "medication":      ", ".join(data.get("medication") or []),
            "therapy":         last_entry_data.get("therapy", ""),
            "fused_segments":  last_entry_data.get("fused_segments"),
            "fused_to_pelvis": last_entry_data.get("fused_to_pelvis"),
            "scoliosis_type":  last_entry_data.get("scoliosis_type", ""),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Lade Definitionen …")
def load_definitions() -> dict:
    def fetch(name):
        return requests.get(RES_BASE + name).json()
    return {
        "whodas":      fetch("whodas12_de.json"),
        "sf36":        fetch("sf36_de.json"),
        "sf36_cats":   fetch("sf36_categories_de.json"),
    }


# ── Normdaten ─────────────────────────────────────────────────────────────────

NORMDATA_PATH = "/Users/peterbernstein/Vue/icfx-app2/src/assets/sf36_normdata.json"
AGE_BANDS = [(18, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79)]
SEX_MAP = {
    "weiblich": "f", "female": "f", "w": "f", "f": "f",
    "männlich": "m", "male":   "m", "m": "m",
}

@st.cache_data
def load_normdata() -> dict:
    with open(NORMDATA_PATH) as fh:
        return json.load(fh)


def age_sex_key(birthdate_str, sex: str) -> str | None:
    """Maps birthdate + sex string → normdata key like 'f_50-59'. Returns None if unmatchable."""
    try:
        age = (pd.Timestamp.now() - pd.to_datetime(birthdate_str)).days // 365
    except Exception:
        age = None
    prefix = SEX_MAP.get(str(sex).strip().lower(), "t")
    if age is None:
        return None
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{prefix}_{lo}-{hi}"
    return None


def norm_comparison(df_sf36: pd.DataFrame, df_all: pd.DataFrame, df_xray: pd.DataFrame,
                    normdata: dict, cats_def: list) -> pd.DataFrame:
    """
    Per-patient: look up matched normdata (age+sex). Return DataFrame with one row
    per SF36 subscale: patient_mean, patient_std, norm_mid, norm_min, norm_max.
    norm_* are the mean across all per-patient matched reference values.
    """
    abbr_to_title = {c["abbr"]: c["title"] for c in cats_def}
    title_to_abbr = {v: k for k, v in abbr_to_title.items()}

    # build metadata only for rows that survived dropna in compute_sf36
    meta = df_all.loc[df_sf36.index, ["owner"]].merge(
        df_xray[["icfx_uuid", "birthdate", "sex"]],
        left_on="owner", right_on="icfx_uuid", how="left"
    )
    meta.index = df_sf36.index

    rows = []
    for title in df_sf36.columns:
        abbr = title_to_abbr.get(title)
        if not abbr:
            continue
        patient_vals = df_sf36[title].dropna()
        # per-patient matched norm midpoint
        norm_mids = []
        for idx in patient_vals.index:
            key = age_sex_key(meta.loc[idx, "birthdate"], meta.loc[idx, "sex"])
            if key is None:
                key = f"t_{AGE_BANDS[2][0]}-{AGE_BANDS[2][1]}"  # fallback: total 40-49
            lo = normdata.get(f"{abbr}_min", {}).get(key)
            hi = normdata.get(f"{abbr}_max", {}).get(key)
            if lo is not None and hi is not None:
                norm_mids.append((lo + hi) / 2)
        rows.append({
            "subscale":    title,
            "patient_mean": patient_vals.mean(),
            "patient_std":  patient_vals.std(),
            "norm_mid":     np.mean(norm_mids) if norm_mids else np.nan,
            "norm_min":     np.mean([normdata.get(f"{abbr}_min", {}).get(
                                age_sex_key(meta.loc[i, "birthdate"], meta.loc[i, "sex"])
                                or f"t_{AGE_BANDS[2][0]}-{AGE_BANDS[2][1]}", np.nan)
                               for i in patient_vals.index]),
            "norm_max":     np.mean([normdata.get(f"{abbr}_max", {}).get(
                                age_sex_key(meta.loc[i, "birthdate"], meta.loc[i, "sex"])
                                or f"t_{AGE_BANDS[2][0]}-{AGE_BANDS[2][1]}", np.nan)
                               for i in patient_vals.index]),
        })
    return pd.DataFrame(rows).set_index("subscale")


# ── SF36 ──────────────────────────────────────────────────────────────────────

def compute_sf36(df_patients: pd.DataFrame, defs: dict) -> pd.DataFrame:
    sf36_def   = defs["sf36"]
    cats_def   = defs["sf36_cats"]
    df_sf36    = df_patients["sf36"].apply(pd.Series).dropna(how="all")
    if df_sf36.empty:
        return pd.DataFrame()

    def normalizer(s):
        key  = s.name
        if key not in sf36_def:
            return s
        n    = len(sf36_def[key]["answers"])
        default = sf36_def[key]["default"]
        f    = 100 / n
        return f * (s + 1) if default == n - 1 else f * (n - s)

    df_n  = df_sf36.apply(normalizer)
    out   = pd.DataFrame(index=df_sf36.index)
    for cat in cats_def:
        cols = [c for c in cat["keys"] if c in df_n.columns]
        if cols:
            out[cat["abbr"]] = df_n[cols].mean(axis=1)
    out.rename(columns={c["abbr"]: c["title"] for c in cats_def}, inplace=True)
    return out


# ── WHODAS ────────────────────────────────────────────────────────────────────

def compute_whodas(df_patients: pd.DataFrame, defs: dict) -> pd.DataFrame:
    whodas_def = defs["whodas"]
    df_w       = df_patients["whodas"].apply(pd.Series).dropna(how="all")
    if df_w.empty:
        return pd.DataFrame()
    rename = {c: whodas_def[c]["s"] + " (WHODAS)" for c in df_w.columns if c in whodas_def}
    df_w   = df_w.rename(columns=rename)
    # WHODAS: 1=kein Problem … 5=volles Problem → invertiert auf 0-100
    df_w_n = (5 - df_w) * 20
    df_w_n["Gesamt (WHODAS)"] = df_w_n.mean(axis=1)
    return df_w_n


# ── ICF ───────────────────────────────────────────────────────────────────────

def compute_icf(df_patients: pd.DataFrame) -> pd.DataFrame:
    def explode(d):
        if not isinstance(d, dict):
            return {}
        return {k: v["value"] for k, v in d.items() if isinstance(v, dict) and v.get("selected") == 1}

    df_icf = df_patients["icf"].apply(explode).apply(pd.Series)
    # nur Beeinträchtigungs-Codes (d…), keine Umweltfaktoren (e…)
    d_cols = [c for c in df_icf.columns if not str(c).startswith("e")]
    return df_icf[d_cols]


# ── Merge xray + ICFx ─────────────────────────────────────────────────────────

def merge_datasets(df_icfx: pd.DataFrame, df_xray: pd.DataFrame) -> pd.DataFrame:
    """Join key: df_icfx.owner == df_xray.icfx_uuid"""
    return df_icfx.rename(columns={"owner": "icfx_uuid"}).merge(df_xray, on="icfx_uuid", how="inner")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ICFx Analyse", layout="wide")
st.title("ICFx · SF36 · WHODAS · ICF — Wissenschaftliche Auswertung")

# Sidebar: Token-Eingabe (Token aus xray-annotator kopieren)
with st.sidebar:
    st.subheader("Zugang")
    token_input = st.text_input(
        "Bearer Token",
        type="password",
        help="xray-annotator → Account-Info → Token kopieren",
    )
    if st.button("Verbinden"):
        if not token_input.strip():
            st.error("Bitte Token eingeben.")
        else:
            try:
                icfx_token = get_icfx_token(token_input.strip())
                st.session_state["xray_token"] = token_input.strip()
                st.session_state["icfx_token"] = icfx_token
                st.cache_data.clear()
                st.success("Verbunden.")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler: {e}")

    if "xray_token" in st.session_state:
        st.caption("✓ Token aktiv")
        if st.button("Token löschen"):
            for k in ("xray_token", "icfx_token"):
                st.session_state.pop(k, None)
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.subheader("Datensatz-Strategie")
    dedup_strategy = st.radio(
        "Mehrfacheinträge pro Patient",
        options=["latest", "most_complete"],
        format_func=lambda x: "Letzter Eintrag" if x == "latest" else "Vollständigster Eintrag",
        help="Patienten können mehrere ICFx-Datensätze haben. Hier wählen, welcher verwendet wird.",
    )

if "icfx_token" not in st.session_state:
    st.info("Bitte links Bearer Token eingeben (aus xray-annotator → Account-Info).")
    st.stop()

# Load data — in the correct order
xray_token = st.session_state["xray_token"]
icfx_token = st.session_state["icfx_token"]

defs = load_definitions()

# Step 1+2: patients from xray_api, filtered to those with hasICFxEntry
df_xray = load_xray_patients(xray_token)
icfx_uuids = set(df_xray["icfx_uuid"]) if not df_xray.empty else set()

# Step 3+4: fetch all ICFx data, filter to the UUIDs from step 2, deduplicate
df_icfx_raw = load_icfx_data(icfx_token)
df_filtered = df_icfx_raw[df_icfx_raw["owner"].isin(icfx_uuids)].reset_index(drop=True)
df_all = deduplicate_icfx(df_filtered, dedup_strategy)

df_sf36   = compute_sf36(df_all, defs)
df_whodas = compute_whodas(df_all, defs)
df_icf    = compute_icf(df_all)

# Metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("ICFx-Datensätze (eindeutig)", len(df_all))
c2.metric("Mit SF36",  int(df_sf36.notna().any(axis=1).sum())  if not df_sf36.empty  else 0)
c3.metric("Mit WHODAS", int(df_whodas.notna().any(axis=1).sum()) if not df_whodas.empty else 0)
c4.metric("Verknüpft (xray_api)", len(merge_datasets(df_all, df_xray)))

# Tabs
tab_sf36, tab_whodas, tab_icf, tab_combined, tab_raw = st.tabs(
    ["SF36", "WHODAS", "ICF", "Kombiniert", "Rohdaten"]
)

# ── Tab: SF36 ─────────────────────────────────────────────────────────────────
with tab_sf36:
    if df_sf36.empty:
        st.warning("Keine SF36-Daten vorhanden.")
    else:
        normdata  = load_normdata()
        cats_def  = defs["sf36_cats"]
        df_norm   = norm_comparison(df_sf36, df_all, df_xray, normdata, cats_def)

        st.subheader("SF36 — Kohorte vs. alters-/geschlechtsgematchte Normbevölkerung")
        st.caption("Normdaten: Ellert & Kurth (2013), DEGS1, deutsche Bevölkerung. "
                   "Normband = 95%-KI des Bevölkerungsdurchschnitts, gematcht nach Alter und Geschlecht.")

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos   = np.arange(len(df_norm))
        height  = 0.35

        # norm CI band als horizontale Balken (hellgrau)
        ax.barh(y_pos + height / 2, df_norm["norm_max"] - df_norm["norm_min"],
                left=df_norm["norm_min"], height=height,
                color="lightgray", label="Norm 95%-KI (gematcht)")
        # norm midpoint
        ax.scatter(df_norm["norm_mid"], y_pos + height / 2,
                   color="dimgray", zorder=3, s=30, marker="|", linewidths=2)

        # Patienten-Mittelwert ± SD
        ax.barh(y_pos - height / 2, df_norm["patient_mean"],
                height=height, color="steelblue", alpha=0.85, label="Kohorte (Mittelwert)")
        ax.errorbar(df_norm["patient_mean"], y_pos - height / 2,
                    xerr=df_norm["patient_std"], fmt="none",
                    color="navy", capsize=3, linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_norm.index, fontsize=9)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score (0–100)")
        ax.axvline(50, color="salmon", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.legend(fontsize=8, loc="lower right")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Differenz Kohorte − Norm (Mittelwert)")
        df_diff = (df_norm["patient_mean"] - df_norm["norm_mid"]).rename("Δ (Patient − Norm)")
        col_a, col_b = st.columns(2)
        with col_a:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            colors = ["steelblue" if v >= 0 else "tomato" for v in df_diff]
            ax3.barh(df_diff.index, df_diff.values, color=colors)
            ax3.axvline(0, color="black", linewidth=0.8)
            ax3.set_xlabel("Δ Score")
            ax3.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
        with col_b:
            st.dataframe(
                pd.concat([df_norm[["patient_mean", "patient_std", "norm_mid",
                                    "norm_min", "norm_max"]], df_diff], axis=1).round(1),
                use_container_width=True,
            )

        st.subheader("Verteilung je Kategorie (Boxplot)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        df_sf36.boxplot(ax=ax2, rot=30, grid=False)
        ax2.set_ylabel("Score (0–100)")
        ax2.set_ylim(0, 105)
        # overlay matched norm midpoints as red markers
        for i, col in enumerate(df_sf36.columns):
            if col in df_norm.index:
                ax2.plot(i + 1, df_norm.loc[col, "norm_mid"], "r_",
                         markersize=12, markeredgewidth=2)
        ax2.plot([], [], "r_", markersize=10, markeredgewidth=2, label="Norm (Mittelwert)")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ── Tab: WHODAS ───────────────────────────────────────────────────────────────
with tab_whodas:
    if df_whodas.empty:
        st.warning("Keine WHODAS-Daten vorhanden.")
    else:
        st.subheader("WHODAS — Domänen (0–100, höher = besser)")
        col_a, col_b = st.columns(2)

        with col_a:
            means_w = df_whodas.mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(means_w.index, means_w.values, color="darkorange")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score (invertiert)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.dataframe(
                df_whodas.describe().T[["mean", "std", "min", "max"]].round(1),
                use_container_width=True
            )

        st.subheader("Korrelationsmatrix WHODAS")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        mask = df_whodas.isna().all(axis=0)
        corr = df_whodas.loc[:, ~mask].corr()
        sns.heatmap(corr, ax=ax3, cmap="rocket_r", vmin=0, vmax=1, annot=False)
        plt.xticks(fontsize=7, rotation=45, ha="right")
        plt.yticks(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()


# ── Tab: ICF ──────────────────────────────────────────────────────────────────
with tab_icf:
    if df_icf.empty:
        st.warning("Keine ICF-Daten vorhanden.")
    else:
        st.subheader("ICF — Beeinträchtigungs-Score nach Code")
        n_patients = df_icf.notna().any(axis=1).sum()
        st.caption(f"{n_patients} Patienten mit ICF-Einträgen, {len(df_icf.columns)} Codes")

        prevalence = df_icf.notna().mean().sort_values(ascending=False).head(30)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(prevalence.index.astype(str), prevalence.values * 100, color="mediumseagreen")
            ax.set_ylabel("Prävalenz (%)")
            ax.set_xlabel("ICF-Code")
            plt.xticks(rotation=60, fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col_b:
            st.dataframe(
                pd.DataFrame({"Prävalenz %": (prevalence * 100).round(1),
                              "Mittlere Ausprägung": df_icf[prevalence.index].mean().round(2)}),
                use_container_width=True,
            )

        st.subheader("Mittlere Ausprägung (Top 20, wo vorhanden)")
        top20 = df_icf[prevalence.index[:20]].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(top20.index.astype(str), top20.values, color="steelblue")
        ax2.set_ylabel("Mittlere Ausprägung (1–4)")
        ax2.set_ylim(0, 4.5)
        plt.xticks(rotation=45, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ── Tab: Kombiniert ───────────────────────────────────────────────────────────
with tab_combined:
    st.subheader("SF36 × WHODAS — Korrelation")
    if df_sf36.empty or df_whodas.empty:
        st.warning("SF36 und/oder WHODAS fehlen für Kombinations-Analyse.")
    else:
        df_combo = pd.concat([df_sf36, df_whodas], axis=1).dropna(how="all")
        if len(df_combo) < 3:
            st.warning("Zu wenig gemeinsame Datensätze.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df_combo.corr()
            sns.heatmap(corr, ax=ax, cmap="rocket_r", vmin=-1, vmax=1,
                        linewidths=0.3, linecolor="white")
            plt.xticks(fontsize=7, rotation=45, ha="right")
            plt.yticks(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(f"n = {len(df_combo)}")

    st.divider()
    st.subheader("Verknüpfung mit xray_api (Join Key: icfx_uuid)")
    df_merged = merge_datasets(df_all, df_xray)
    if df_merged.empty:
        st.warning("Keine gemeinsamen Patienten zwischen xray_api und ICFx gefunden.")
        st.info(
            "Mögliche Ursache: `clinicalbasedata.data.icfx_uuid` ist bei den Patienten "
            "noch nicht gesetzt, oder es wurden noch keine ICFx-Datensätze angelegt."
        )
    else:
        st.success(f"{len(df_merged)} Patienten verknüpft.")
        st.subheader("SF36 nach Therapie")
        if "therapy" in df_merged.columns and not df_sf36.empty:
            # df_sf36 shares its integer index with df_all; df_merged has the same rows
            sf36_merged = df_sf36.join(df_merged["therapy"], how="inner")
            sf36_by_therapy = sf36_merged.groupby("therapy").mean().T
            if not sf36_by_therapy.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                sf36_by_therapy.plot(kind="bar", ax=ax)
                ax.set_ylabel("SF36 Score")
                ax.set_ylim(0, 100)
                plt.xticks(rotation=30, ha="right", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.subheader("Verknüpfte Datentabelle")
        show_cols = ["patient_id", "sex", "birthdate", "diagnoses", "therapy",
                     "fused_segments", "scoliosis_type"]
        show_cols = [c for c in show_cols if c in df_merged.columns]
        st.dataframe(df_merged[show_cols], use_container_width=True)


# ── Tab: Rohdaten ─────────────────────────────────────────────────────────────
with tab_raw:
    st.subheader(f"xray_api Patienten mit ICFx-Account ({len(df_xray)})")
    st.dataframe(df_xray, use_container_width=True)

    n_raw = len(df_filtered)
    n_dedup = len(df_all)
    st.subheader(f"ICFx-Datensätze ({n_dedup} nach Dedup, {n_raw} roh)")
    if n_raw > n_dedup:
        st.caption(f"⚠ {n_raw - n_dedup} Mehrfacheinträge entfernt (Strategie: {dedup_strategy})")
    show = df_all.drop(columns=["data", "sf36", "whodas", "icf"], errors="ignore")
    st.dataframe(show, use_container_width=True)
    st.divider()
    st.caption("Join Key: `xray_api.hasICFxEntry` (= `clinicalbasedata.data.icfx_uuid`) = `ICFx.icfdata.owner`")

