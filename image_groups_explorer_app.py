"""
image_groups Explorer — welche Gruppen enthalten mindestens die gewählten img_types,
aufgeteilt nach Label und Messstatus.

Start:
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run image_groups_explorer_app.py
"""
import altair as alt
import pandas as pd
import streamlit as st

from image_groups_overview import CACHE, load

st.set_page_config(page_title="image_groups Explorer", layout="wide")


@st.cache_data
def groups_df(data_key):  # data_key nur für den Cache-Key
    data = st.session_state["data"]
    imgs = {i["uuid"]: i for i in data}
    rows = {}
    for i in imgs.values():
        rp = i.get("referring_patient") or {}
        for g in rp.get("image_groups") or []:
            rows[(rp.get("uuid"), g.get("label"))] = (rp.get("patient_id"), g.get("uuids") or [])
    out = []
    for (_, label), (pid, uuids) in rows.items():
        # img_type -> [(image_uuid, Anzahl Messobjekte), ...]
        t = {}
        for u in uuids:
            if u in imgs:
                t.setdefault(imgs[u]["img_type"], []).append((u, len(imgs[u].get("measure_dataset") or [])))
        out.append({"label": label, "patient_id": pid, "types": t})
    return out


with st.sidebar:
    token = st.text_input("Bearer Token (optional, lädt neu)", type="password")
    if st.button("Von API laden", disabled=not token):
        st.session_state["data"] = load(token)
        st.session_state["ver"] = st.session_state.get("ver", 0) + 1
if "data" not in st.session_state:
    if not CACHE.exists():
        st.error("Kein Cache — Token eingeben und laden"); st.stop()
    st.session_state["data"] = load()
groups = groups_df(st.session_state.get("ver", 0))

all_types = sorted({t for g in groups for t in g["types"]})
sel = st.multiselect("img_types (Gruppe muss mindestens diese enthalten)", all_types)
labels = sorted({g["label"] for g in groups})
sel_labels = st.multiselect("Labels", labels, default=labels)

only_sel = st.checkbox("UUID-Export: nur Bilder der gewählten img_types (sonst alle Bilder der Gruppen)", value=True)
rows, uuid_rows = [], []
for g in groups:
    if g["label"] not in sel_labels or not set(sel) <= set(g["types"]):
        continue
    rel = sel or list(g["types"])  # ohne Auswahl: alle Typen der Gruppe
    n_ok = sum(any(n for _, n in g["types"][t]) for t in rel)
    status = "alle gemessen" if n_ok == len(rel) else "teilweise gemessen" if n_ok else "nicht gemessen"
    row = {"label": g["label"], "patient_id": g["patient_id"], "status": status}
    for t in all_types:  # Messobjekte je Bild, z.B. "2 | 0" = zwei Bilder
        row[t] = " | ".join(str(n) for _, n in g["types"].get(t, []))
    rows.append(row)
    for t, imgs_t in g["types"].items():
        if sel and t not in sel and only_sel:
            continue
        for u, n in imgs_t:
            uuid_rows.append({"label": g["label"], "patient_id": g["patient_id"], "group_status": status,
                              "img_type": t, "image_uuid": u, "n_messobjekte": n})
df = pd.DataFrame(rows)

st.caption("Status bezieht sich auf die gewählten img_types (ohne Auswahl: alle Typen der Gruppe). "
           "'gemessen' = mind. ein Bild des Typs hat ≥1 Messobjekt.")
if df.empty:
    st.info("Keine Gruppe passt."); st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Gruppen", len(df))
c2.metric("Patienten", df["patient_id"].nunique())
c3.metric("alle gemessen", int((df["status"] == "alle gemessen").sum()))

cnt = df.groupby(["label", "status"]).size().reset_index(name="n_gruppen")
st.altair_chart(
    alt.Chart(cnt).mark_bar().encode(
        x=alt.X("label:N", title=None),
        y=alt.Y("n_gruppen:Q", title="Gruppen"),
        color=alt.Color("status:N", scale=alt.Scale(
            domain=["alle gemessen", "teilweise gemessen", "nicht gemessen"],
            range=["#2a9d8f", "#e9c46a", "#c8c8c8"])),
        tooltip=["label", "status", "n_gruppen"],
    ), width="stretch")

st.subheader("Gruppen")
st.caption("Zellen: Messobjekte je Bild, ein Wert pro Bild des Typs (z.B. „2 | 0“ = zwei Bilder, eins mit 2 Messobjekten, eins ohne)")
st.dataframe(df.sort_values(["label", "status", "patient_id"]), hide_index=True, width="stretch",
             column_config={t: st.column_config.TextColumn(width="small") for t in all_types})

st.subheader("Export")
ex = pd.DataFrame(uuid_rows)
st.caption(f"{len(ex)} Bilder (ImageData-UUIDs), eine Zeile pro Bild")
c1, c2 = st.columns(2)
c1.download_button("UUIDs als CSV", ex.to_csv(index=False), "image_group_uuids.csv", "text/csv")
c2.download_button("Nur UUID-Liste (JSON)", ex["image_uuid"].to_json(orient="values", indent=1),
                   "image_group_uuids.json", "application/json")
with st.expander("Vorschau Export"):
    st.dataframe(ex, hide_index=True, width="stretch")
