"""
Corner Annotate App — Streamlit

Start:
    cd /Users/peterbernstein/Python/XrayStats
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run corner_annotate_app.py

Ground Truth für das Slip-Modell (~/Python/Xray-Model/train_slip.ipynb): je Wirbel auf seitlichen
LWS-Aufnahmen (IMG_TYPES) die Hinterkante als PS (hinten oben) und PI (hinten unten) setzen, S1 ebenso.
  - blättern (◀ ▶), zu einer Bild-ID springen, "nächstes zu bearbeitendes Bild"; bearbeitete Bilder
    zeigen ihre gespeicherten Ecken, einzelne Ecken lassen sich neu setzen
  - gezoomter Ausschnitt um den aktuellen Wirbel, Klick setzt die angezeigte Ecke und springt weiter
  - Reihenfolge: liegt data/slip_priority.csv vor (letzte Zelle von train_slip.ipynb), kommen die Bilder
    mit dem größten vom Modell vorhergesagten |Slip| zuerst — große Slips sind im Training knapp
  - "nicht beurteilbar" statt raten (Implantat, Überlagerung)
  - instrumentierte Wirbel ankreuzen (Confounder in der Auswertung)
  - Durchgang 2 = Wiederholung an REPEAT_N bereits annotierten Bildern, blind (Intra-Rater-Referenz)
Speichert nach data/corner_annotations.csv in Canvas-Koordinaten (Bildhöhe 600, wie die Nodes) und
legt die Vorhersage des klassischen spine_geom.posterior_wall als Baseline daneben (classic_*; bei S1
ist classic_pi das ANTERIORE Ende der Sakrum-Deckplatte, Slip L5/S1 dort nach sacral_slip).
"""

import hashlib
import io
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

from prepare_outlier_data import fetch_img_urls
from spine_geom import SPINE_LABELS
from spine_geom.posterior_wall import posterior_walls

DATA_DIR = "data"
OUT_FILE = os.environ.get("CORNER_OUT", f"{DATA_DIR}/corner_annotations.csv")   # CORNER_OUT: Testkopie
PRIORITY_FILE = os.environ.get("CORNER_PRIO", f"{DATA_DIR}/slip_priority.csv")
STAGE_H = 600            # Nodes liegen im Konva-Stage-Raum (Bild auf Höhe 600 skaliert)
REPEAT_N = 20
IMG_TYPES = ["Xray-LWS-lat", "Xray-LWS-incl", "Xray-LWS-recl", "MRI-LWS-sag"]   # alle anterior links
ZOOM_PX = 620
ANNOTATE_FROM = "T12"      # thorakal oberhalb: kein Slip-Interesse, Kanten unter Rippen/Zwerchfell kaum sichtbar
CORNER_TXT = {"PS": "hinten OBEN (PS)", "PI": "hinten UNTEN (PI)"}
COLS = ["image_id", "patient_id", "img_type", "m_uuid", "pass", "level", "cx", "cy", "wkd",
        "ps_x", "ps_y", "pi_x", "pi_y", "instrumented",
        "classic_ps_x", "classic_ps_y", "classic_pi_x", "classic_pi_y", "timestamp"]


@st.cache_resource
def load_data():
    df4d = pd.read_pickle(f"{DATA_DIR}/df4d_corrected_spine_measurements.pkl")
    with open(f"{DATA_DIR}/img_urls.pkl", "rb") as fh:
        img_urls = pickle.load(fh)
    return df4d, img_urls


@st.cache_data(show_spinner="Messungen werden geladen …")
def build_index(_df4d: pd.DataFrame) -> pd.DataFrame:
    """Je Bild (IMG_TYPES) die erste Messung mit vollständigen Nodes; Reihenfolge per Hash (stabil, zufällig)."""
    rows = []
    for _, r in _df4d[_df4d.img_type.isin(IMG_TYPES)].iterrows():
        so = r.xray_measurement_data.get("stateObject", {})
        nodes, vr = so.get("nodes", []), so.get("vertebra_range", [0, 0])
        labels = SPINE_LABELS[vr[0]:vr[1] + 1]
        if len(nodes) < 3 or len(nodes) != len(labels):
            continue
        rows.append({"image_id": int(r.id), "patient_id": r.patient_id, "img_type": r.img_type, "m_uuid": r.m_uuid,
                     "labels": labels, "nodes": [(n["x"], n["y"]) for n in nodes],
                     "wkd": (r.xray_measurement_data.get("calibration") or {}).get("length", np.nan)})
    idx = pd.DataFrame(rows).drop_duplicates("image_id")
    idx["order"] = idx.image_id.map(lambda i: hashlib.md5(f"corner:{i}".encode()).hexdigest())
    return idx.sort_values("order").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fetch_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("L")
    except Exception:
        return None


def load_annotations() -> pd.DataFrame:
    return pd.read_csv(OUT_FILE) if os.path.exists(OUT_FILE) else pd.DataFrame(columns=COLS)


def save_image(row, pass_no: int, pts: dict, instrumented: list, img: Image.Image | None, skip=False):
    ann = load_annotations()
    ann = ann[~((ann.image_id == row.image_id) & (ann["pass"] == pass_no))]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"image_id": row.image_id, "patient_id": row.patient_id, "img_type": row.img_type, "m_uuid": row.m_uuid,
            "pass": pass_no, "wkd": row.wkd, "timestamp": ts}
    if skip:
        new = [{**base, "level": "SKIP"}]
    else:
        scale = img.height / STAGE_H
        walls = posterior_walls(np.asarray(img, float) / 255, np.array(row.nodes) * scale, row.labels)
        new = []
        for lab, (cx, cy), w in zip(row.labels, row.nodes, walls):
            if SPINE_LABELS.index(lab) < SPINE_LABELS.index(ANNOTATE_FROM):
                continue
            rec = {**base, "level": lab, "cx": cx, "cy": cy, "instrumented": lab in instrumented}
            for c in ("PS", "PI"):
                p = pts.get((lab, c))
                rec[f"{c.lower()}_x"], rec[f"{c.lower()}_y"] = p if p else (np.nan, np.nan)
            if w is not None:
                (rec["classic_ps_x"], rec["classic_ps_y"]), (rec["classic_pi_x"], rec["classic_pi_y"]) = \
                    np.asarray(w[0]) / scale, np.asarray(w[1]) / scale
            new.append(rec)
    pd.concat([ann, pd.DataFrame(new)], ignore_index=True).reindex(columns=COLS).to_csv(OUT_FILE, index=False)


def spacing(nodes, i):
    c = np.asarray(nodes, float)
    nb = [np.linalg.norm(c[j] - c[i]) for j in (i - 1, i + 1) if 0 <= j < len(c)]
    return float(np.mean(nb))


def draw_points(im: Image.Image, pts: dict, box, factor, r=4):
    """Gesetzte Ecken (grün) als Kreuze in einen (gezoomten) Ausschnitt zeichnen. pts in Bild-px."""
    d = ImageDraw.Draw(im)
    for (lab, c), p in pts.items():
        if p is None:
            continue
        x, y = (p[0] - box[0]) * factor, (p[1] - box[1]) * factor
        d.line([(x - r, y), (x + r, y)], fill=(0, 255, 0), width=2)
        d.line([(x, y - r), (x, y + r)], fill=(0, 255, 0), width=2)
    return im




def model_slips(row) -> str:
    """Vorhergesagte Slips aller Segmente, |Slip| ≥ 10 % fett; ältere Prioritätsdateien haben nur das größte."""
    if not isinstance(getattr(row, "alle", None), str):
        return f"**{row.segment} {row.slip:+.1f}**"
    segs = [part.rsplit(" ", 1) for part in row.alle.split(" | ")]
    return " · ".join(f"**{n} {v}**" if abs(float(v)) >= 10 else f"{n} {v}" for n, v in segs)


def next_open(ids: list, done: set, start: int):
    """Index des nächsten nicht bearbeiteten Bildes ab start (zyklisch), None wenn alle bearbeitet."""
    for j in range(len(ids)):
        if ids[(start + j) % len(ids)] not in done:
            return (start + j) % len(ids)
    return None


def main():
    st.set_page_config(page_title="Ecken annotieren", layout="wide", page_icon="📐")
    df4d, img_urls = load_data()
    index = build_index(df4d)
    ann = load_annotations()
    ss = st.session_state

    with st.sidebar:
        pass_no = st.radio("Durchgang", [1, 2], format_func=lambda p: "1 – Annotation" if p == 1
                           else f"2 – Wiederholung ({REPEAT_N} Bilder, blind)")
        types = st.multiselect("Bildtypen", IMG_TYPES, default=IMG_TYPES)
        prio = (pd.read_csv(PRIORITY_FILE).reindex(columns=["image_id", "segment", "slip", "alle"])
                if os.path.exists(PRIORITY_FILE) else None)
        order = st.radio("Reihenfolge", ["Modell: größter |Slip| zuerst", "Zufall"],
                         disabled=prio is None or pass_no == 2, help=f"braucht {PRIORITY_FILE}")
        done1 = set(ann.loc[ann["pass"] == 1, "image_id"])
        pool = index[index.img_type.isin(types)]
        if prio is not None:                                        # ohne Vorhersage (z. B. schon annotiert) ans Ende
            pool = pool.merge(prio, on="image_id", how="left")
            if order.startswith("Modell"):
                pool = pool.assign(_k=-pool.slip.abs().fillna(-1)).sort_values(["_k", "order"]).drop(columns="_k")
            pool = pool.reset_index(drop=True)
        if pass_no == 2:
            pool = index[index.image_id.isin(done1)].head(REPEAT_N)
        ids = pool.image_id.tolist()
        mine = ann[ann["pass"] == pass_no]
        done = set(mine.image_id)
        st.metric("bearbeitet", f"{len(done & set(ids))} / {len(ids)}")
        st.dataframe(pool[pool.image_id.isin(done)].img_type.value_counts().rename("bearbeitet"))
        st.divider()
        token = st.text_input("Bearer Token (URLs auffrischen)", type="password")
        if st.button("URLs auffrischen") and token.strip():
            with open(f"{DATA_DIR}/img_urls.pkl", "wb") as fh:
                pickle.dump(fetch_img_urls(token.strip()), fh)
            load_data.clear(); fetch_image.clear(); st.rerun()
    if not ids:
        st.info("Keine Bilder in dieser Auswahl.")
        return

    if ss.get("pool_key") != (pass_no, tuple(types), order):
        ss.pool_key, ss.pos = (pass_no, tuple(types), order), next_open(ids, done, 0) or 0
    ss.pos = min(ss.pos, len(ids) - 1)

    def go(pos):
        ss.pos = pos % len(ids)

    def go_open(start):
        nxt = next_open(ids, done, start)
        if nxt is not None:
            ss.pos = nxt

    status = lambda i: "⏭" if (mine[(mine.image_id == i)].level == "SKIP").any() else "✓" if i in done else "·"
    n1, n2, n3, n4 = st.columns([1, 4, 1, 3])
    n1.button("◀", width="stretch", on_click=go, args=(ss.pos - 1,))
    ss.jump = ids[ss.pos]
    n2.selectbox("Bild", ids, key="jump", label_visibility="collapsed",
                 format_func=lambda i: f"{status(i)} {ids.index(i) + 1}/{len(ids)} · ID {i}",
                 on_change=lambda: go(ids.index(ss.jump)))
    n3.button("▶", width="stretch", on_click=go, args=(ss.pos + 1,))
    n4.button("Nächstes zu bearbeitendes Bild ⏩", type="primary", width="stretch",
              on_click=go_open, args=(ss.pos + 1,), disabled=len(done & set(ids)) == len(ids))

    row = pool.iloc[ss.pos]
    levels = [lab for lab in row.labels if SPINE_LABELS.index(lab) >= SPINE_LABELS.index(ANNOTATE_FROM)]
    targets = [(lab, c) for lab in levels for c in ("PS", "PI")]
    img = fetch_image(img_urls.get(row.image_id, ""))
    if img is None:
        st.error("Bild nicht ladbar – URLs abgelaufen? Links Token eingeben und auffrischen.")
        return
    scale = img.height / STAGE_H
    nodes_px = np.array(row.nodes, float) * scale

    saved = mine[mine.image_id == row.image_id]
    instr_key = f"instr_{row.image_id}_{pass_no}"
    if ss.get("img_key") != (row.image_id, pass_no):           # Bild gewechselt: gespeicherten Stand laden
        ss.img_key, ss.pts, ss.k, ss.nclick = (row.image_id, pass_no), {}, 0, 0
        for r in saved[saved.level.isin(levels)].itertuples():
            for c in ("PS", "PI"):
                x, y = getattr(r, f"{c.lower()}_x"), getattr(r, f"{c.lower()}_y")
                ss.pts[(r.level, c)] = (x * scale, y * scale) if pd.notna(x) else None
        if ss.pts:
            ss.k = len(targets)
        ss[instr_key] = saved.loc[saved.instrumented.astype(str).str.lower() == "true", "level"].tolist()

    state = ("übersprungen" if (saved.level == "SKIP").any() else
             f"bearbeitet {saved.timestamp.iloc[0]}" if len(saved) else "offen")
    st.caption(f"ID {row.image_id} · {row.img_type} · {row.patient_id} · {row.labels[0]}–{row.labels[-1]} · "
               f"Durchgang {pass_no} · {state}" +
               (f" · Modell (% WKD): {model_slips(row)}" if pd.notna(getattr(row, "slip", np.nan)) else ""))

    def advance():                                              # zur nächsten noch fehlenden Ecke
        ss.nclick += 1                                          # neuer Klick-Widget-Key, alter Klick verfällt
        ss.k = next((j for j, t in enumerate(targets) if t not in ss.pts), len(targets))

    def after_save():
        ss.pop("img_key")
        go_open(ss.pos + 1)

    col_over, col_zoom = st.columns([1, 2])
    with col_over:
        s_med = np.median(np.linalg.norm(np.diff(nodes_px, axis=0), axis=1))
        box = (nodes_px[:, 0].min() - 1.5 * s_med, nodes_px[:, 1].min() - s_med,
               nodes_px[:, 0].max() + 1.5 * s_med, nodes_px[:, 1].max() + s_med)
        over = img.crop(tuple(int(b) for b in box)).convert("RGB")
        st.image(draw_points(over, ss.pts, box, 1.0), width="stretch")
        instrumented = st.multiselect("Instrumentiert (Schraube/Cage im Wirbel)", levels, key=instr_key)

    with col_zoom:
        if ss.k < len(targets):
            lab, corner = targets[ss.k]
            i = row.labels.index(lab)
            s = spacing(nodes_px, i)
            cx, cy = nodes_px[i]
            zbox = (cx - 0.8 * s, cy - 1.2 * s, cx + 1.6 * s, cy + 1.2 * s)   # Hinterkante ~0,6–0,8 s posterior
            zbox = tuple(int(round(b)) for b in zbox)
            zw, zh = zbox[2] - zbox[0], zbox[3] - zbox[1]
            factor = ZOOM_PX / zw
            zoom = ImageOps.autocontrast(img.crop(zbox), cutoff=1)
            zoom = zoom.resize((ZOOM_PX, ZOOM_PX), Image.LANCZOS).convert("RGB")
            draw_points(zoom, {k: p for k, p in ss.pts.items() if k != (lab, corner)}, zbox, factor)
            ImageDraw.Draw(zoom).ellipse([(cx - zbox[0]) * factor - 4, (cy - zbox[1]) * factor - 4,
                                          (cx - zbox[0]) * factor + 4, (cy - zbox[1]) * factor + 4],
                                         fill=(0, 170, 255))
            st.markdown(f"### {lab}: {CORNER_TXT[corner]}  ·  {ss.k + 1}/{len(targets)}")
            click = streamlit_image_coordinates(zoom, key=f"z_{row.image_id}_{pass_no}_{ss.nclick}")
            if click:
                ss.pts[(lab, corner)] = (zbox[0] + (click["x"] + 0.5) * zw / click["width"] - 0.5,
                                         zbox[1] + (click["y"] + 0.5) * zh / click["height"] - 0.5)
                advance()
                st.rerun()
            b1, b2 = st.columns(2)
            if b1.button("nicht beurteilbar", width="stretch"):
                ss.pts[(lab, corner)] = None
                advance()
                st.rerun()
        else:
            st.success("Alle Ecken gesetzt." + (" (gespeicherter Stand)" if len(saved) else ""))
            b1, b2 = st.columns(2)
            if b1.button("Speichern ▶", type="primary", width="stretch"):
                pts_canvas = {k: (p[0] / scale, p[1] / scale) if p else None for k, p in ss.pts.items()}
                save_image(row, pass_no, pts_canvas, instrumented, img)
                done.add(row.image_id)
                after_save()
                st.rerun()
        if b2.button("◀ Ecke zurück", width="stretch", disabled=ss.k == 0):
            ss.k -= 1
            ss.pts.pop(targets[ss.k], None)
            st.rerun()
        if ss.pts:
            redo = st.selectbox("Ecke neu setzen", [None] + targets, key=f"redo_{row.image_id}_{pass_no}_{ss.k}",
                                format_func=lambda t: "—" if t is None else f"{t[0]} {t[1]}")
            if redo:
                ss.k = targets.index(redo)
                st.rerun()
        if st.button("Bild ungeeignet – überspringen"):
            save_image(row, pass_no, {}, [], None, skip=True)
            done.add(row.image_id)
            after_save()
            st.rerun()


if __name__ == "__main__":
    main()
