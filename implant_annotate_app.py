"""
Implant Annotate App — Streamlit

Start:
    cd /Users/peterbernstein/Python/XrayStats
    XRAY_TOKEN=... /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run implant_annotate_app.py

Ground Truth für ../Xray-Model/train_implants.ipynb. Je Implantat Klasse wählen, dann zwei Klicks:
  - Pedikelschraube (lumbal, S1/S2): Kopf → Spitze
  - Iliumschraube (inkl. S2-Alar-Iliakal): Kopf → Spitze
  - Cage: die beiden Enden entlang der langen Achse (ap: links–rechts, lat: vorn–hinten)
Jede sichtbare Schraube einzeln; seitlich genau deckungsgleiche Schrauben beider Seiten nur einmal.
Seite (bds./re./li.) und Höhe werden NICHT gelabelt — sie folgen aus der Lage (Mittellinie bzw.
Wirbel-Nodes) und sind damit Nachverarbeitung, kein Modellausgang.
Jedes fertige Implantat (zweiter Klick) und jedes Löschen wird sofort gespeichert; Blättern verliert nichts.
Bilder ohne Implantat mit "Keine Implantate" speichern — das Modell braucht Negativbeispiele.
Reihenfolge: in data/corner_annotations.csv als instrumentiert markierte Bilder zuerst, dann per Hash.
Speichert nach data/implant_annotations.csv in Canvas-Koordinaten (Bildhöhe 600, wie die Nodes),
eine Zeile je Implantat; cls = NONE (keine Implantate) / SKIP (ungeeignet) für das ganze Bild.
"""

import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

DATA_DIR = "data"
OUT_FILE = os.environ.get("IMPLANT_OUT", f"{DATA_DIR}/implant_annotations.csv")   # IMPLANT_OUT: Testkopie
META_FILE = os.environ.get("IMPLANT_META", f"{DATA_DIR}/implant_imagedata.json")
CORNER_FILE = f"{DATA_DIR}/corner_annotations.csv"
API_URL = "https://xray-api.renecol.org"
STAGE_H = 600
IMG_TYPES = ["Xray-LWS-ap", "Xray-LWS-lat", "Xray-LWS-incl", "Xray-LWS-recl", "Xray-WS-ap", "Xray-WS-lat"]
CLASSES = {"pedicle": ("Pedikelschraube", (255, 60, 60)),
           "iliac": ("Iliumschraube", (255, 200, 0)),
           "cage": ("Cage", (0, 200, 255))}
TARGET_PER_CLASS = 150     # grobe Zielmarke Instanzen je Klasse (Train+Test), s. Notebook
COLS = ["image_id", "patient_id", "img_type", "cls", "x1", "y1", "x2", "y2", "timestamp"]


def fetch_meta(token: str):
    r = requests.get(f"{API_URL}/api/imagedata/", headers={"Authorization": f"Bearer {token}"},
                     params={"detailed": True, "showall": True}, timeout=300)
    r.raise_for_status()
    meta = [{k: d.get(k) for k in ("id", "img", "img_type", "patient_id")} for d in r.json()]
    with open(META_FILE, "w") as fh:
        json.dump(meta, fh)


@st.cache_data(show_spinner="Bildliste …")
def load_index(mtime: float) -> pd.DataFrame:
    idx = pd.DataFrame(json.load(open(META_FILE))).rename(columns={"id": "image_id"})
    idx["order"] = idx.image_id.map(lambda i: hashlib.md5(f"implant:{i}".encode()).hexdigest())
    known = set()
    if os.path.exists(CORNER_FILE):                               # bekannte instrumentierte Bilder zuerst
        c = pd.read_csv(CORNER_FILE)
        known = set(c.loc[c.instrumented.astype(str).str.lower() == "true", "image_id"])
    idx["known"] = idx.image_id.isin(known)
    return idx.sort_values(["known", "order"], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fetch_image(url: str):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("L")
    except Exception:
        return None


def _span(profile, fill=0.05, run=0.02, margin=0.01):
    """Längster Inhaltsbereich entlang einer Achse (wie content_box in train.ipynb)."""
    n, idx = len(profile), np.flatnonzero(profile > fill)
    runs = [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(r) >= run * n] if idx.size else []
    if not runs:
        return 0, n
    m = round(margin * n)
    return max(0, int(runs[0][0]) - m), min(n, int(runs[-1][-1]) + 1 + m)


def content_box(img):
    a = np.asarray(img) > 20
    (x0, x1), (y0, y1) = _span(a.mean(0)), _span(a.mean(1))
    return x0, y0, x1, y1


def load_annotations() -> pd.DataFrame:
    return pd.read_csv(OUT_FILE) if os.path.exists(OUT_FILE) else pd.DataFrame(columns=COLS)


def save_image(row, implants: list, scale: float, cls_all=None):
    ann = load_annotations()
    ann = ann[ann.image_id != row.image_id]
    base = {"image_id": row.image_id, "patient_id": row.patient_id, "img_type": row.img_type,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if cls_all:
        new = [{**base, "cls": cls_all}]
    elif not implants:                                            # alle Implantate gelöscht → Bild wieder offen
        new = []
    else:
        new = [{**base, "cls": im["cls"], "x1": im["p1"][0] / scale, "y1": im["p1"][1] / scale,
                "x2": im["p2"][0] / scale, "y2": im["p2"][1] / scale} for im in implants]
    pd.concat([ann, pd.DataFrame(new)], ignore_index=True).reindex(columns=COLS).to_csv(OUT_FILE, index=False)


def draw(view, implants, pending, box, f):
    d = ImageDraw.Draw(view)
    to = lambda p: ((p[0] - box[0]) * f, (p[1] - box[1]) * f)
    for n, im in enumerate(implants, 1):
        col = CLASSES[im["cls"]][1]
        a, b = to(im["p1"]), to(im["p2"])
        d.line([a, b], fill=col, width=3)
        d.ellipse([a[0] - 4, a[1] - 4, a[0] + 4, a[1] + 4], fill=col)
        d.text((b[0] + 5, b[1] - 5), str(n), fill=col)
    if pending:
        x, y = to(pending)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(0, 255, 0), width=2)
    return view


def next_open(ids, done, start):
    for j in range(len(ids)):
        if ids[(start + j) % len(ids)] not in done:
            return (start + j) % len(ids)
    return None


def main():
    st.set_page_config(page_title="Implantate annotieren", layout="wide", page_icon="🔩")
    ss = st.session_state
    with st.sidebar:
        token = st.text_input("Bearer Token", value=os.environ.get("XRAY_TOKEN", ""), type="password")
        stale = not os.path.exists(META_FILE) or time.time() - os.path.getmtime(META_FILE) > 20 * 3600
        if st.button("Bildliste/URLs auffrischen" + (" (abgelaufen)" if stale else "")) and token.strip():
            fetch_meta(token.strip())
            fetch_image.clear()
            st.rerun()
    if not os.path.exists(META_FILE):
        st.info("Noch keine Bildliste — links Token eingeben und auffrischen.")
        return
    index = load_index(os.path.getmtime(META_FILE))
    ann = load_annotations()
    done = set(ann.image_id)

    with st.sidebar:
        types = st.multiselect("Bildtypen", IMG_TYPES, default=IMG_TYPES)
        display_h = st.slider("Anzeigehöhe px", 500, 1600, 900, 50)
        band = st.slider("Ausschnitt oben–unten %", 0, 100, (0, 100), 5, help="Zoom für Ganzwirbelsäulen-Bilder")
        pool = index[index.img_type.isin(types)].reset_index(drop=True)
        ids = pool.image_id.tolist()
        lab = ann[ann.image_id.isin(ids)]
        st.metric("bearbeitet", f"{len(done & set(ids))} / {len(ids)}")
        imgs = lab.drop_duplicates("image_id")
        st.caption(f"mit Implantat {(~imgs.cls.isin(['NONE', 'SKIP'])).sum()} · ohne {(imgs.cls == 'NONE').sum()} · "
                   f"übersprungen {(imgs.cls == 'SKIP').sum()}")
        cnt = lab.cls.value_counts()
        st.dataframe(pd.DataFrame({"Instanzen": [int(cnt.get(c, 0)) for c in CLASSES],
                                   "Bilder": [lab[lab.cls == c].image_id.nunique() for c in CLASSES],
                                   "Ziel": TARGET_PER_CLASS}, index=[v[0] for v in CLASSES.values()]))
    if not ids:
        st.info("Keine Bilder in dieser Auswahl.")
        return

    if ss.get("pool_key") != tuple(types):
        ss.pool_key, ss.pos = tuple(types), next_open(ids, done, 0) or 0
    ss.pos = min(ss.pos, len(ids) - 1)

    def go(pos):
        ss.pos = pos % len(ids)

    def go_open(start):
        nxt = next_open(ids, done, start)
        if nxt is not None:
            ss.pos = nxt

    status = lambda i: "✓" if i in done else "·"
    n1, n2, n3, n4 = st.columns([1, 4, 1, 3])
    n1.button("◀", width="stretch", on_click=go, args=(ss.pos - 1,))
    ss.jump = ids[ss.pos]
    n2.selectbox("Bild", ids, key="jump", label_visibility="collapsed",
                 format_func=lambda i: f"{status(i)} {ids.index(i) + 1}/{len(ids)} · ID {i}",
                 on_change=lambda: go(ids.index(ss.jump)))
    n3.button("▶", width="stretch", on_click=go, args=(ss.pos + 1,))
    n4.button("Nächstes offenes Bild ⏩", type="primary", width="stretch", on_click=go_open, args=(ss.pos + 1,),
              disabled=len(done & set(ids)) == len(ids))

    row = pool.iloc[ss.pos]
    img = fetch_image(row.img or "")
    if img is None:
        st.error("Bild nicht ladbar – URLs abgelaufen? Links auffrischen.")
        return
    scale = img.height / STAGE_H
    saved = ann[ann.image_id == row.image_id]
    if ss.get("img_key") != row.image_id:                       # Bild gewechselt: gespeicherten Stand laden
        ss.img_key, ss.pending, ss.last_t = row.image_id, None, None
        ss.implants = [{"cls": r.cls, "p1": (r.x1 * scale, r.y1 * scale), "p2": (r.x2 * scale, r.y2 * scale)}
                       for r in saved.itertuples() if r.cls in CLASSES]
    state = (f"gespeichert {saved.timestamp.iloc[0]}" + (" · keine Implantate" if (saved.cls == "NONE").any() else
             " · übersprungen" if (saved.cls == "SKIP").any() else "")) if len(saved) else "offen"
    st.caption(f"ID {row.image_id} · {row.img_type} · {row.patient_id} · {state}"
               + (" · laut Ecken-Annotation instrumentiert" if row.known else ""))

    cls = st.radio("Klasse", list(CLASSES), horizontal=True, format_func=lambda c: CLASSES[c][0], key="cls")
    col_img, col_ctl = st.columns([3, 1])
    with col_img:
        x0, y0, x1, y1 = content_box(img)
        h = y1 - y0
        box = (x0, y0 + round(band[0] / 100 * h), x1, y0 + round(band[1] / 100 * h))
        if box[3] - box[1] < 10:
            box = (x0, y0, x1, y1)
        f = display_h / (box[3] - box[1])
        view = img.crop(box).resize((max(1, round((box[2] - box[0]) * f)), display_h), Image.LANCZOS).convert("RGB")
        hint = ("Endpunkt" if ss.pending else "Startpunkt") + (" (Spitze)" if ss.pending and cls != "cage" else
                                                               " (Kopf)" if cls != "cage" else "")
        st.markdown(f"**{CLASSES[cls][0]}: {hint} klicken**")
        click = streamlit_image_coordinates(draw(view, ss.implants, ss.pending, box, f),
                                            key=f"c_{row.image_id}")
        if click and click.get("unix_time") != ss.last_t:     # fester Key: kein Remount je Klick, neue Klicks per Zeitstempel
            ss.last_t = click.get("unix_time")
            p = (box[0] + (click["x"] + 0.5) * (box[2] - box[0]) / click["width"] - 0.5,
                 box[1] + (click["y"] + 0.5) * (box[3] - box[1]) / click["height"] - 0.5)
            if ss.pending is None:
                ss.pending = p
            else:
                ss.implants.append({"cls": cls, "p1": ss.pending, "p2": p})
                ss.pending = None
                save_image(row, ss.implants, scale)             # Autosave: Blättern verliert nichts
            st.rerun()

    with col_ctl:
        for n, im in enumerate(ss.implants, 1):
            st.write(f"{n}. {CLASSES[im['cls']][0]}")
        if st.button("Letztes / Startpunkt löschen", width="stretch", disabled=not (ss.implants or ss.pending)):
            if ss.pending:
                ss.pending = None
            else:
                ss.implants.pop()
                save_image(row, ss.implants, scale)
            st.rerun()
        drop = st.selectbox("Implantat löschen", [None] + list(range(1, len(ss.implants) + 1)),
                            key=f"del_{row.image_id}_{len(ss.implants)}", format_func=lambda n: "—" if n is None else str(n))
        if drop:
            ss.implants.pop(drop - 1)
            save_image(row, ss.implants, scale)
            st.rerun()
        st.divider()

        def finish(cls_all=None):
            save_image(row, ss.implants, scale, cls_all)
            done.add(row.image_id)
            ss.pop("img_key")
            go_open(ss.pos + 1)

        if ss.implants:
            st.caption(f"{len(ss.implants)} Implantat(e) automatisch gespeichert")
            st.button("Fertig – nächstes Bild ▶", type="primary", width="stretch", on_click=finish,
                      disabled=ss.pending is not None)
        else:
            st.button("Keine Implantate ▶", type="primary", width="stretch", on_click=finish, args=("NONE",))
        st.button("Bild ungeeignet – überspringen", width="stretch", on_click=finish, args=("SKIP",))


if __name__ == "__main__":
    main()
