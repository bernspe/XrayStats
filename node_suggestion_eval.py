"""NodeSuggestion-Auswertung: Modell-Inferenz (POST /api/suggestnodes/) vs. bestehende Messungen.

Interaktives Skript im "percent"-Format (# %%): Zelle für Zelle in PyCharm / VS Code / Spyder / Jupytext
ausführen, oder gesamt mit  `python -i node_suggestion_eval.py`  (danach sind alle Tabellen im REPL).

Ablauf
  1. UUIDs aus data/image_group_uuids.csv
  2. ImageData inkl. MeasureData je UUID von der API   -> data/node_suggestion_imagedata.json
  3. NodeSuggestion je Bild (Nodes, Endplatten, Cobb)  -> data/node_suggestions_raw.json (wiederaufnehmbar)
  4. Modell-Nodes -> 600px-Stage -> getReconstructedCobbAngles (regional + segmental), gleiche Pipeline wie Rater
  5. Fehler: Rater vs. Modell und Rater vs. Rater (MAE, RMSE, Bias, LoA, ICC) je img_type / Region / Segment / Wirbel
  6. Ergebnis -> data/node_suggestion_eval.pkl   (pd.read_pickle -> dict mit DataFrames)

Token: Umgebungsvariable XRAY_TOKEN oder Eingabeaufforderung (xray-annotator -> Account info -> Copy token).
Nur nötig, solange Caches fehlen bzw. REFRESH_* gesetzt ist.
Selbsttest ohne API:  python node_suggestion_eval.py --check
"""
import getpass
import io
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from annotation_analysis import (
    aggregate_angles_by_region, build_angles_df, build_df4d, build_nodes_df,
    compute_icc, correct_measurement_label_shift, getReconstructedCobbAngles,
)
from bspline_python import spine_labels

# %% Konfiguration
API = "https://xray-api.renecol.org"
DATA = Path(__file__).parent / "data"
UUID_CSV = DATA / "image_group_uuids.csv"
IMAGEDATA_CACHE = DATA / "node_suggestion_imagedata.json"   # nur die UUIDs der CSV
SUGG_CACHE = DATA / "node_suggestions_raw.json"             # {uuid: {resp, w, h}}
LABEL_ERRORS = DATA / "label_errors.json"                   # Shift-Korrektur der Rater-Labels (per m_uuid)
RESULT = DATA / "node_suggestion_eval.pkl"
FULL_DUMP = DATA / "image_groups_cache.json"                # Offline-Fallback (komplette API-Antwort)

REFRESH_IMAGEDATA = True    # True: MeasureData neu von der API holen
REFRESH_SUGGESTIONS = False  # True: alle NodeSuggestions neu rechnen lassen (ca. 1.5 s/Bild, erster Aufruf ~30 s)
LABEL_CORRECTION = True
STAGE_H = 600                # Annotator-Stage; Rater-Nodes liegen darin, Modell-Nodes in Original-Pixeln
MODEL = "NodeSuggestion"     # rater_name der Modellzeilen

# Rater-Typ -> vergleichbare Modell-Typen. endplate_segmental = Cobb aus den vorhergesagten Endplatten.
COMPARE = {
    "Cobb-regional": ["reconstructed_regional"],
    "Spline": ["reconstructed_regional"],
    "reconstructed_regional": ["reconstructed_regional"],
    "Cobb-segmental": ["reconstructed_segmental", "endplate_segmental"],
    "reconstructed_segmental": ["reconstructed_segmental", "endplate_segmental"],
}
_TOKEN = None


# %% API
def get_token():
    global _TOKEN
    _TOKEN = _TOKEN or os.environ.get("XRAY_TOKEN") or getpass.getpass("xray-annotator Token: ").strip()
    return _TOKEN


def auth():
    return {"Authorization": f"Bearer {get_token()}"}


def load_imagedata(uuids, refresh=False, force_live=False):
    """ImageData (inkl. measure_dataset) der UUIDs. Live von der API, sonst aus dem Cache."""
    if refresh or force_live or not IMAGEDATA_CACHE.exists():
        r = requests.get(f"{API}/api/imagedata/", params={"detailed": True, "complete": True},
                         headers=auth(), timeout=600)
        r.raise_for_status()
        data = [i for i in r.json() if i["uuid"] in uuids]
        IMAGEDATA_CACHE.write_text(json.dumps(data))
        return data, True
    return json.loads(IMAGEDATA_CACHE.read_text()), False


def image_size(url):
    # signierte GCS-URL: bewusst OHNE Bearer-Token
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).size


def fetch_suggestions(images, raw):
    """Fehlende NodeSuggestions nachholen; speichert alle 10 Bilder (Abbruch-sicher). Fehler werden nicht gecached."""
    todo = [i for i in images if i["uuid"] not in raw]
    errors = {}
    for k, im in enumerate(todo, 1):
        u = im["uuid"]
        try:
            r = requests.post(f"{API}/api/suggestnodes/", json={"image": u, "cobb": True},
                              headers=auth(), timeout=300)
            r.raise_for_status()
            w, h = image_size(im["img"])
            raw[u] = {"resp": r.json(), "w": w, "h": h}
        except Exception as e:  # ponytail: ein Bild darf den Lauf nicht abbrechen; Fehler werden unten gelistet
            errors[u] = f"{type(e).__name__}: {e}"[:200]
        print(f"\r{k}/{len(todo)} Suggestions", end="")
        if k % 10 == 0:
            SUGG_CACHE.write_text(json.dumps(raw))
    if todo:
        SUGG_CACHE.write_text(json.dumps(raw))
        print()
    return errors


# %% Modell -> xray_measurement_data
def model_xmd(resp, h):
    """Suggestion -> xmd-Dict wie bei Ratern. Nodes: Original-Pixel -> 600px-Stage.

    Wie im Frontend (suggestNodes): nur Wirbel-Labels, y strikt abwärts. Zusätzlich längster lückenloser
    Label-Lauf, weil Nodes positional gegen vertebra_range gelesen werden. None bei < 3 Nodes.
    """
    k = STAGE_H / h
    pts = sorted((spine_labels.index(n["label"]), n["x"] * k, n["y"] * k)
                 for n in resp["nodes"] if n["label"] in spine_labels)
    kept = []
    for p in pts:
        if not kept or p[2] > kept[-1][2]:
            kept.append(p)
    best, cur = [], []
    for p in kept:
        cur = cur + [p] if cur and p[0] == cur[-1][0] + 1 else [p]
        if len(cur) > len(best):
            best = cur
    if len(best) < 3:
        return None
    angles = [{"cobbType": "endplate_segmental",
               "cobbAngle": {"angle": a["angle"], "vertebra_upper": a["vertebra_upper"],
                             "vertebra_lower": a["vertebra_lower"]}}
              for a in resp.get("cobb_angles") or []]
    return {"angles": angles,
            "stateObject": {"nodes": [{"x": x, "y": y} for _, x, y in best],
                            "vertebra_range": [best[0][0], best[-1][0]]}}


# %% Tabellen
def build_sources(images, raw):
    """Ein df4d mit Rater-Messungen + je Bild einer Modellzeile (rater_name == MODEL)."""
    df4d = build_df4d(pd.DataFrame(images))
    n = df4d.groupby(["id", "rater_name"]).cumcount()
    df4d["rater_name"] = df4d["rater_name"] + np.where(n > 0, "#" + (n + 1).astype(str), "")
    n_corr = 0
    if LABEL_CORRECTION and LABEL_ERRORS.exists():
        shift = {e["m_uuid"]: int(e["shift"]) for e in json.loads(LABEL_ERRORS.read_text()).values()
                 if e["error"] == "shift"}
        fixed = []
        for x, m in zip(df4d["xray_measurement_data"], df4d["m_uuid"]):
            hit = m in shift and isinstance(x, dict)
            n_corr += hit
            fixed.append(correct_measurement_label_shift(x, shift[m]) if hit else x)
        df4d["xray_measurement_data"] = fixed
    meta = {i["uuid"]: i for i in images}
    rows, skipped = [], []
    for u, s in raw.items():
        if u not in meta:
            continue
        xmd = model_xmd(s["resp"], s["h"])
        if xmd is None:
            skipped.append(u)
            continue
        rows.append({"id": meta[u]["id"], "uuid": u, "img_type": meta[u]["img_type"], "rater_name": MODEL,
                     "rater_id": -1, "xray_measurement_data": xmd})
    return pd.concat([df4d, pd.DataFrame(rows)], ignore_index=True), n_corr, skipped


def build_units(all4d):
    """Ein Wert je (Bild, Rater, cobbType, Einheit): regional -> max je Region, segmental -> Mittel je Segment."""
    a = build_angles_df(all4d, include_reconstructed=True).dropna(subset=["angle"])
    a.loc[(a.rater_name == MODEL) & (a.cobbType == "Cobb-segmental"), "cobbType"] = "endplate_segmental"
    a["source"] = np.where(a.rater_name == MODEL, "model", "rater")
    reg = a[a.cobbType.str.contains("regional") | (a.cobbType == "Spline")]
    seg = a[~a.index.isin(reg.index)]
    keys = ["image_id", "img_type", "rater_name", "source", "cobbType"]
    reg = (aggregate_angles_by_region(reg)
           .merge(reg[keys].drop_duplicates(), on=["image_id", "img_type", "rater_name", "cobbType"])
           .rename(columns={"region": "unit"}).assign(level="regional"))
    seg = (seg.assign(unit=seg.vertebra_upper + "-" + seg.vertebra_lower).groupby(keys + ["unit"], as_index=False)
           ["angle"].mean().assign(level="segmental"))
    cols = keys + ["unit", "level", "angle"]
    return pd.concat([reg[cols], seg[cols]], ignore_index=True)


def pair_errors(units, key_cols=("image_id", "img_type", "unit", "level")):
    """Alle Paarfehler. kind: rater_vs_model (b - a = Modell - Rater) / rater_vs_rater (Vorzeichen ~ Namensreihenfolge)."""
    key_cols = list(key_cols)
    rat, mod = units[units.source == "rater"], units[units.source == "model"]
    out = []
    for ref, cmps in COMPARE.items():
        r = rat[rat.cobbType == ref]
        for cmp_ in cmps:
            x = r.merge(mod[mod.cobbType == cmp_], on=key_cols, suffixes=("_a", "_b"))
            out.append(x.assign(kind="rater_vs_model", ref=ref, cmp=cmp_))
        x = r.merge(r, on=key_cols, suffixes=("_a", "_b"))
        out.append(x[x.rater_name_a < x.rater_name_b].assign(kind="rater_vs_rater", ref=ref, cmp=ref))
    e = pd.concat(out, ignore_index=True)
    e["a"], e["b"] = e["angle_a"], e["angle_b"]
    e["err"] = e["b"] - e["a"]
    e["abs_err"] = e["err"].abs()
    return e[key_cols + ["kind", "ref", "cmp", "rater_name_a", "rater_name_b", "a", "b", "err", "abs_err"]]


def with_all(df, col):
    """Hängt eine Kopie mit col == 'ALL' an, damit Summaries gepoolte Zeilen enthalten."""
    return pd.concat([df, df.assign(**{col: "ALL"})], ignore_index=True)


def summarize(err, by):
    g = err.groupby(by)
    s = g.agg(n=("abs_err", "size"), n_images=("image_id", "nunique"), MAE=("abs_err", "mean"),
              median_AE=("abs_err", "median"), p90_AE=("abs_err", lambda v: v.quantile(.9)),
              RMSE=("err", lambda v: np.sqrt((v ** 2).mean())), bias=("err", "mean"), sd_diff=("err", "std"),
              within_5deg=("abs_err", lambda v: (v <= 5).mean() * 100))
    s["LoA_low"], s["LoA_high"] = s.bias - 1.96 * s.sd_diff, s.bias + 1.96 * s.sd_diff
    return s.round(2)


def icc_table(units, min_images=5):
    """ICC(2,1) je img_type x Rater-Typ x Modell-Typ x Einheit: nur Rater vs. Rater + Modell als zusätzlicher Rater.

    Beide ICCs auf denselben Zielen (Bild x Einheit), die das Modell abdeckt.
    """
    u = with_all(units, "img_type").assign(target=lambda d: d.image_id.astype(str) + "|" + d.unit)
    rows = []
    for img_type, ui in u.groupby("img_type"):
        for ref, cmps in COMPARE.items():
            r_all = ui[(ui.source == "rater") & (ui.cobbType == ref)]
            for cmp_ in cmps:
                m_all = ui[(ui.source == "model") & (ui.cobbType == cmp_)]
                for unit in ["ALL", *sorted(set(r_all.unit) & set(m_all.unit))]:
                    r = r_all if unit == "ALL" else r_all[r_all.unit == unit]
                    m = m_all if unit == "ALL" else m_all[m_all.unit == unit]
                    r = r[r.target.isin(m.target)]
                    ir = compute_icc(r, "target", "rater_name", "angle", min_images)
                    im = compute_icc(pd.concat([r, m]), "target", "rater_name", "angle", min_images)
                    rows.append({"img_type": img_type, "ref": ref, "cmp": cmp_, "unit": unit,
                                 "n_targets": ir["n"], "ICC_raters": ir["ICC"],
                                 "ICC_raters_lo": ir["CI95_lower"], "ICC_raters_hi": ir["CI95_upper"],
                                 "ICC_with_model": im["ICC"], "ICC_with_model_lo": im["CI95_lower"],
                                 "ICC_with_model_hi": im["CI95_upper"]})
    return pd.DataFrame(rows).set_index(["img_type", "ref", "cmp", "unit"])


def node_errors(all4d):
    """Nodes (Stage-Pixel) je Wirbel: Rater vs. Modell und Rater vs. Rater."""
    n = build_nodes_df(all4d.drop_duplicates(["id", "rater_name"]))
    n = n.merge(all4d[["id", "img_type"]].drop_duplicates("id"), left_on="image_id", right_on="id").drop(columns="id")
    n["source"] = np.where(n.rater_name == MODEL, "model", "rater")
    r, m = n[n.source == "rater"], n[n.source == "model"]
    key = ["image_id", "img_type", "vertebra_label"]
    x = r.merge(m, on=key, suffixes=("_a", "_b")).assign(kind="rater_vs_model")
    y = r.merge(r, on=key, suffixes=("_a", "_b"))
    e = pd.concat([x, y[y.rater_name_a < y.rater_name_b].assign(kind="rater_vs_rater")], ignore_index=True)
    e["dx"], e["dy"] = (e.x_b - e.x_a).abs(), (e.y_b - e.y_a).abs()
    e["dist"] = np.hypot(e.dx, e.dy)
    return e[key + ["kind", "rater_name_a", "rater_name_b", "dx", "dy", "dist"]]


def node_summary(ne):
    ne = with_all(with_all(ne, "img_type"), "vertebra_label")
    return ne.groupby(["kind", "img_type", "vertebra_label"]).agg(
        n=("dist", "size"), n_images=("image_id", "nunique"), MAE_x=("dx", "mean"), MAE_y=("dy", "mean"),
        mean_dist=("dist", "mean"), median_dist=("dist", "median"),
        p90_dist=("dist", lambda v: v.quantile(.9))).round(2)


# %% Selbsttest (ohne API): "perfektes Modell" aus Rater-Nodes muss dieselben Winkel liefern
def selfcheck():
    df4d = pd.read_pickle(DATA / "df4d_spine_measurements.pkl")
    h = 1800
    for xmd in df4d["xray_measurement_data"]:
        so = xmd.get("stateObject", {}) if isinstance(xmd, dict) else {}
        nodes, vr = so.get("nodes") or [], so.get("vertebra_range") or [0, 0]
        if len(nodes) < 5 or vr[1] - vr[0] + 1 != len(nodes):
            continue
        resp = {"nodes": [{"label": spine_labels[vr[0] + i], "x": n["x"] * h / STAGE_H, "y": n["y"] * h / STAGE_H}
                          for i, n in enumerate(nodes)], "endplates": [], "cobb_angles": []}
        m = model_xmd(resp, h)
        if m is None or len(m["stateObject"]["nodes"]) != len(nodes):
            continue  # nicht streng abwärts -> vom Frontend-Filter gekürzt, anderes Beispiel nehmen
        got, want = getReconstructedCobbAngles(m), getReconstructedCobbAngles(xmd)
        assert want and got == want, (got, want)
        assert m["stateObject"]["vertebra_range"] == vr
        print("selfcheck ok:", len(want), "Winkel identisch")
        return
    raise AssertionError("kein geeignetes Beispiel in df4d_spine_measurements.pkl")


if "--check" in sys.argv:
    selfcheck()
    sys.exit()

# %% 1. UUIDs + ImageData/MeasureData
uuid_df = pd.read_csv(UUID_CSV)
uuids = set(uuid_df.image_uuid)
raw = {} if REFRESH_SUGGESTIONS or not SUGG_CACHE.exists() else json.loads(SUGG_CACHE.read_text())
need_sugg = [u for u in uuids if u not in raw]
# frische signierte Bild-URLs nur nötig, wenn noch Suggestions (inkl. Bildgröße) fehlen
images, live = load_imagedata(uuids, REFRESH_IMAGEDATA, force_live=bool(need_sugg))
missing = uuids - {i["uuid"] for i in images}
n_meas = sum(len(i["measure_dataset"]) for i in images)
print(f"{len(uuids)} UUIDs in CSV, {len(images)} ImageData ({'live' if live else 'Cache'}), {n_meas} Messungen"
      + (f", {len(missing)} UUIDs NICHT in der API-Antwort" if missing else ""))

# %% 2. NodeSuggestion-Inferenz
sugg_errors = fetch_suggestions(images, raw)
print(f"{len(raw)} Suggestions im Cache, {len(sugg_errors)} Fehler")
for u, e in sugg_errors.items():
    print(" ", u, e)

# %% 3. Tabellen + Statistik
all4d, n_label_corrected, no_model_nodes = build_sources(images, raw)
units = build_units(all4d)
errors = pair_errors(units)
uuid_of = {i["id"]: i["uuid"] for i in images}
errors["image_uuid"] = errors.image_id.map(uuid_of)
summary = summarize(with_all(with_all(errors, "img_type"), "unit"), ["kind", "img_type", "level", "ref", "cmp", "unit"])
icc = icc_table(units)
nerr = node_errors(all4d)
nsummary = node_summary(nerr)
meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_uuids": len(uuids), "n_images": len(images),
        "n_measurements": n_meas, "n_suggestions": len(raw), "suggestion_errors": sugg_errors,
        "missing_uuids": sorted(missing), "no_model_nodes": no_model_nodes, "label_corrected": n_label_corrected,
        "models": sorted({f"{s['resp']['model']['name']} {s['resp']['model']['version']}" for s in raw.values()
                          if s["resp"].get("model")})}
print(meta["models"], f"| Label-Shift korrigiert: {n_label_corrected}, Bilder ohne nutzbare Modell-Nodes: {len(no_model_nodes)}")

# %% 4. Cache schreiben (pd.read_pickle(RESULT) -> dict)
result = {"meta": meta, "images": pd.DataFrame(images)[["id", "uuid", "img_type", "patient_id"]], "units": units,
          "errors": errors, "summary": summary, "icc": icc, "node_errors": nerr, "node_summary": nsummary}
pd.to_pickle(result, RESULT)
print("gespeichert:", RESULT)


# %% 5. Inspektion
pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 30)


def show(img_type="ALL", kind="rater_vs_model", level=None, unit="ALL", ref=None):
    """Summary gefiltert, z.B. show('Xray-LWS-lat'), show('MRI-LWS-sag', unit='L4-L5'), show(kind='rater_vs_rater')."""
    s = summary.reset_index()
    q = (s.img_type == img_type) & (s.kind == kind) & (s.unit == unit)
    q &= True if level is None else (s.level == level)
    q &= True if ref is None else (s.ref == ref)
    return s[q].drop(columns=["kind", "img_type", "unit"]).set_index(["level", "ref", "cmp"])


def show_icc(img_type="ALL", unit="ALL"):
    return icc.xs(img_type, level="img_type").xs(unit, level="unit")


def worst(n=20, kind="rater_vs_model", img_type=None, cmp_=None):
    """Größte Abweichungen mit Bild-UUID zum Nachschauen im Annotator."""
    e = errors[errors.kind == kind]
    if img_type:
        e = e[e.img_type == img_type]
    if cmp_:
        e = e[e.cmp == cmp_]
    return e.nlargest(n, "abs_err")[["image_uuid", "img_type", "unit", "ref", "cmp", "rater_name_a", "a", "b", "err"]]


print("Überblick (alle img_types, gepoolt):")
print(summary.xs("ALL", level="img_type").xs("ALL", level="unit")[["n", "n_images", "MAE", "RMSE", "bias", "within_5deg"]])
print("\nBeispiele: show('Xray-LWS-lat') | show_icc('MRI-LWS-sag') | worst(10) | nsummary.xs('ALL', level='vertebra_label')")

# %% 6. Optional: MAE je img_type, Modell vs. Rater im Vergleich zu Rater vs. Rater
if "--no-plot" not in sys.argv:
    import matplotlib.pyplot as plt
    p = (errors[errors.level == "segmental"].groupby(["img_type", "kind"]).abs_err.mean().unstack("kind"))
    p.plot.bar(figsize=(9, 4), ylabel="MAE segmental [°]", title="Segmentale Cobb-Winkel: Modell vs. Rater / Rater vs. Rater")
    plt.tight_layout()
    plt.show()
