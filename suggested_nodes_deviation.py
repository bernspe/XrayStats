"""Abweichung stateObject.nodes (final) vs. stateObject.suggestedNodes (Modell-Snapshot beim Suggest-Klick).

Je Messung werden die Nodes per Hungarian-Matching (euklidisch) den Vorschlägen zugeordnet; Abweichung =
Korrektur des Annotators in Stage-Pixeln (Stage ist ~600 px hoch, daher bildübergreifend vergleichbar).
Ausgewertet nach img_type, Annotator und Zeitpunkt (created_at: Monat, Verlauf, Bearbeitungsdauer).
Modell: GET /api/mlmodels/ liefert je Modell nur created_at (Registrierung), is_active (aktuell) und img_types -- keine
Aktivierungshistorie. "Wahrscheinliches Modell" = wie MLModel.active_for, aber zeitlich: neuestes bis created_at der
Messung registriertes Modell, das den img_type ausdrücklich nennt, sonst neuester Generalist (img_types = None).

  python suggested_nodes_deviation.py                    # Token aus XRAY_TOKEN oder Eingabe -> API
  python suggested_nodes_deviation.py --cache data/image_groups_cache.json   # offline aus JSON-Dump (Registry aus data/mlmodels.json, falls vorhanden)
  python suggested_nodes_deviation.py --check            # Selbsttest ohne API
Ergebnis: data/suggested_nodes_deviation.pkl (dict mit DataFrames) und .md (eigenständiger Bericht für andere Sitzungen).
"""
import getpass
import os
import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

API = "https://xray-api.renecol.org"
OUT = Path(__file__).parent / "data" / "suggested_nodes_deviation.pkl"
REPORT = OUT.with_suffix(".md")   # eigenständiger Bericht zum Weiterreichen an eine andere Sitzung
MODELS = Path(__file__).parent / "data" / "mlmodels.json"   # Registry-Kopie (wird beim Live-Lauf geschrieben)
KIND = "vertebra-keypoints-endplates"
MOVED_PX = 1.0   # Node gilt als verschoben, wenn Korrektur > MOVED_PX
# wie bspline_python.spine_labels (hier dupliziert, damit das Skript ohne matplotlib/dotmap läuft)
SPINE = [f"C{i}" for i in range(8)] + [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)] + ["S1"]
MIN_N = 3        # Mindestgröße einer Gruppe für den Kruskal-Wallis-Test


def load(cache=None):
    """-> (images, models). Live: beides von der API (Registry wird nach MODELS gespeichert)."""
    if cache:
        return json.loads(Path(cache).read_text()), (json.loads(MODELS.read_text()) if MODELS.exists() else [])
    h = {"Authorization": "Bearer " + (os.environ.get("XRAY_TOKEN") or getpass.getpass("xray-annotator Token: ").strip())}
    r = requests.get(f"{API}/api/imagedata/", params={"detailed": True, "complete": True}, headers=h, timeout=600)
    r.raise_for_status()
    m = requests.get(f"{API}/api/mlmodels/", headers=h, timeout=60)
    m.raise_for_status()
    MODELS.write_text(json.dumps(m.json()))
    return r.json(), m.json()


def probable_model(models, img_type, t):
    """Wahrscheinlich benutztes Modell zum Zeitpunkt t (Spezialist vor Generalist, neuestes zuerst). models: nach created_at absteigend."""
    cand = [m for m in models if m["created_at"] <= t]
    for pick in (lambda m: img_type in (m["img_types"] or []), lambda m: m["img_types"] is None):
        for m in cand:
            if pick(m):
                return f"{m['name']} {m['version']}"
    return "unbekannt"


def _xy(nodes):
    return np.array([[n["x"], n["y"]] for n in nodes], float).reshape(-1, 2)


def build(images, models=()):
    """-> (meas, nodes, n_skipped): eine Zeile je Messung bzw. je zugeordnetem Node-Paar."""
    models = sorted((dict(m, created_at=pd.to_datetime(m["created_at"], utc=True)) for m in models
                     if m.get("kind", KIND) == KIND), key=lambda m: m["created_at"], reverse=True)
    meas, nodes, skipped = [], [], 0
    for im in images:
        for m in im.get("measure_dataset") or []:
            s = ((m.get("data") or {}).get("xray_measurement_data") or {}).get("stateObject") or {}
            fin, sug = _xy(s.get("nodes") or []), _xy(s.get("suggestedNodes") or [])
            if not len(fin) or not len(sug):   # Feature nicht genutzt / leer
                skipped += 1
                continue
            u = m.get("user") or {}
            meta = {"m_uuid": m["uuid"], "img_uuid": im["uuid"], "img_type": im["img_type"],
                    "annotator": f"{u.get('first_name', '?')} (#{u.get('id')})",
                    "created_at": pd.to_datetime(m["created_at"], utc=True),
                    "last_modified": pd.to_datetime(m["last_modified"], utc=True)}
            meta["model"] = probable_model(models, im["img_type"], meta["created_at"])
            # Registry kennt keinen Suggest-Zeitpunkt: liegt ein Modellwechsel zwischen Start und Ende der Messung, ist es unsicher
            meta["model_ambiguous"] = meta["model"] != probable_model(models, im["img_type"], meta["last_modified"])
            r, c = linear_sum_assignment(cdist(fin, sug))
            d = fin[r] - sug[c]                # Korrektur des Annotators (final - Vorschlag)
            dev = np.hypot(d[:, 0], d[:, 1])
            n_add, n_rem = len(fin) - len(r), len(sug) - len(r)
            meas.append({**meta, "n_nodes": len(fin), "n_sugg": len(sug), "n_added": n_add, "n_removed": n_rem,
                         "mean_dev": dev.mean(), "max_dev": dev.max(), "mean_dx": d[:, 0].mean(), "mean_dy": d[:, 1].mean(),
                         "unchanged": bool((dev <= MOVED_PX).all() and not n_add and not n_rem)})
            # Wirbel wie in annotation_analysis.build_nodes_df: Node i (y-sortiert) = vertebra_range[0] + i; nur wenn Range zur Nodezahl passt
            vr = s.get("vertebra_range") or [0, -1]
            lab = [SPINE[vr[0] + i] if vr[1] - vr[0] + 1 == len(fin) and vr[0] + i < len(SPINE) else None for i in r]
            nodes += [{**meta, "vertebra": l, "dev": v, "dx": x, "dy": y} for l, v, (x, y) in zip(lab, dev, d)]
    meas, nodes = pd.DataFrame(meas), pd.DataFrame(nodes)
    for df in (meas, nodes):
        df["month"] = df["created_at"].dt.strftime("%Y-%m")
    meas["days"] = (meas["created_at"] - meas["created_at"].min()).dt.total_seconds() / 86400
    meas["minutes"] = (meas["last_modified"] - meas["created_at"]).dt.total_seconds() / 60  # Obergrenze der Bearbeitungszeit
    return meas, nodes, skipped


def summary(meas, nodes, by):
    g = meas.groupby(by)["mean_dev"]
    s = g.agg(n_meas="size", mean="mean", sd="std", median="median",
              q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
    s["unchanged_%"] = meas.groupby(by)["unchanged"].mean() * 100
    s["added/meas"] = meas.groupby(by)["n_added"].mean()
    s["removed/meas"] = meas.groupby(by)["n_removed"].mean()
    gn = nodes.groupby(by)["dev"]
    s["n_nodes"], s["node_median"], s["node_p90"] = gn.size(), gn.median(), gn.quantile(.9)
    s["moved_%"] = (nodes["dev"] > MOVED_PX).groupby([nodes[b] for b in ([by] if isinstance(by, str) else by)]).mean() * 100
    return s


def tests(meas):
    """Kruskal-Wallis je Faktor (Messungs-Mittelwerte) und Spearman gegen Zeit / Dauer / Verlauf je Annotator."""
    rows = []
    for col in ("img_type", "annotator", "month", "model"):
        gs = [g["mean_dev"].values for _, g in meas.groupby(col) if len(g) >= MIN_N]
        if len(gs) > 1:
            h, p = stats.kruskal(*gs)
            rows.append({"test": f"Kruskal-Wallis ~ {col}", "stat": h, "p": p, "n": sum(map(len, gs))})
    for col in ("days", "minutes"):
        rho, p = stats.spearmanr(meas[col], meas["mean_dev"], nan_policy="omit")
        rows.append({"test": f"Spearman mean_dev ~ {col}", "stat": rho, "p": p, "n": len(meas)})
    for a, g in meas.groupby("annotator"):   # Lerneffekt: wird die Korrektur mit der Zeit kleiner?
        if len(g) >= 5:
            rho, p = stats.spearmanr(g["days"], g["mean_dev"])
            rows.append({"test": f"Spearman mean_dev ~ days | {a}", "stat": rho, "p": p, "n": len(g)})
    return pd.DataFrame(rows)


def _wilcoxon_p(x):
    x = x[x != 0]
    return stats.wilcoxon(x).pvalue if len(x) >= 8 else np.nan


def vertebra_corrections(nodes, by=()):
    """Je Wirbel: wie oft verschoben (> MOVED_PX) und in welche Richtung. dx > 0 = nach rechts, dy > 0 = nach unten (Bildkoordinaten).

    Richtung nur über die verschobenen Nodes (sonst verwässern die Nullen). *_pos_% = Anteil der Verschiebungen mit
    positivem Vorzeichen, *_p = Wilcoxon gegen 0 (ab 8 Nodes). Hinzugefügte/entfernte Nodes haben kein Paar und fehlen hier.
    """
    n = nodes.dropna(subset=["vertebra"]).assign(moved=lambda d: d["dev"] > MOVED_PX)
    keys = ["vertebra", *by]
    s = n.groupby(keys).agg(n_nodes=("dev", "size"), n_moved=("moved", "sum"), moved_pct=("moved", "mean"))
    s["moved_pct"] *= 100
    mv = n[n.moved].groupby(keys)
    s = s.join(mv.agg(dev_moved=("dev", "median"), dx_med=("dx", "median"), dy_med=("dy", "median"),
                      dx_pos_pct=("dx", lambda x: (x > 0).mean() * 100), dy_pos_pct=("dy", lambda x: (x > 0).mean() * 100),
                      dx_p=("dx", _wilcoxon_p), dy_p=("dy", _wilcoxon_p)))
    if by:
        s = s.reorder_levels([*by, "vertebra"])
    return s.sort_index(key=lambda i: i.map(SPINE.index) if i.name == "vertebra" else i)


def model_usage(meas):
    """Je img_type und Modell: Zeitraum und Anzahl der Messungen, denen es zugeordnet wurde."""
    g = meas.groupby(["img_type", "model"])
    return g.agg(n_meas=("m_uuid", "size"), n_ambiguous=("model_ambiguous", "sum"),
                 first=("created_at", "min"), last=("created_at", "max"))


HEADER = """# Abweichung nodes vs. suggestedNodes (Xray-Annotator)

Erzeugt von `XrayStats/suggested_nodes_deviation.py` am {now}. Alle Daten: `GET /api/imagedata/?detailed=True&complete=True`
(`measure_dataset[].data.xray_measurement_data.stateObject.nodes` = final, `.suggestedNodes` = Modell-Snapshot beim Suggest-Klick).
Rohdaten dieses Laufs: `data/suggested_nodes_deviation.pkl` (dict: `measurements`, `nodes`, alle Tabellen unten als DataFrames,
`report` = dieser Text); Neuberechnung: `python suggested_nodes_deviation.py` (Token) oder `--cache data/image_groups_cache.json`.

## Methodik
- Je Messung Hungarian-Matching der finalen Nodes auf die Vorschläge (euklidisch, Stage-Pixel, Stage ca. 600 px hoch).
  Korrektur = final - Vorschlag; `dev` = Betrag, `dx > 0` = nach rechts, `dy > 0` = nach unten (Bildkoordinaten).
  Nicht zuordenbare Nodes zählen als `added` (final mehr) bzw. `removed` (Vorschlag mehr), nicht in dev/dx/dy.
- Verschoben = dev > {moved} px. Messung `unchanged` = nichts verschoben, hinzugefügt oder entfernt.
- Messungen ohne `suggestedNodes` (null/leer) werden übersprungen.
- Wirbel-Label: Node i (y-sortiert) = `vertebra_range[0] + i`; nur wenn Range zur Nodezahl passt, sonst fehlt der Node in den Wirbeltabellen.
- Zeit = `created_at` der Messung (Monat); `minutes` = last_modified - created_at (Obergrenze der Bearbeitungszeit).
- Annotator = `first_name (#user_id)`.
- Modell: GET /api/mlmodels/ kennt nur `created_at` (Registrierung) und aktuelles `is_active`, KEINE Aktivierungshistorie. `model` = neuestes bis
  `created_at` der Messung registriertes Modell, das den img_type ausdrücklich nennt (Spezialist), sonst neuester Generalist; ohne Registry
  `unbekannt`. `n_ambiguous` = Modellwechsel zwischen Start und Ende der Messung. Nur Schätzung.

## Spalten
- Gruppen (`summary`): `n_meas`, `mean/sd/median/q25/q75` = Verteilung der Messungs-Mittelwerte der Korrektur (px); `unchanged_%`;
  `added/meas`, `removed/meas`; `n_nodes`, `node_median`, `node_p90` (Node-Ebene); `moved_%` = Anteil verschobener Nodes.
- Wirbel: `n_moved`, `moved_pct`, `dev_moved` (Median der Korrektur der verschobenen), `dx_med/dy_med` (Median nur über verschobene),
  `dx_pos_pct/dy_pos_pct` (Anteil positiver Vorzeichen), `dx_p/dy_p` (Wilcoxon gegen 0, ab 8 verschobenen Nodes).

## Einschränkungen
- Annotator, img_type, Monat und Modell sind stark vermischt (z. B. ein Annotator macht fast nur MRI-sag); Tests zeigen nur Unterschiede, keine Ursache.
- Ob dx links/rechts vorne/hinten heißt, hängt von der Bildausrichtung ab (nicht bekannt) -> dx am besten je img_type lesen.
- Hinzugefügte/entfernte Nodes sind pro Messung gezählt, nicht pro Wirbel.
- Kleine Gruppen (n < ~10) nicht überinterpretieren.

"""


def report(res, skipped, n_models):
    """Markdown-Bericht: Header (Methodik/Spalten/Grenzen) + alle Tabellen; eigenständig lesbar in einer anderen Sitzung."""
    meas, nodes = res["measurements"], res["nodes"]
    out = [HEADER.format(now=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), moved=MOVED_PX),
           "## Überblick",
           f"- {len(meas)} Messungen mit suggestedNodes ({len(nodes)} Node-Paare), {skipped} ohne -> übersprungen",
           f"- Zeitraum created_at: {meas.created_at.min():%Y-%m-%d} bis {meas.created_at.max():%Y-%m-%d}",
           f"- Median der Ø-Korrektur {meas.mean_dev.median():.2f} px; {meas.unchanged.mean() * 100:.0f}% der Messungen unverändert"]
    if not n_models:
        out.append("- WARNUNG: keine Registry (data/mlmodels.json fehlt) -> model = 'unbekannt'")
    tabs = [("Wirbel nach Häufigkeit der Korrektur (alle img_types)", res["by_vertebra"].sort_values("moved_pct", ascending=False)),
            ("Wirbel in Wirbelsäulen-Reihenfolge je img_type", res["by_img_type_vertebra"]),
            ("Modell-Nutzung (wahrscheinlich)", res["model_usage"]), ("Nach Modell", res["by_model"]),
            ("Nach img_type und Modell", res["by_img_type_model"]), ("Nach img_type", res["by_img_type"]),
            ("Nach Annotator", res["by_annotator"]), ("Nach Monat", res["by_month"]),
            ("Nach img_type und Annotator", res["by_img_type_annotator"]), ("Nach img_type und Monat", res["by_img_type_month"]),
            ("Tests (Kruskal-Wallis über Messungs-Mittelwerte, Spearman)", res["tests"])]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        for title, t in tabs:
            out.append(f"\n## {title}\n```\n{t.to_string(float_format='{:.3g}'.format)}\n```")
    return "\n".join(out)


def run(images, models=()):
    meas, nodes, skipped = build(images, models)
    res = {"measurements": meas, "nodes": nodes,
           "by_img_type": summary(meas, nodes, "img_type"), "by_annotator": summary(meas, nodes, "annotator"),
           "by_month": summary(meas, nodes, "month"), "by_img_type_annotator": summary(meas, nodes, ["img_type", "annotator"]),
           "by_img_type_month": summary(meas, nodes, ["img_type", "month"]), "by_model": summary(meas, nodes, "model"),
           "by_img_type_model": summary(meas, nodes, ["img_type", "model"]), "model_usage": model_usage(meas),
           "registry": pd.DataFrame(models), "by_vertebra": vertebra_corrections(nodes),
           "by_img_type_vertebra": vertebra_corrections(nodes, ["img_type"]), "tests": tests(meas)}
    res["report"] = report(res, skipped, len(models))
    print(res["report"])
    return res


def check():
    def msr(uid, nodes, sugg):
        pts = lambda a: [{"x": x, "y": y} for x, y in a]
        return {"uuid": uid, "user": {"id": 1, "first_name": "A"}, "created_at": "2026-01-01T10:00:00Z",
                "last_modified": "2026-01-01T10:05:00Z",
                "data": {"xray_measurement_data": {"stateObject": {"nodes": pts(nodes), "suggestedNodes": pts(sugg),
                                                            "vertebra_range": [17, 17 + len(nodes) - 1]}}}}
    base = [(10, 10), (12, 60), (15, 110)]
    ims = [{"uuid": "i1", "img_type": "T", "measure_dataset": [
        msr("m1", base, base),                                      # unverändert
        msr("m2", [(x + 3, y + 4) for x, y in base] + [(20, 200)], base),   # alle um 5 px, 1 Node dazu
        msr("m3", base, [])]}]                                      # kein Vorschlag -> übersprungen
    meas, nodes, skipped = build(ims)
    assert skipped == 1 and list(meas.m_uuid) == ["m1", "m2"], meas
    assert meas.unchanged.tolist() == [True, False] and meas.n_added.tolist() == [0, 1]
    assert np.allclose(meas.mean_dev, [0, 5]) and np.allclose(meas.mean_dx, [0, 3]) and np.allclose(meas.mean_dy, [0, 4])
    assert summary(meas, nodes, "img_type").loc["T", "moved_%"] == 50.0   # 3 von 6 Paaren verschoben
    reg = [{"name": n, "version": "1", "kind": KIND, "img_types": it, "created_at": t}
           for n, it, t in [("gen", None, "2025-12-01T00:00:00Z"), ("spec", ["T"], "2026-01-01T10:02:00Z"),
                            ("gen2", None, "2026-01-01T09:00:00Z")]]
    meas, _, _ = build(ims, reg)
    # created_at 10:00: Spezialist erst 10:02 registriert -> Generalist gen2; last_modified 10:05 -> Spezialist => unsicher
    assert (meas.model == "gen2 1").all() and meas.model_ambiguous.all(), meas[["model", "model_ambiguous"]]
    assert build(ims)[0].model.eq("unbekannt").all()   # ohne Registry
    # m2: alle 3 gepaarten Nodes um (+3, +4) verschoben -> T10..T12 je 1x; T10 unverändert in m1 -> 50 % verschoben
    v = vertebra_corrections(nodes)
    assert v.loc["T10", ["n_nodes", "n_moved"]].tolist() == [2, 1] and v.loc["T10", "dx_med"] == 3 and v.loc["T10", "dy_pos_pct"] == 100
    assert list(v.index) == ["T10", "T11", "T12"] and vertebra_corrections(nodes, ["img_type"]).index[0] == ("T", "T10")
    assert "## Wirbel in Wirbelsäulen" in report(run(ims), 1, 0)
    print("check ok")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
        res = run(*load(cache))
        pd.to_pickle(res, OUT)
        REPORT.write_text(res["report"])
        print(f"\n-> {OUT}\n-> {REPORT}")
