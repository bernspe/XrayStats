"""
Übersicht über Patient.image_groups der xray-api:
  1. Zusammensetzung (img_types) der Gruppen
  2. Anzahl Bilder / Messobjekte (measure_dataset) je Gruppen-Zusammensetzung und je img_type

Aufruf:
    python image_groups_overview.py <TOKEN>     # lädt von der API und schreibt Cache
    python image_groups_overview.py              # nutzt data/image_groups_cache.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://xray-api.renecol.org"
CACHE = Path("data/image_groups_cache.json")


def load(token=None):
    if token:
        r = requests.get(f"{API_URL}/api/imagedata/", params={"detailed": True, "complete": True},
                         headers={"Authorization": f"Bearer {token}"}, timeout=300)
        r.raise_for_status()
        data = r.json()
        CACHE.write_text(json.dumps(data))
        return data
    return json.loads(CACHE.read_text())


def main(data):
    imgs = {i["uuid"]: i for i in data}  # dedupe nach Bild-UUID
    n_meas = lambda u: len(imgs[u].get("measure_dataset") or [])

    # Gruppen deduplizieren: (Patient, Label) -> uuids (image_groups steht an jedem Bild des Patienten)
    groups, unknown = {}, 0
    for i in imgs.values():
        rp = i.get("referring_patient") or {}
        for g in rp.get("image_groups") or []:
            groups[(rp.get("uuid"), g.get("label"))] = g.get("uuids") or []
    print(f"{len(imgs)} Bilder, {len(groups)} Gruppen, "
          f"{len({p for p, _ in groups})} Patienten mit image_groups\n")

    comp_rows, type_rows = [], []
    for (pat, label), uuids in groups.items():
        known = [u for u in uuids if u in imgs]
        unknown += len(uuids) - len(known)
        types = Counter(imgs[u]["img_type"] for u in known)
        comp_rows.append({
            "label": label,
            "zusammensetzung": " + ".join(f"{t} x{n}" if n > 1 else t for t, n in sorted(types.items())),
            "n_bilder": len(known),
            "n_messobjekte": sum(n_meas(u) for u in known),
            "n_bilder_mit_messung": sum(n_meas(u) > 0 for u in known),
        })
        for u in known:
            type_rows.append({"label": label, "img_type": imgs[u]["img_type"], "messobjekte": n_meas(u)})

    comp = pd.DataFrame(comp_rows)
    pd.set_option("display.width", 200, "display.max_colwidth", 90, "display.max_rows", 200)

    print("== Label-Häufigkeit ==")
    print(comp["label"].value_counts().to_string(), "\n")

    print("== Zusammensetzung der Gruppen (je Label) ==")
    print(comp.groupby(["label", "zusammensetzung"])
              .agg(n_gruppen=("n_bilder", "size"), n_bilder=("n_bilder", "sum"),
                   n_messobjekte=("n_messobjekte", "sum"), n_bilder_mit_messung=("n_bilder_mit_messung", "sum"))
              .sort_values("n_gruppen", ascending=False).to_string(), "\n")

    print("== Je img_type innerhalb von Gruppen ==")
    t = pd.DataFrame(type_rows)
    print(t.groupby("img_type").agg(n_bilder=("messobjekte", "size"), n_messobjekte=("messobjekte", "sum"),
                                    bilder_ohne_messung=("messobjekte", lambda s: (s == 0).sum()),
                                    max_messobjekte_pro_bild=("messobjekte", "max"))
              .sort_values("n_bilder", ascending=False).to_string(), "\n")

    print("== Verteilung: Messobjekte pro Bild (in Gruppen) ==")
    print(t["messobjekte"].value_counts().sort_index().to_string())

    if unknown:
        print(f"\nHinweis: {unknown} UUIDs in image_groups ohne passendes Bild in der API-Antwort.")


if __name__ == "__main__":
    main(load(sys.argv[1] if len(sys.argv) > 1 else None))
