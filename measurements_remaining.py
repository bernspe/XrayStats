"""Wie viele Messungen fehlen noch, bis jedes Bild TARGET unabhängige Messungen hat?

Unabhängig = verschiedene Nutzer (mehrere Messungen desselben Nutzers am selben Bild zählen einmal).
Auswertung je img_type, gruppiert in aufwendig (Xray-WS-ap/-lat) und übrige.

Token: Umgebungsvariable XRAY_TOKEN oder Eingabeaufforderung (xray-annotator -> Account info -> Copy token).
Aufruf: python measurements_remaining.py [--check]   (--check = Selbsttest ohne API)
"""
import getpass
import os
import sys

import pandas as pd
import requests

API = "https://xray-api.renecol.org"
TARGET = 3
AUFWENDIG = {"Xray-WS-ap", "Xray-WS-lat"}


def fetch_imagedata():
    token = os.environ.get("XRAY_TOKEN") or getpass.getpass("xray-annotator Token: ").strip()
    r = requests.get(f"{API}/api/imagedata/", params={"detailed": True, "complete": True},
                     headers={"Authorization": f"Bearer {token}"}, timeout=600)
    r.raise_for_status()
    return r.json()


def remaining(images, target=TARGET):
    """Eine Zeile je Bild: n_users (unabhängig), n_missing, Gruppe."""
    rows = []
    for im in images:
        users = {m["user"]["id"] for m in im.get("measure_dataset") or []}
        t = im.get("img_type") or "(leer)"
        rows.append({"img_type": t, "gruppe": "aufwendig" if t in AUFWENDIG else "andere",
                     "n_users": len(users), "n_missing": max(0, target - len(users))})
    return pd.DataFrame(rows)


def report(df):
    agg = lambda g: g.agg(bilder=("n_users", "size"), unvollständig=("n_missing", lambda s: (s > 0).sum()),
                          fehlende_messungen=("n_missing", "sum"))
    by_type = agg(df.groupby(["gruppe", "img_type"])).sort_values("fehlende_messungen", ascending=False)
    by_group = agg(df.groupby("gruppe"))
    by_group.loc["GESAMT"] = by_group.sum()
    print(f"Ziel: {TARGET} unabhängige Messungen (verschiedene Nutzer) je Bild\n")
    print("Je Bildtyp:\n", by_type.to_string(), "\n")
    print("Je Gruppe:\n", by_group.to_string())
    print("\nVerteilung vorhandener Messungen (Bilder):\n",
          df.pivot_table(index="gruppe", columns="n_users", values="img_type", aggfunc="size", fill_value=0).to_string())


def _check():
    u = lambda i: {"user": {"id": i}}
    imgs = [{"img_type": "Xray-WS-ap", "measure_dataset": [u(1), u(1)]},      # 1 Nutzer -> 2 fehlen
            {"img_type": "Xray-LWS-ap", "measure_dataset": [u(1), u(2), u(3), u(4)]},  # fertig
            {"img_type": "", "measure_dataset": []}]                           # 3 fehlen, "(leer)"
    df = remaining(imgs)
    assert df.n_missing.tolist() == [2, 0, 3]
    assert df.gruppe.tolist() == ["aufwendig", "andere", "andere"]
    print("check ok")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        report(remaining(fetch_imagedata()))
