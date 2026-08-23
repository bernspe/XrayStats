# Intermodale Vergleichs-App (`image_group_comparison_app.py`)

Statistischer Vergleich von Cobb-Winkel-Messungen zwischen verschiedenen Bildmodalitäten
(Röntgen vs. MRT, Inklination vs. Reklination) für Patienten mit zugeordneten `image_groups`.

## Start

```bash
cd /Users/peterbernstein/Python/XrayStats
/Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run image_group_comparison_app.py
```

---

## Konzept: image_groups als Paarungsschlüssel

Die Grundidee ist, dass Bilder desselben Patienten über das Feld `Patient.image_groups`
miteinander verknüpft werden. Nur Bilder, die derselben Gruppe angehören, werden
verglichen — Bilder aus verschiedenen Gruppen werden nie gemischt.

Das Feld `image_groups` ist ein `JSONField` auf dem `Patient`-Modell im xray-annotator-Backend
und wird über den Endpoint `GET /api/imagedata/?detailed=True&complete=True` unter
`referring_patient.image_groups` zurückgegeben. Die App unterstützt drei Formate:

| Format | Beispiel |
|---|---|
| Strings | `["Gruppe A", "Gruppe B"]` |
| Objekte mit UUID-Liste | `[{"name": "G1", "images": ["uuid1", "uuid2"]}]` |
| Objekte mit img_type-Liste | `[{"name": "G1", "img_types": ["Xray-LWS-ap", "MRI-LWS-cor"]}]` |

Ohne explizite UUID- oder img_type-Filterung werden alle Bilder des Patienten der Gruppe
zugeordnet.

---

## Daten laden und cachen

Die App liest Daten aus dem API-Cache `data/image_groups_cache.json`. Die Sidebar bietet:

- **Bearer Token** (Passwortfeld): Token von `xray-annotator.renecol.org` → Account → Token kopieren
- **"Von API laden & cachen"**: Ruft den Endpoint ab und speichert das Ergebnis lokal.
  Bei erneutem Start wird der Cache automatisch verwendet, kein Token erforderlich.

Der Cache ist ein einfaches JSON-Dump der vollständigen API-Antwort. Er enthält keine
signierten Bild-URLs (diese wären innerhalb von ~24 h abgelaufen und werden hier nicht
benötigt).

---

## Vergleichspaare

In der Sidebar konfigurierbar (eine Zeile pro Paar, `TypeA, TypeB`). Standardpaare:

| TypeA | TypeB | Klinische Frage |
|---|---|---|
| `Xray-LWS-ap` | `MRI-LWS-cor` | Übereinstimmung Röntgen ↔ MRT frontal |
| `Xray-LWS-lat` | `MRI-LWS-sag` | Übereinstimmung Röntgen ↔ MRT sagittal |
| `Xray-LWS-recl` | `Xray-LWS-incl` | Reklination ↔ Inklination |

Jedes Paar erzeugt im Vergleichs-Tab einen eigenen Abschnitt.

---

## Cobb-Typ-Auswahl

Über ein Dropdown wird einer der drei Mess-Typen ausgewählt:

| Wert | Bedeutung |
|---|---|
| `Spline` | Rekonstruierter Spline-Winkel (glattere Kurve) |
| `Cobb-regional` | Manuell gemessener regionaler Cobb-Winkel |
| `Cobb-segmental` | Manuell gemessener segmentaler Cobb-Winkel |

---

## Regions-Mapping

Jeder Winkel wird anhand von `vertebra_upper` und `vertebra_lower` einer Region
zugeordnet. Die Logik (in `angle_region()`):

**Regional / Spline:**

| Region | Vertebra-Bereich | Anmerkung |
|---|---|---|
| `LU` | ≤ L1 → S1 | Gesamte Lumballordose; Sonderfall, der MT–L-LU überlagert |
| `L-LU` | L3–S1 | Unterer Lumbalbereich (`spine_regions['L-LU'] = [22, 25]`) |
| `U-LU` | L1–L3 | Oberer Lumbalbereich (`spine_regions['U-LU'] = [20, 22]`) |
| `TL` | T11–T12 | Thorakolumbaler Übergang |
| `MT` | T5–T10 | Mittlere Thorakalregion |

Die LU-Erkennung hat Vorrang: wenn `upper_idx ≤ 20` (L1 oder kranial) **und**
`lower_idx ≥ 25` (S1), wird die Region als `LU` klassifiziert, unabhängig vom
Ergebnis von `convertMeasurementBorderToSpineRegion`.

**Segmental:**

| Region | Vertebrapaar |
|---|---|
| `T11/T12` | T11 → T12 |
| `T12/L1` | T12 → L1 |
| `L1/L2` | L1 → L2 |
| `L2/L3` | L2 → L3 |
| `L3/L4` | L3 → L4 |
| `L4/L5` | L4 → L5 |
| `L5/S1` | L5 → S1 |

---

## Statistischer Vergleich — Methodik

### Paarbildung

Für jeden Patienten und jede seiner `image_groups`:

1. Alle Bilder vom Typ `TypeA` → `mean_angles_for_images(imgs_a)`
2. Alle Bilder vom Typ `TypeB` → `mean_angles_for_images(imgs_b)`
3. Inner Join über `(cobbType, region)` → gepaarte Werte

`mean_angles_for_images` fasst alle Messungen aller Bilder desselben Typs für einen
Patienten zusammen und berechnet:
- **Mittlerer Winkel** pro `(cobbType, region)` über alle Messungen aller Annotatoren
- **n_raters**: Anzahl eindeutiger Annotator-Usernames über alle Bilder dieses Typs

### Kennzahlen (über alle Regionen)

| Kennzahl | Beschreibung |
|---|---|
| n Paare gesamt | Gesamtanzahl gepaarter (Patient, Region)-Kombinationen |
| Bias (A−B) | Mittlere Differenz TypeA minus TypeB in Grad |
| Pearson r | Korrelationskoeffizient über alle Paare |
| p (paired t) | p-Wert des gepaarten t-Tests |

### Visualisierungen

**Balkendiagramm (Mittelwert ± SD je Region)**
- X-Achse: Regionen in klinisch sinnvoller Reihenfolge (MT → TL → U-LU → L-LU → LU
  bzw. T11/T12 → … → L5/S1)
- Zwei Balkengruppen pro Region: TypeA (blau) und TypeB (orange)
- Fehlerbalken: ±1 Standardabweichung über alle Patienten dieser Region
- Beschriftung: n-Anzahl unter jedem Regionspaar

**Bland-Altman-Diagramm**
- X-Achse: Mittelwert beider Modalitäten pro Paar
- Y-Achse: Differenz TypeA − TypeB
- Farbe: Region (Tab10-Palette in Reihenfolge der Regionen)
- Linien: Bias (durchgezogen) und ±1,96·SD / Limits of Agreement (gestrichelt)

### Per-Region-Statistiktabelle

Direkt unterhalb der Grafiken: MW, SD, Bias und Pearson-r je Region. Regions ohne
mindestens 2 Paare erhalten keinen r-Wert (`—`).

---

## Tabs

### Gruppen-Übersicht
Zusammenfassung aller erkannten `image_groups`: Anzahl Patienten und vorhandene
`img_types`. Expandable-Sektionen zeigen Bild-UUID, Aufnahmedatum und Anzahl
Messungen je Bild.

### Statistischer Vergleich
Hauptansicht. Cobb-Typ-Dropdown → je Vergleichspaar ein Abschnitt mit Kennzahlen,
Grafiken, Statistiktabelle und aufklappbaren Rohdaten.

In den **Rohdaten** erscheinen pro Zeile:
`patient_id`, `image_group`, `cobbType`, `region`, Winkelwerte beider Typen,
`n_raters_<TypeA>`, `n_raters_<TypeB>`, Differenz.

### Debug / Rohdaten
Zeigt die Rohstruktur von `image_groups` aus der API (erste 10 Einträge),
Anzahl Patienten mit nicht-leeren Gruppen und den aufgelösten `group_index`.

---

## Datenfluss

```
API /api/imagedata/
  └─ data/image_groups_cache.json
       └─ build_group_index(data)
            └─ {group_name: [{patient_id, images: {img_type: [ImageData]}}]}
                 └─ für jedes Vergleichspaar (TypeA, TypeB):
                      ├─ mean_angles_for_images(imgs_a)  → df_a (cobbType, region, mean_angle, n_raters)
                      ├─ mean_angles_for_images(imgs_b)  → df_b
                      └─ pair_angles(df_a, df_b)         → gepaarte Zeilen
                           └─ data_by_region {region: {a: [...], b: [...]}}
                                ├─ region_bar_plot()
                                ├─ bland_altman_by_region()
                                └─ Per-Region-Statistiktabelle
```

---

## Abhängigkeiten

Zusätzlich zu den Projekt-Standardpaketen (`numpy`, `scipy`, `pandas`, `matplotlib`,
`requests`, `streamlit`) importiert die App:

- `bspline_python.spine_labels` — Vertebra-Index-Lookup für LU-Erkennung
- `bspline_python.convertMeasurementBorderToSpineRegion` — Vertebrapaar → Regionsname
