# API `/api/imagedata/` – Strukturanalyse

**Endpoint:** `GET https://xray-api.renecol.org/api/imagedata/`  
**Parameter:** `?detailed=True&complete=True`  
**Auth:** Bearer Token  
**Abfragedatum:** 2026-06-12  

---

## 1. Überblick

Der Endpoint liefert eine **JSON-Array** mit Röntgenbild-Datensätzen.

| Kennzahl | Wert |
|---|---|
| Gesamtanzahl Datensätze | **792** |
| Datensätze mit Messung(en) | **483** |
| Datensätze ohne Messung | **309** |
| Gesamtanzahl Messungen | **1.117** |
| Gesamtanzahl Winkel | **7.440** |
| Zeitraum Aufnahmen | 2007-11-22 bis 2025-12-02 |
| Zeitraum API-Einträge | 2025-08-23 bis 2026-06-11 |

---

## 2. Top-Level Struktur

Jedes Element im Array repräsentiert ein **Röntgenbild** (`ImageData`):

```json
{
  "id": 800,
  "uuid": "c057b7b7-397a-4572-8a78-a9d4fef72c4e",
  "img": "<signed-GCS-URL>",
  "thumb": "<signed-GCS-URL>",
  "img_type": "Xray-LWS-ap",
  "acquisition_date": "2021-03-15",
  "patient_id": "VXP3",
  "patient_id_integrity_check": true,
  "public": true,
  "created_at": "2025-10-01T10:00:00Z",
  "last_modified": "2025-10-01T10:00:05Z",
  "image_hash": "<sha256-hash>",
  "upload_user": { ... },
  "referring_patient": { ... },
  "measure_dataset": [ ... ]
}
```

### Felder im Überblick

| Feld | Typ | Beschreibung |
|---|---|---|
| `id` | integer | Primärschlüssel des Bildes |
| `uuid` | string (UUID) | Eindeutige Bild-UUID |
| `img` | string (URL) | Signierte Google Cloud Storage URL zum Vollbild (24h gültig) |
| `thumb` | string (URL) | Signierte GCS URL zum Thumbnail (WebP, 24h gültig) |
| `img_type` | string | Bildmodalität und Region (siehe §3) |
| `acquisition_date` | string (ISO 8601) | Aufnahmedatum (`null` bei 3 Einträgen) |
| `patient_id` | string | Patienten-Kürzel (z. B. `VXP3`) |
| `patient_id_integrity_check` | boolean | Immer `true` (alle 792 Einträge) |
| `public` | boolean | `true` bei 734, `false` bei 58 Einträgen |
| `created_at` | string (ISO 8601) | Erstellungszeitpunkt |
| `last_modified` | string (ISO 8601) | Letzter Änderungszeitpunkt |
| `image_hash` | string (sha256) | Prüfsumme des Bildinhalts |
| `upload_user` | object | Hochladender Nutzer (§4) |
| `referring_patient` | object | Zugehöriger Patient (§5) |
| `measure_dataset` | array | Messungen zu diesem Bild (§6) |

---

## 3. `img_type` – Bildmodalitäten

| Wert | Anzahl | Bedeutung |
|---|---|---|
| `Xray-LWS-ap` | 247 | Röntgen Lendenwirbelsäule, anterior-posterior |
| `Xray-LWS-lat` | 231 | Röntgen LWS, lateral |
| `MRI-LWS-cor` | 64 | MRT LWS, koronal |
| `MRI-LWS-sag` | 58 | MRT LWS, sagittal |
| `Xray-WS-ap` | 54 | Röntgen Wirbelsäule gesamt, ap |
| `Xray-LWS-incl` | 42 | Röntgen LWS, Inklination |
| `Xray-LWS-recl` | 41 | Röntgen LWS, Reklination |
| `Xray-WS-lat` | 40 | Röntgen WS gesamt, lateral |
| `MRI-WS-cor` | 4 | MRT WS gesamt, koronal |
| `Xray-BWS-ap` | 4 | Röntgen Brustwirbelsäule, ap |
| `Xray-BWS-lat` | 1 | Röntgen BWS, lateral |
| *(leer)* | 6 | Nicht klassifiziert |

---

## 4. `upload_user` – Hochladender Nutzer

```json
{
  "id": 16,
  "first_name": "Victoria",
  "last_name": "Ilner",
  "email": "victoria.ilner1@gmail.com",
  "username": "victoria.ilner1"
}
```

- **10 distinkte Nutzer** im Datensatz
- Felder: `id`, `first_name`, `last_name`, `email`, `username`

---

## 5. `referring_patient` – Patientenobjekt

```json
{
  "id": 121,
  "uuid": "ddbfc4e3-f854-4c73-8466-85e90f39ef6f",
  "patient_id": "VXP3",
  "institution": 1,
  "created_at": "2026-06-11T11:10:52.697152Z",
  "index_date": null,
  "hasClinicalBaseData": 0,
  "completeNessClinicalBaseData": 0,
  "numberClinicalEntries": 0,
  "completeNessClinicalEntries": 0,
  "hasICFxEntry": 0,
  "progresstag": []
}
```

| Feld | Typ | Beschreibung |
|---|---|---|
| `id` | integer | Patienten-ID |
| `uuid` | string (UUID) | Patienten-UUID |
| `patient_id` | string | Alphanumerisches Kürzel |
| `institution` | integer | Institution (nur Wert `1` vorhanden) |
| `created_at` | string (ISO 8601) | Erstellungsdatum |
| `index_date` | string / null | Indexdatum (Erstvorstellung) |
| `hasClinicalBaseData` | integer (0/1) | Klinische Basisdaten vorhanden (466× ja, 267× nein) |
| `completeNessClinicalBaseData` | integer (0–100) | Vollständigkeit Basisdaten in % |
| `numberClinicalEntries` | integer | Anzahl klinischer Einträge |
| `completeNessClinicalEntries` | integer (0–100) | Vollständigkeit Einträge in % |
| `hasICFxEntry` | integer (0/1) | ICF-Eintrag vorhanden |
| `progresstag` | array | Fortschritts-Tags (meist leer) |

---

## 6. `measure_dataset` – Messungen

Jedes Bild kann **0 bis 4** Messungen enthalten:

| Anzahl Messungen | Bilder |
|---|---|
| 0 | 309 |
| 1 | 96 |
| 2 | 142 |
| 3 | 243 |
| 4 | 2 |

### Struktur einer Messung

```json
{
  "id": 1284,
  "uuid": "31484ec5-a857-4460-93d5-4f9b0faf13fd",
  "referring_image": "c057b7b7-397a-4572-8a78-a9d4fef72c4e",
  "created_at": "2025-10-01T10:05:00Z",
  "last_modified": "2025-10-01T10:05:10Z",
  "protocol_img": "<signed-GCS-URL>",
  "user": { ... },
  "data": {
    "xray_measurement_data": { ... }
  }
}
```

| Feld | Typ | Beschreibung |
|---|---|---|
| `id` | integer | Messungs-Primärschlüssel |
| `uuid` | string (UUID) | Messungs-UUID |
| `referring_image` | string (UUID) | UUID des zugehörigen Bildes |
| `created_at` | string (ISO 8601) | Erstellungszeitpunkt |
| `last_modified` | string (ISO 8601) | Letzte Änderung |
| `protocol_img` | string (URL) / null | Signierte URL zum Protokollbild (JPEG); bei 6 Messungen `null` |
| `user` | object | Messender Nutzer (selbe Struktur wie `upload_user`) |
| `data` | object | Enthält `xray_measurement_data` (§7) |

---

## 7. `data.xray_measurement_data` – Messdaten

Das zentrale Datenobjekt mit vier Unterfeldern:

```json
{
  "angles": [ ... ],
  "calibration": { ... },
  "globalParams": { ... },
  "stateObject": { ... }
}
```

### 7.1 `angles` – Gemessene Winkel

Array von Cobb-Winkel-Objekten. **7.440 Winkel** gesamt (Ø 6,7 pro Messung).

```json
{
  "cobbType": "Spline",
  "cobbAngle": {
    "angle": 12,
    "vertebra_upper": "T10",
    "vertebra_lower": "T11"
  },
  "lineColor": "#FF5733",
  "upperLine": {
    "x1": 581.0,
    "x2": 561.9,
    "y1": 71.0,
    "y2": 65.0
  },
  "lowerLine": {
    "x1": 565.9,
    "x2": 548.6,
    "y1": 99.0,
    "y2": 89.1
  }
}
```

#### `cobbType` – Winkeltypen

| cobbType | Anzahl | Bedeutung |
|---|---|---|
| `Cobb-segmental` | 4.183 (56 %) | Segmentaler Cobb-Winkel (je zwei benachbarte Wirbelkörper) |
| `Spline` | 2.004 (27 %) | Aus B-Spline-Rekonstruktion abgeleiteter Winkel |
| `Cobb-regional` | 1.253 (17 %) | Regionaler Cobb-Winkel (über mehrere Segmente) |

#### `cobbAngle` Felder

| Feld | Typ | Beschreibung |
|---|---|---|
| `angle` | integer (0–89°) | Cobb-Winkel in Grad; Ø = 15,0°, Max = 89° |
| `vertebra_upper` | string | Oberer Wirbelkörper (Wirbelsäulenlabel, s. §8) |
| `vertebra_lower` | string | Unterer Wirbelkörper |

**Häufigste Wirbelkörper (`vertebra_upper`):** L3 (1228×), T12 (1149×), L2 (1022×), L4 (1021×), L1 (1005×)  
**Häufigste Wirbelkörper (`vertebra_lower`):** S1 (2333×), L5 (1317×), L4 (1305×), L3 (997×), L2 (776×)

#### `upperLine` / `lowerLine` – Linienkoordinaten

Pixel-Koordinaten im Bildkoordinatensystem:

| Variante | Felder | Vorkommen |
|---|---|---|
| Basis (Spline) | `x1`, `x2`, `y1`, `y2` | 2.004 Winkel |
| Mit Mode (Cobb) | `x1`, `x2`, `y1`, `y2`, `mode` | 5.436 Winkel |

`mode` Werte in `upperLine`/`lowerLine`: `Cobb-regional`, `Cobb-segmental`

#### `lineColor` – Farbkodierung

| Farbe | Hex | Anzahl |
|---|---|---|
| Orange-Rot | `#FF5733` | 1.919 |
| Gold | `#FFD700` | 1.105 |
| Blau | `#33A8FF` | 1.101 |
| Lila | `#8000FF` | 1.071 |
| Grün | `#28A745` | 1.060 |
| Pink | `#FF1493` | 549 |
| Cyan | `#14FFCE` | 293 |
| Tiefrot | `#FF4500` | 162 |
| Himmelblau | `#00BFFF` | 91 |
| Blauviolett | `#8A2BE2` | 49 |

---

### 7.2 `calibration` – Kalibrierungsobjekt

Vorhanden bei **1.062** von 1.117 Messungen (55 fehlen).

```json
{
  "length": 87,
  "target": "L3"
}
```

| Feld | Typ | Beschreibung |
|---|---|---|
| `length` | integer | Kalibrierungslänge in Pixeln (Bereich: 12–310, Ø = 66,2) |
| `target` | string | Referenzwirbel für die Kalibrierung |

**Häufigste `target`-Wirbelkörper:** L4 (636×), L3 (270×), L2 (58×), L5 (44×), L1 (29×)

---

### 7.3 `globalParams` – Beckenparameter

Vorhanden bei **665** von 1.117 Messungen. Enthält globale spinopelvine Parameter:

```json
{
  "pelvic_incidence": 30,
  "hipline_angle_to_ground": 63,
  "hipcenter_s1_distance_x_raw": 127,
  "hipcenter_s1_distance_x_relative": 45,
  "hipcenter_c7_distance_x_raw": 210,
  "hipcenter_c7_distance_x_relative": 38
}
```

| Feld | Einheit | Vorkommen | Beschreibung |
|---|---|---|---|
| `pelvic_incidence` | Grad | 664× | Pelvic Incidence (PI) |
| `hipline_angle_to_ground` | Grad | 664× | Winkel der Hüftlinie zur Horizontalen |
| `hipcenter_s1_distance_x_raw` | Pixel | 664× | Horizontaler Abstand Hüftmitte–S1 (roh) |
| `hipcenter_s1_distance_x_relative` | (relativ) | 100× | Horizontaler Abstand Hüftmitte–S1 (kalibriert) |
| `hipcenter_c7_distance_x_raw` | Pixel | 41× | Horizontaler Abstand Hüftmitte–C7 (roh) |
| `hipcenter_c7_distance_x_relative` | (relativ) | 5× | Horizontaler Abstand Hüftmitte–C7 (kalibriert) |

---

### 7.4 `stateObject` – Annotationszustand

Enthält die interaktive Annotationsebene der Messung:

```json
{
  "vertebra_range": [16, 25],
  "nodes": [ ... ],
  "suggestedNodes": [ ... ],
  "cobbLines": [ ... ],
  "calibrationLine": { ... }
}
```

| Feld | Typ | Beschreibung |
|---|---|---|
| `vertebra_range` | array[2] | Start- und End-Index im Wirbelsäulenlabel-Array (§8) |
| `nodes` | array | Manuell gesetzte (ggf. nachbearbeitete) Stützpunkte des Splines (§7.4.1) |
| `suggestedNodes` | array \| null | Snapshot der vom ONNX-Modell vorgeschlagenen Nodes zum Zeitpunkt des Suggest-Buttons; `null` wenn Feature nicht genutzt (§7.4.1) |
| `cobbLines` | array | Eingezeichnete Cobb-Linien (§7.4.2) |
| `calibrationLine` | object | Kalibrierungslinie im Bild (§7.4.3) |

#### `vertebra_range` Verteilung

- **Start-Wirbel:** T10 (315×), T11 (274×), T9 (169×), T12 (124×), T8 (77×), C7 (38×)
- **End-Wirbel:** S1 (1.098×), L5 (12×), L4 (3×)
- **Spannweite:** 5–26 Wirbelkörper, Ø = 9,6

#### 7.4.1 `nodes` – Spline-Stützpunkte

Manuell platzierte Kontrollpunkte für die B-Spline-Rekonstruktion.

```json
{
  "x": 586,
  "y": 22,
  "fill": "blue",
  "type": "Circle",
  "radius": 4
}
```

| Feld | Typ | Beschreibung |
|---|---|---|
| `x` | integer | X-Koordinate im Bildpixelraum |
| `y` | integer | Y-Koordinate (Ursprung oben links) |
| `fill` | string | Darstellungsfarbe (`"blue"`) |
| `type` | string | Geometrietyp (`"Circle"`) |
| `radius` | integer | Darstellungsradius in Pixeln (`4`) |

**Nodes pro Messung:** min=0, max=25, Ø=9,5  
Häufigste Anzahl: 9 Nodes (308×), 8 Nodes (279×), 10 Nodes (168×), 7 Nodes (129×)

#### 7.4.2 `cobbLines` – Cobb-Linien im Annotationszustand

```json
{
  "x1": 437.0,
  "x2": 676.0,
  "y1": 183.0,
  "y2": 225.0,
  "mode": "Cobb-regional"
}
```

| Feld | Typ | Beschreibung |
|---|---|---|
| `x1`, `y1` | float | Startpunkt der Linie (Pixel) |
| `x2`, `y2` | float | Endpunkt der Linie (Pixel) |
| `mode` | string / null | `"Cobb-regional"`, `"Cobb-segmental"`, oder fehlt |

**Verteilung:** `Cobb-segmental` (8.427×), `Cobb-regional` (2.349×), ohne `mode` (1.603×)

#### 7.4.3 `calibrationLine` – Kalibrierungslinie

```json
{
  "x1": 561.0,
  "x2": 637.0,
  "y1": 338.0,
  "y2": 294.0
}
```

Liniensegment über dem Referenzwirbel (`calibration.target`) für die Längenkalibrierung.

---

## 8. Wirbelsäulenlabels (Referenz)

Die `vertebra_range`-Indizes referenzieren dieses geordnete Label-Array:

```python
spine_labels = [
  'C0','C1','C2','C3','C4','C5','C6','C7',   # Index 0–7   (Halswirbelsäule)
  'T1','T2','T3','T4','T5','T6','T7',          # Index 8–14  (Brustwirbelsäule)
  'T8','T9','T10','T11','T12',                 # Index 15–19
  'L1','L2','L3','L4','L5',                   # Index 20–24 (Lendenwirbelsäule)
  'S1'                                         # Index 25   (Kreuzbein)
]
```

Typischer Messbereich: **T10 (Index 17) bis S1 (Index 25)**

---

## 9. Datenqualität & Besonderheiten

### Leere Messungen
3 Messungen haben `nodes: []` und `angles: []` – diese repräsentieren abgebrochene oder leere Annotierungen.

### Labeling-Fehler (Vertebra Shift)
Wie im Notebook `reconstructing_spline.ipynb` analysiert: Ein signifikanter Anteil der Messungen weist einen **systematischen Vertebra-Label-Shift** auf. Ursache ist das Verrutschen beim manuellen Annotieren.
- Erkennungsmethode: k-Means Clustering der Node-Positionen + Vergleich mit Original-Labels
- Original-Std der Node-Positionen: Ø **8,0 px**
- Nach k-Means Relabeling: Ø **2,1 px**
- Korrekturdatei: `data/label_errors.json`

### URL-Ablauf
Alle `img`-, `thumb`- und `protocol_img`-URLs sind **Google Cloud Storage Signed URLs** mit einer Gültigkeit von **24 Stunden** (Parameter `X-Goog-Expires=86400`).

### Spline-Rekonstruktion
Die `stateObject.nodes` werden in der Anwendung (`bspline_python.py`) zu einem B-Spline interpoliert. Daraus werden:
- Tangentenlinien an den Node-Positionen
- Perpendikularlinien zur Winkelberechnung
- Segmentale und regionale Cobb-Winkel (`reconstructedAngles`)

---

## 10. Vollständiges Schema (Zusammenfassung)

```
ImageData[]
├── id: int
├── uuid: UUID
├── img: str (signed URL, 24h)
├── thumb: str (signed URL, 24h)
├── img_type: str
├── acquisition_date: str | null
├── patient_id: str
├── patient_id_integrity_check: bool
├── public: bool
├── created_at: datetime
├── last_modified: datetime
├── image_hash: str
├── upload_user:
│   ├── id: int
│   ├── first_name: str
│   ├── last_name: str
│   ├── email: str
│   └── username: str
├── referring_patient:
│   ├── id: int
│   ├── uuid: UUID
│   ├── patient_id: str
│   ├── institution: int
│   ├── created_at: datetime
│   ├── index_date: str | null
│   ├── hasClinicalBaseData: int (0|1)
│   ├── completeNessClinicalBaseData: int
│   ├── numberClinicalEntries: int
│   ├── completeNessClinicalEntries: int
│   ├── hasICFxEntry: int (0|1)
│   └── progresstag: list
└── measure_dataset[]:
    ├── id: int
    ├── uuid: UUID
    ├── referring_image: UUID
    ├── created_at: datetime
    ├── last_modified: datetime
    ├── protocol_img: str | null (signed URL, 24h)
    ├── user: { id, first_name, last_name, email, username }
    └── data:
        └── xray_measurement_data:
            ├── angles[]:
            │   ├── cobbType: "Spline" | "Cobb-regional" | "Cobb-segmental"
            │   ├── lineColor: str (hex)
            │   ├── cobbAngle:
            │   │   ├── angle: int (0–89)
            │   │   ├── vertebra_upper: str (spine_label)
            │   │   └── vertebra_lower: str (spine_label)
            │   ├── upperLine: { x1, x2, y1, y2 [, mode] }
            │   └── lowerLine: { x1, x2, y1, y2 [, mode] }
            ├── calibration:
            │   ├── length: int
            │   └── target: str (spine_label)
            ├── globalParams:
            │   ├── pelvic_incidence: int
            │   ├── hipline_angle_to_ground: int
            │   ├── hipcenter_s1_distance_x_raw: int
            │   ├── hipcenter_s1_distance_x_relative?: int
            │   ├── hipcenter_c7_distance_x_raw?: int
            │   └── hipcenter_c7_distance_x_relative?: int
            └── stateObject:
                ├── vertebra_range: [int, int]
                ├── nodes[]: { x, y, fill, type, radius }
                ├── suggestedNodes[]: { x, y, fill, type, radius } | null
                ├── cobbLines[]: { x1, x2, y1, y2 [, mode] }
                └── calibrationLine: { x1, x2, y1, y2 }
```
