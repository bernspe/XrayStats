# Datenmodell: xray-annotator/API → Verlaufsmodell (Phase 0)

Stand 29.09.2026. Quellen: `docs/api_imagedata_structure.md`, `xray-annotator/src/helpers/measureCompute.ts` und `bspline.ts`, Auswertung der Live-Daten (219 Patienten, 2.758 Messungen). Entscheidungen stehen in `DECISIONS.md`.

## 1. Mapping Spec → vorhandene Daten

| Spec-Begriff | Quelle im Schema | Umsetzung / Einschränkung |
|---|---|---|
| Wirbel-Centroid mit Label | `stateObject.nodes[i]` + `vertebra_range` | Label = `spine_labels[vr[0] + i]`; k-Means-Labelkorrektur (Flag `label_shift_corrected`) |
| Aufnahmetyp | `ImageData.img_type` | LWS/WS/BWS × ap/lat/incl/recl, MRT cor/sag |
| stehend / liegend | nicht vorhanden | abgeleitet: Xray = stehend, incl/recl = Funktionsaufnahme, MRT = liegend |
| Hüftkopfzentren | `stateObject.hipLine` (x1, y1, x2, y2) | beide Kopfzentren als Linienendpunkte, H = Mittelpunkt; **ein einzelner Hüftkopf ist nicht kodierbar** |
| S1-/L1-/L4-Deckplatte | `stateObject.cobbLines` (unbeschriftet, `mode` regional/segmental) | Zuordnung über den Schnittpunkt mit der Centroid-Sehne; **nur die Richtung** ist verwertbar (Cobb-Linien sind im Median 2,1 WKD lang). Fallback ist die Tangente am Node |
| Deckplatten-Endpunkte anterior/posterior | nicht vorhanden (Endpunkte ungeordnet) | Blickrichtung heuristisch: tiefer liegendes S1-Ende = anterior, Lordose als Kontrolle (Flags `orientation_*`) |
| Pixel Spacing | nicht vorhanden | Ersatz: `calibrationLine` = ein Wirbelkörperdurchmesser (L3/L4) → Einheit WKD, Level-Faktoren auf L4 |
| Rater | `measure_dataset[].user` | 1–4 Rater je Bild; Basis für Phase 2 |
| Aufnahmedatum | `ImageData.acquisition_date` | Visite = Patient × Datum |
| Alter / Geschlecht | `/api/clinicalbasedata/` (`birthdate`, `sex`) | Join über `referring_patient.uuid`; Alter < 18 J. = Testeintrag → verworfen |
| Koordinatensystem | Konva-Stage, Bildhöhe = 600, y nach unten | einheitlicher Maßstab `img.height/600`; Patientenkoordinaten über `spine_geom.frames` |
| Koronare Seite | nicht vorhanden | Annahme radiologische Konvention (Flag `side_assumed`) |
| Fraktur / Implantat / Übergangswirbel | nicht vorhanden | nicht kodiert (Übergangswirbel: Nutzerentscheidung „vorerst ignorieren“) |
| Frontend-Parameter | `globalParams` (PI, PT, SS, T1PA, GT je Spline und `_cobb`) | nur zum Vergleich; eigene Werte werden aus dem `stateObject` neu berechnet |

## 2. Vorschlag für Schema- und Metadatenfelder (zur Freigabe, nicht umgesetzt)

Nach Nutzen für das Modell geordnet:

1. **Beschriftete Deckplatten mit geordneten Endpunkten:** `endplates: [{label: 'S1', side: 'sup', anterior: {x, y}, posterior: {x, y}}]`.
   - Das ersetzt die heuristische Zuordnung der Cobb-Linien und die Blickrichtungsheuristik.
   - Es liefert die echte Deckplattenmitte M und die hintere obere S1-Ecke für das SVA.
   - Mindestens S1 sup, L1 sup und L4 sup (für den LDI).
2. **Hüftköpfe einzeln:** `hipHeads: [{side: 'left'|'right'|'unknown', x, y, r?}]` statt `hipLine`. Damit sind auch Aufnahmen mit nur einem sichtbaren Hüftkopf nutzbar. Hüftwerte von Ratern weichen heute in 14 % (seitlich) bzw. 27 % (a.p.) der Fälle grob voneinander ab.
3. **Aufnahme-Metadaten an `ImageData`:**
   - `position`: standing / supine / functional
   - `pixel_spacing_mm` aus DICOM, sofern vorhanden
   - `patient_orientation` aus DICOM (0020,0020) für die Seitenzuordnung
   - `facing` für Seitaufnahmen
4. **Flags je Messung oder Bild:** `fracture_levels`, `fusion_levels` bzw. `implants`, `transitional_vertebra` (L6 / sakralisierter L5 / lumbalisierter S1). Das erklärt PI-Ausreißer und schützt Formvektoren vor Kollaps-Artefakten.
5. **Eindeutiger Kalibrierwirbel:** `calibration.target` verweist heute auf den Wirbel vor der Label-Korrektur, deshalb wird das Level geometrisch bestimmt. Außerdem die Messanweisung „L3 oder L4“ im UI erzwingen: 23 % der Kalibrierungen liegen außerhalb dieser Anweisung, z. B. 271-mal L5 bei einem Rater.
6. **Optional (Spec §4.5):** Deckplattenpunkte an End- und Apexwirbeln a.p., um Kippung und Translation zu trennen, sowie Pedikel-Landmarken für die axiale Rotation.

## 3. Datenlage in Zahlen (Live-Daten 29.09.2026)

- **Umfang:** 219 Patienten, 751 Visiten, 2.758 Messungen.
- **Seitliche Stehaufnahmen:** 185 Patienten mit SS und LL. PI, PT und L1PA sind bei 159 messbar.
- **Globale Parameter:** T4PA und der T4-L1PA-Mismatch sind bei 36 Patienten messbar, T1PA und Global Tilt bei 26, SVA bei 35. Hauptgrund ist der Bildausschnitt: LWS-Aufnahmen reichen nicht bis T4.
- **Koronar:** `centroid_cobb` aus a.p.-Stehaufnahmen liegt bei 176 Patienten vor.
