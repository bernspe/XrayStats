# Untersuchung der Annotationspräzision verschiedener Labelling-Methoden bei Röntgenbildern

## Ziel
- Deskription der Präzision der händischen Spline-Konstruktion (stateObject→nodes) einschließlich Labelling-Fehler
- Quantifizierung des Optimierungs-Wertes einer k-Means-basierten Clusteranalyse zur Identifikation systematischer Labelling-Fehler (Vorher/Nachher-Vergleich)
- Deskription der Messgenauigkeit verschiedener Winkel-Techniken (cobbType = Spline vs. Cobb-regional vs. Cobb-segmental) nach Wirbelsäulenregion bzw. Wirbelkörperpaar

---

## Variablen

### Unabhängige Variablen
- `patient_id`, `img_type`, `measure_dataset_index`
- innerhalb `xray_measurement_data`:
  - `angles.cobbType`
  - `stateObject.cobbLines.mode`

### Abhängige Variablen
- `angles.cobbAngle` (angle °, vertebra_upper, vertebra_lower)
- `stateObject.cobbLines` – line1/line2 Koordinaten (x1, y1, x2, y2)
- `stateObject.nodes` – x, y pro Node
- `stateObject.vertebra_range` → Wirbellabels via `spine_labels`-Array

### Spezielle Variablen
- `stateObject.calibrationLine` – Länge eines Wirbelkörpers in Pixeln (Label aus Koordinaten zu rekonstruieren)


---

## Studiendesign

**Typ:** Observationell, Inter-Rater-Reliabilitätsstudie  
**Rater-Struktur:** Jedes Bild wird von 1–3 verschiedenen Nutzern unabhängig annotiert (immer verschiedene Nutzer pro Bild)  
**Einheit:** Röntgenbild (image `id`) mit ≥2 Messungen von verschiedenen Nutzern  
**Präzisionsmaß:** Intraklassen-Korrelationskoeffizient (ICC), 2-Wege-Zufallseffekte, Einzelmessungen, absolute Übereinstimmung – **ICC(2,1)**  
**Kalibrierung:** Nur Winkel in Grad (°); keine Pixelumrechnung erforderlich  
**Datenbasis:** aktueller Request des image data Endpunkts

---

## Analyseplan

### Schritt 0 – Datenvorbereitung

Datenbasis ist ausschließlich ein **frischer API-Request** zum Analysezeitpunkt. Die Transformations-Pipeline folgt der Methodik aus `reconstructing_spline.ipynb`, ohne auf gespeicherte Pickle-Dateien zurückzugreifen.

**0.1 API-Request**
```python
api_url = 'https://xray-api.renecol.org'
token = '<aktueller Token>'
response = requests.get(
    headers={'Authorization': 'Bearer ' + token},
    url=api_url + '/api/imagedata/',
    params={'detailed': True, 'complete': True}
).json()
df = pd.DataFrame(response)
```

**0.2 Transformation zu `df4d` (Methodik aus `reconstructing_spline.ipynb`)**
- Filtern auf Zeilen mit nicht-leerem `measure_dataset`
- Entschachteln: `measure_dataset` → Long-Format (`df4b`)
- `data`-Spalte auspacken → eine Zeile pro Messung mit Spalte `xray_measurement_data` (`df4d`)
- `user`-Objekt aus `measure_dataset` als separate Spalte `rater_name` (= `first_name`) beibehalten

**0.3 Label-Fehlerkorrektur → `df4d_corrected`**
- k-Means-Clustering der Node-Positionen pro Bild (Methodik aus `reconstructing_spline.ipynb`)
- Shift-Fehler erkennen und `correct_measurement_label_shift()` anwenden
- Fehler-Tabelle (`label_errors`) als DataFrame im Notebook halten – kein Rückgriff auf `data/label_errors.json`

**0.4 Rater-Struktur validieren**
- Prüfen, dass pro Bild alle `measure_dataset`-Einträge von verschiedenen `user.id` stammen
- Bilder mit nur 1 Messung aus der ICC-Analyse ausschließen (können nur für deskriptive Statistik genutzt werden)
- Ergebnis: `eligible_images` – Bilder mit ≥2 Messungen von ≥2 verschiedenen Nutzern

**0.5 Analyse-DataFrames aufbauen**
- **Nodes-DataFrame:** eine Zeile pro (image_id, rater_name, vertebra_label) mit Spalten x, y – je einmal aus `df4d` und `df4d_corrected`
- **Winkel-DataFrame:** eine Zeile pro (image_id, rater_name, cobbType, vertebra_upper, vertebra_lower) mit Spalte `angle` – aus `df4d_corrected` (nach Korrektur Basis für Schritt 3); für Schritt 2 zusätzlich aus `df4d`
- Rekonstruierte Winkel (Perpendicular-Lines) werden im Notebook neu berechnet via `getReconstructedCobbAngles()` aus `bspline_python.py`

---

### Schritt 1 – Deskription der Node-Präzision (stateObject.nodes)

**Forschungsfrage:** Wie konsistent platzieren verschiedene Annotator*innen die Spline-Nodes für denselben Wirbelkörper?

**1.1 Deskriptive Statistik**
- Für jedes Vertebra-Label: Mittelwert und Standardabweichung von x und y (über alle Bilder und Rater)
- Tabelle: `label | x_mean | x_std | y_mean | y_std | n_measurements`
- Heatmap der Std-Werte über die Wirbelkörper (y-Achse = spine_labels, x-Achse = x/y)

**1.2 ICC pro Vertebra-Label**
- ICC(2,1) für x-Koordinate und y-Koordinate getrennt, pro Vertebra-Label
- Nur Bilder einbeziehen, bei denen der jeweilige Wirbelkörper in ≥2 Messungen vorhanden ist
- Ergebnis: ICC-Tabelle mit 95%-Konfidenzintervall pro Label und Koordinate

**1.3 Aggregation nach Wirbelsäulenregion**
- Regionen aus `bspline_python.spine_regions`: CT [0–9], PT [10–12], MT [13–18], TL [19–20], LU [21–25]
- Mittlerer ICC pro Region (gewichtet nach n)
- Balkendiagramm: ICC mit CI-Fehlerbalken pro Region

---

### Schritt 2 – Effekt der k-Means-Labelkorrektur (Vorher/Nachher)

**Forschungsfrage:** In welchem Ausmaß reduziert die k-Means-basierte Relabellierung die Inter-Rater-Varianz der Node-Positionen?

**2.1 Vergleich Std vor/nach Korrektur**
- Std der Node-Positionen (x und y) pro Vertebra-Label: unkorriegiert vs. korrigiert
- Paired-Visualisierung: Linienplot mit Std_original (rot) und Std_corrected (grün) über spine_labels
- Erwartung aus Notebook: Ø Std sinkt von ~8,0 px auf ~2,1 px

**2.2 ICC-Vergleich vor/nach Korrektur**
- ICC(2,1) der Node-Koordinaten: unkorriegiert vs. korrigiert (dieselbe Stichprobe)
- Darstellung als Tabelle und Scatterplot (ICC_original vs. ICC_corrected pro Label)

**2.3 Charakterisierung der Labelling-Fehler**
- Aus dem in Schritt 0.3 erzeugten `label_errors`-DataFrame: Häufigkeit und Richtung der Shift-Fehler
- Histogramm der Shift-Beträge (in Wirbelkörper-Stufen)
- Anteil leerer Messungen (`error: "empty"`) vs. Shift-Fehler

---

### Schritt 3 – Präzision der Winkelmessungen nach Technik

**Forschungsfrage:** Welche Mess-Technik liefert die höchste Inter-Rater-Konsistenz der Cobb-Winkel?

#### 3A – Regionale Winkel (cobbType = `Spline` und `Cobb-regional`)

**Granularität:** Pro Wirbelsäulenregion (CT, PT, MT, TL, LU) und img_type

**3A.1** Für jedes (image_id, vertebra_upper, vertebra_lower)-Paar: alle Winkelwerte der verschiedenen Rater zusammenführen  
**3A.2** ICC(2,1) pro Region und cobbType, nur Paare mit ≥2 Rater-Werten  
**3A.3** Vergleich `Spline` vs. `Cobb-regional` innerhalb derselben Region:
- Tabelle: `Region | ICC_Spline | ICC_Cobb-regional | n_pairs`
- Gruppierten Boxplot: Winkelwerte nach cobbType und Region

#### 3B – Segmentale Winkel (cobbType = `Cobb-segmental`)

**Granularität:** Pro Wirbelkörperpaar (z. B. L3–L4, L4–L5, L5–S1 etc.)

**3B.1** Matching der Winkel über Bilder und Rater nach (vertebra_upper, vertebra_lower)  
**3B.2** ICC(2,1) pro Wirbelkörperpaar, nur wenn ≥2 Rater-Werte vorliegen  
**3B.3** Heatmap der ICC-Werte auf der Wirbelsäule (x-Achse: von-Wirbel, y-Achse: bis-Wirbel)  
**3B.4** Vergleich mit den Perpendicular-Line-Winkeln (= `reconstructedAngles`, frisch berechnet via `getReconstructedCobbAngles()`), sofern die Zuordnung zu denselben Wirbeln möglich ist

#### 3C – Vergleich aller drei Techniken

- Mittlerer ICC und 95%-CI nach cobbType: `Spline` | `Cobb-regional` | `Cobb-segmental`
- Violin-Plot der Winkelwerte pro Technik (um Varianz-Unterschiede sichtbar zu machen)

---

## Statistische Methodik

| Methode | Anwendung | Bibliothek |
|---|---|---|
| ICC(2,1) | Inter-Rater-Präzision aller metrischen Variablen | `pingouin.intraclass_corr()` |
| 95%-KI für ICC | Bootstrap oder analytisch | `pingouin` (integriert) |
| Wilcoxon-Vorzeichen-Rang | Paired-Vergleich ICC vor/nach Korrektur (über Wirbelkörper) | `scipy.stats.wilcoxon` |
| Deskriptive Statistik | Mittelwert, Std, Median, IQR | `pandas.DataFrame.describe()` |

**ICC-Interpretation (Cicchetti 1994):**
- < 0.40 = ungenügend
- 0.40–0.59 = ausreichend
- 0.60–0.74 = gut
- ≥ 0.75 = ausgezeichnet

**Daten-Voraussetzungen:**
- Pro ICC-Berechnung: mindestens 10 Bilder mit ≥2 Rater-Werten für das jeweilige Merkmal
- Bei Zellen mit nur 1 Rater: Ausschluss aus ICC, Einschluss in Deskriptivstatistik

---

## Erwartete Abbildungen / Tabellen

| # | Typ | Inhalt |
|---|---|---|
| Tab. 1 | Tabelle | Node-ICC(x,y) pro Vertebra-Label, vor/nach Korrektur |
| Tab. 2 | Tabelle | Winkel-ICC pro cobbType und Region/Paar |
| Abb. 1 | Linienplot | Node-Std vor/nach k-Means-Korrektur über spine_labels |
| Abb. 2 | Balkenplot | ICC mit CI pro Wirbelsäulenregion (Nodes) |
| Abb. 3 | Histogramm | Shift-Fehler-Verteilung (in Wirbelkörper-Stufen) |
| Abb. 4 | Grouped Boxplot | Cobb-Winkel nach cobbType × Region |
| Abb. 5 | Heatmap | ICC Cobb-segmental nach Vertebra-Paar |
| Abb. 6 | Violin-Plot | Winkel-Verteilung nach Messtechnik (alle 3) |

---

## Implementierung (Reihenfolge)

1. **Datenvorbereitung** (neues Notebook `annotation_correctness.ipynb`)
   - Frischer API-Request mit aktuellem Token (Zelle am Notebook-Anfang)
   - Pipeline-Schritte 0.2–0.5 analog zu `reconstructing_spline.ipynb`, aber inline im neuen Notebook
   - Kein Laden von Pickle-Dateien; alle Zwischenstände als Notebook-Variablen

2. **Schritt 1**: Node-ICC mit `pingouin`, Visualisierung

3. **Schritt 2**: Vorher/Nachher-Vergleich, Auswertung des `label_errors`-DataFrames

4. **Schritt 3**: Winkel-ICC nach cobbType, erst regional, dann segmental

5. **Export**: Tabellen als CSV in `output/`, Abbildungen als PNG in `output/`


---

## Ergebnisse

Analyse auf Basis des API-Requests vom 2026-06-13 (860 Bilder, 1117 Messungen).
387 Bilder mit ≥2 verschiedenen Ratern (eligible für ICC).

### Schritt 1 – Node-Präzision

| Koordinate | ICC(2,1) nach Korrektur |
|---|---|
| x | ≈ 0.999 |
| y | ≈ 0.974 |

Nodes sind nach Labelkorrektur hoch konsistent über alle Rater.

### Schritt 2 – Effekt der k-Means-Labelkorrektur

- **73 Shift-Fehler** (Rater hat Nodes um 1–2 Wirbelkörper verschoben) und **5 leere Messungen** erkannt.
- Ø Inter-Rater-Std der Node-Positionen sinkt von **~8 px → ~2 px** nach Korrektur.
- Wilcoxon-Test bestätigt signifikante ICC-Verbesserung durch Korrektur.

### Schritt 3 – ICC Winkelmessungen nach Technik

| Messtechnik | ICC(2,1) gesamt | Bewertung |
|---|---|---|
| Cobb-regional | **≈ 0.82** | ausgezeichnet (beste Methode) |
| Cobb-segmental | ≈ 0.65–0.70 | gut |
| Spline | ≈ 0.54 | ausreichend |
| reconstructed_regional | ≈ 0.50 | ausreichend |
| reconstructed_segmental | ≈ 0.55 | ausreichend |

`Cobb-regional` ist die beste Methode, da Rater explizit dieselben Endplatten einzeichnen (manuelle Geraden) – unabhängig von der Spline-Kurvenform. `Spline` und `reconstructed_regional` zeigen trotz hoher Node-ICC schlechtere Winkel-ICC, da Rater verschiedene Kurvenabschnitte messen (siehe Schritt 4).

### Schritt 4 – Endpunkt-Variabilität

- **Jaccard-Übereinstimmung** der gemessenen Winkelpaare: Spline **17 %**, Cobb-regional **24 %** vollständige Übereinstimmung. Die Endpunkt-Wahl ist der Haupttreiber der Winkel-ICC-Verringerung.
- **ICC vr_start** (obere Messbereichsgrenze): ≈ 0.916 – Rater stimmen im oberen Startpunkt gut überein.
- **ICC vr_end** (untere Messbereichsgrenze): ≈ 0.237 – trotz faktisch hoher Konsistenz (fast alle wählen S1) zeigt der ICC einen niedrigen Wert, da die geringe Bild-zu-Bild-Varianz den Quotienten im ICC-Modell gegen 0 drückt.

### Schritt 5 – Inflektionspunkt-Methode

Geometrisch abgeleitete Winkel aus lokalen Tangenten-Extrema der Spline. Schwellwert `MIN_INFLECTION_ANGLE` = 10°.

| Rang | ICC(2,1) bei 5° | ICC(2,1) bei 10° |
|---|---|---|
| inflection_r0 | ≈ 0.45 | ≈ 0.55 |
| inflection_r1 | ≈ 0.61 | **≈ 0.80** |

Die Inflektionsmethode erreicht bei `min_angle = 10°` für `inflection_r1` eine ausgezeichnete ICC (≈ 0.80), übertrifft aber `Cobb-regional` (0.82) nicht. Hauptlimitation: Die positionale Rang-Nummerierung führt bei unterschiedlichen Messbereichen zu Rang-Verschiebungen – `inflection_r0` messen dann verschiedene Rater an verschiedenen anatomischen Positionen.
