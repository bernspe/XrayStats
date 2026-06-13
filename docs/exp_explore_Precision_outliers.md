# Explore Annotation Precision Outliers
## Ziel
 - Identifizieren von Images, die eine hohe Abweichung zwischen den Messungen eines cobbTypes aufweisen
 - Anzeige der Bilder und Overlay der Nodes & abgeleiteten Messungen, um optisch Probleme zu identifizieren

## Grundlagen
 - [exp_Annotation_correctness.md](exp_Annotation_correctness.md)
 - [annotation_correctness.ipynb](../annotation_correctness.ipynb) und importierte Funktionen

## Parameter
 - Winkelwerte aus cobbType==Spline, Cobb_regional und Cobb_segmental aus measure_dataset.data.xray_measurement_data.angles[]:
 - k-Means-korrigierte Cobb-Winkel
 - inflection-point-Cobb-Winkel mit verschiedenen Thresholds (min_angle=5,10,15°)

## Vorgehensweise
1. Berechnung der Standardabweichung der Cobb-Winkel pro Image und cobbType
2. Ordnen der Images nach Standardabweichung, um die Top-Outlier zu identifizieren
3. Visualisierung der Images mit ihren Nodes und Cobb-Winkeln als interaktive Plots (z.B. mit Plotly oder Matplotlib) mit der Möglichkeit, aus der geordneten Liste auszuwählen
4. Manuelle Inspektion der Top-Outlier, um Muster zu erkennen

---

## Implementierung (2026-06-13)

### Architektur: Hybrid-Ansatz (Python-Script + Streamlit-App)

#### Neue Dateien

| Datei | Zweck |
|---|---|
| `prepare_outlier_data.py` | Einmaliges Datenabruf-Script: API-Request → Pipeline → Pickle-Export |
| `outlier_app.py` | Streamlit Browser-App: interaktiver Viewer |
| `annotation_analysis.py` | Ergänzt um `compute_outlier_ranking()` |

#### Datenfluss

```
python prepare_outlier_data.py --token TOKEN
  └─ GET /api/imagedata/ (860 Bilder, 1137 Messungen)
       ├─ build_df4d() → df4d
       ├─ detect_label_errors() → 73 Shift-Fehler, 6 leere Messungen
       ├─ apply_label_corrections() → df4d_corrected
       ├─ build_angles_df() × 4 (base + inflection @5°/10°/15°) → angles_df
       │    → 24.046 Zeilen, 33 cobbTypes
       ├─ compute_outlier_ranking() → outlier_df
       │    → 5.668 Winkelgruppen mit ≥2 Ratern, sortiert nach σ absteigend
       └─ data/*.pkl + data/img_urls.pkl  (503 Google-signierte URLs, 24 h gültig)
```

#### `compute_outlier_ranking()` (annotation_analysis.py)

Gruppiert `angles_df` nach `(image_id, cobbType, vertebra_upper, vertebra_lower)`, berechnet Inter-Rater-Std und filtert auf ≥2 Rater. Gibt DataFrame sortiert nach `std` absteigend zurück.

#### cobbType-Namenskonvention für Inflection-Winkel

`inflection_r{rank}@{threshold}` — z.B. `inflection_r1@10` = Rang 1 bei min_angle=10°.

---

### Streamlit-App (`outlier_app.py`)

**Start:**
```bash
cd /Users/peterbernstein/Python/XrayStats
/Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run outlier_app.py
```

**Funktionen:**
- Sidebar: cobbType-Selector (33 Typen) + Top-N-Slider (5–200)
- Dropdown: Bilder sortiert nach max. σ (z.B. `Cobb-regional  ID 556  σ_max=14.2°`)
- Linkes Panel: X-Ray mit korrektem Overlay (Spline + Nodes + CobbLines)
- Rechtes Panel: Winkelpaare (σ absteigend) + Einzelmessungen pro Rater

**Performance:** Pickles via `@st.cache_resource` (einmalig geladen), Bilder via `@st.cache_data(url)` (pro URL gecacht).

---

### Koordinaten-Korrektur (wichtige Erkenntnis)

Der xray-annotator (KonvaImage.vue) skaliert jedes Bild auf eine **feste Stage-Höhe von 600px**:

```javascript
// KonvaImage.vue
const stageSize = ref({ width: 400, height: 600 })  // height = immer 600px
const target_height_factor = im.naturalHeight / stageSize.value.height
// Bild wird auf height=600 skaliert; alle Koordinaten in diesem Canvas-Raum
```

Alle gespeicherten Node- und CobbLine-Koordinaten (`stateObject.nodes`, `stateObject.cobbLines`, `angles[].upperLine/lowerLine`) sind im Canvas-Koordinatensystem (0..stageWidth × 0..600), **nicht** im Original-Pixelraum.

**Fix beim Overlay auf das Original-Bild:**
```python
scale = xray.height / 600  # z.B. 847/600 = 1.412 für ein 1612×847 Bild
nodes_xy = [(n['x'] * scale, n['y'] * scale) for n in nodes_raw]
```

Ohne diese Korrektur erscheinen Nodes/Linien ~40% zu weit links/oben — mit bild-spezifischem Versatz je nach Originalhöhe, weshalb die Verschiebung pro Bild unterschiedlich aussieht.

