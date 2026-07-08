# stats_app — Statistische Methoden und Abbildungslogik

Dieses Dokument beschreibt alle statistischen Verfahren und Visualisierungslogiken der
Streamlit-App `stats_app.py`.

---

## Datenbasis

Die App liest `data/outlier_angles_df.pkl`, das von `prepare_outlier_data.py` erzeugt wird.
Jede Zeile entspricht einem einzelnen Winkelwert:

| Spalte | Bedeutung |
|--------|-----------|
| `image_id` | Bild-ID |
| `rater_name` | Annotator/Rater |
| `cobbType` | Art der Winkelmessung (siehe unten) |
| `angle` | Winkelwert in Grad |
| `vertebra_upper` / `vertebra_lower` | Grenzwirbel des gemessenen Segments |
| `img_type` | Bildtyp (z.B. AP, lateral) |
| `main_region` | Spine-Region (abgeleitet via `convertMeasurementBorderToSpineRegion`) |

### cobbType-Übersicht

| cobbType | Quelle | Gauss-invariant? |
|----------|--------|-----------------|
| `Cobb-regional` | API (User-Messung, regional) | ja |
| `Cobb-segmental` | API (User-Messung, segmental) | ja |
| `Spline` | API (User-Messung, aus Spline-Kurve) | nein (ab σ > 0) |
| `reconstructed_regional` | vollautomatisch aus Spline-Tangenten | nein |
| `reconstructed_segmental` | vollautomatisch, nur Lumbalsegmente | nein |
| `inflection_r*@T` | Inflektionspunkt-Methode, Schwelle T° | nein |

---

## Spine-Regionen

Die Zuordnung eines Winkelwerts zu einer Spine-Region erfolgt über
`convertMeasurementBorderToSpineRegion(vertebra_upper, vertebra_lower)` aus
`bspline_python.py`. Die Funktion berechnet den maximalen Überlapp des Vertebra-Paars
`[upper_idx, lower_idx]` mit jedem Regionsintervall und gibt die Region mit dem größten
Überlapp zurück.

```
spine_regions = {
    'CT':   [0,  9],  # C0–T2
    'PT':   [9,  12], # T2–T5  (Überlapp T2)
    'MT':   [12, 18], # T5–T11 (Überlapp T5)
    'TL':   [18, 20], # T11–L1 (Überlapp T11)
    'U-LU': [20, 22], # L1–L3  (Überlapp L1)
    'L-LU': [22, 25], # L3–S1  (Überlapp L3)
}
```

Die Grenzen **überlappen um je einen Wirbel**: der Grenzwirbel gehört zu beiden
angrenzenden Regionen. Damit löst `convertMeasurementBorderToSpineRegion` (Max-Overlap-
Algorithmus) Grenzfälle eindeutig auf — ein Segment genau an der Grenze (z.B. T11/T12)
bekommt in der caudaleren Region Overlap 2, in der cranialen nur 1 und wird so ohne
Gleichstand korrekt zugewiesen. Ohne Überlapp wäre der Overlap beidseitig 1 und die
Zuweisung von der Dict-Reihenfolge abhängig.

Die Region **LU** (L1–S1 gesamt) existiert nicht in `spine_regions`, weil sie sich mit
U-LU und L-LU überschneiden würde. LU wird synthetisch berechnet (→ Abschnitt ICC).

---

## Tab 1 — Deskriptive Statistik

### Verfahren

Für jede Kombinationen der gewählten Gruppierungsvariablen (cobbType, Bildtyp, Rater,
Region) werden folgende Kennwerte berechnet:

| Kenngröße | Funktion |
|-----------|----------|
| n | Anzahl der Winkelwerte |
| Mittelwert | arithmetisches Mittel |
| Std | Standardabweichung (ddof=1) |
| Median | 50. Perzentil |
| Min / Max | Extremwerte |

Basis: `pandas.GroupBy.agg`.

### Boxplot (`make_boxplot`)

- **X-Achse:** erste Gruppierungsvariable
- **Farbe/Gruppen:** optionale zweite Gruppierungsvariable (Hue)
- Gruppierte Boxen bei zwei Variablen: gleichmäßiger Offset je Hue-Gruppe, Breite
  `0.8 / n_hue`
- `patch_artist=True`, Füllung per Tab10-Palette, α = 0.7
- Grenzwert: mehr als 80 Gruppen → kein Plot (zu unübersichtlich)

---

## Tab 2 — ICC (Inter-Rater-Reliabilität)

### ICC(2,1) — Zwei-Wege-ANOVA (Shrout & Fleiss 1979)

**Kernproblem:** Jedes Bild wird von 2–4 verschiedenen Ratern annotiert, aber nicht von
allen. Ein vollständig ausgefüllter Rater×Bild-Block ist nicht vorhanden. `pingouin.
intraclass_corr()` setzt balancierten Datensatz voraus — fehlende Zellen führen dort zu
vollständigem Datenverlust.

**Lösung:** Direkte Ableitung aus der Zwei-Wege-ANOVA nach Shrout & Fleiss (1979),
Formel ICC(2,1), die unbalancierte Designs korrekt behandelt:

```
model:  y_ij = μ + α_i (Bild) + β_j (Rater) + ε_ij

MS_B   = mittleres Quadrat zwischen Bildern
MS_R   = mittleres Quadrat zwischen Ratern
MS_W   = Residual (Within/Error)
k₀     = harmonisches Mittel der Rater-Anzahl pro Bild

ICC(2,1) = (MS_B − MS_W) / (MS_B + (k₀−1)·MS_W + (k₀/n)·(MS_R − MS_W))
```

Das OLS-Modell wird mit `statsmodels.formula.api.ols("_y ~ C(_target) + C(_rater)")` 
geschätzt; die ANOVA-Tabelle liefert die Mean Squares.

**95%-Konfidenzintervall** (Shrout & Fleiss 1979):

```
F_L = (MS_B / MS_W) / F_{0.975}(df_B, df_W)
F_U = (MS_B / MS_W) · F_{0.975}(df_W, df_B)

ICC_L = (F_L − 1) / (F_L + k₀ − 1)
ICC_U = (F_U − 1) / (F_U + k₀ − 1)
```

**Mindestanforderung:** ≥ 10 Bilder mit je ≥ 2 Ratern. Andernfalls ICC = NaN.

**Interpretation:**

| ICC | Bedeutung |
|-----|-----------|
| ≥ 0.75 | gut–exzellent (grün) |
| 0.50–0.74 | moderat–gut (gelb) |
| < 0.50 | schwach (rot) |

---

### Regionale ICC (`compute_icc_table`)

**Problem ohne Aggregation:** Ein Rater kann für dasselbe Bild in derselben Spine-Region
mehrere Winkel haben (z.B. zwei Cobb-regional-Werte im MT-Bereich). Das verletzt die
ANOVA-Voraussetzung einer einzelnen Beobachtung pro Rater×Bild-Zelle und führt zu
verfälschtem (erhöhtem) ICC.

**Lösung — max-Aggregation vor ICC:**

```
agg = df.groupby([image_id, img_type, rater_name, cobbType, main_region])["angle"].max()
```

Für jede Kombination aus Bild, Rater, cobbType und Region wird der maximale Winkel
behalten. Damit hat jede Zelle exakt eine Beobachtung.

**Synthetische LU-Region:**

Da LU nicht in `spine_regions` definiert ist, wird sie nach der max-Aggregation
synthetisch erzeugt:

```
LU_angle = max(U-LU_angle, L-LU_angle)  je (Bild, Rater, cobbType)
```

Diese LU-Zeilen werden per `pd.concat` angehängt. Damit erscheint LU in der Tabelle
neben U-LU und L-LU, ohne die Überlapp-Logik in `convertMeasurementBorderToSpineRegion`
zu stören.

**Sortierung:** Regionen werden in anatomischer Reihenfolge sortiert:
`CT → PT → MT → TL → U-LU → L-LU → LU`.

**Gilt für:** `Cobb-regional`, `Spline`, `reconstructed_regional`

---

### Segmentale ICC (`compute_icc_by_segment`)

Segmentale Cobb-Winkel beziehen sich auf ein einzelnes Bewegungssegment (z.B. L1/L2).
Jeder Rater hat pro Bild und Segment maximal einen Winkel → keine Aggregation nötig.

Die Funktion gruppiert direkt nach `(vertebra_upper, vertebra_lower)` und berechnet
ICC(2,1) für jedes Paar separat.

**Eingeschränkter Bereich:** Nur lumbale Segmente werden ausgewertet:

```
LUMBAR_SEGMENT_PAIRS = {T12/L1, L1/L2, L2/L3, L3/L4, L4/L5, L5/S1}
```

Segmente oberhalb T12 werden ausgeschlossen, da dort keine segmentalen Messdaten
vorliegen.

**Sortierung:** Vertebra-Paare werden nach Wirbelsäulenposition (Spine-Label-Index)
aufsteigend sortiert.

**Gilt für:** `Cobb-segmental`, `reconstructed_segmental`

---

### Medianer Inter-Rater-Std

Ergänzend zum ICC wird für jede Gruppe der **mediane Standardabweichung über Rater**
berechnet:

```
median_std = median( std(angles) per image_id )
```

Das heißt: für jedes Bild wird die Streuung der Rater-Winkel berechnet, dann der Median
über alle Bilder. Die Einheit ist Grad (°). Ein niedriger Wert bei gleichzeitig hohem ICC
zeigt hohe Reproduzierbarkeit an.

---

### ICC-Diagramm (`make_icc_chart`)

**Typ:** Gruppenbalkendiagramm mit zweiter Y-Achse

| Element | Bedeutung |
|---------|-----------|
| Balken (linke Y-Achse) | ICC(2,1) pro cobbType und Region/Segment |
| Diamant-Marker ◆ (rechte Y-Achse) | medianer Inter-Rater-Std (°) |
| X-Achse | Spine-Region bzw. Vertebra-Paar |
| Farbe | cobbType (Tab10-Palette) |
| Gestrichelte Linien | ICC-Schwellen 0.75 (gut) und 0.50 (moderat) |

Pro `img_type` ein Subplot nebeneinander.

Fehlende ICC-Werte (NaN) erzeugen keinen Balken — keine versteckte Lücke.

Legende: oben zentriert, `bbox_to_anchor=(0.5, 1.12)`, damit keine Überlappung mit
den Subplots.

**Lumbar-Toggle:** U-LU, L-LU und LU können unabhängig ein-/ausgeblendet werden,
ohne die ICC-Berechnung neu anzustoßen.

---

## Tab 3 — Korrelation

### Pearson-Korrelation

Pivot: `(image_id, rater_name, vertebra_upper, vertebra_lower)` als Zeilen,
`cobbType` als Spalten, `angle` als Werte (`aggfunc="first"`).

```
corr = pivot.corr(min_periods=10)
```

Nur Paare mit ≥ 10 gemeinsamen Beobachtungen fließen ein; weniger ergibt NaN.

**Heatmap (`make_corr_heatmap`):**

- Colormap `RdYlGn`, Skala −1 bis +1
- Pearson-r-Wert als Text in jeder Zelle
- Weiße Schrift bei |r| ≥ 0.8 (dunkle Farbe), schwarze Schrift sonst
- Colorbar-Label: „Pearson r"

---

## Tab 4 — Sigma-Vergleich

Ziel: Einfluss der Gauss-Glättung der Wirbelsäulenknoten (σ = 0 … 3.0) auf die
rekonstruierten Winkel und die ICC-Reliabilität.

**Gauss-Glättung:** Vor der Spline-Rekonstruktion werden die Knoten-Koordinaten mit
einem 1D-Gauss-Filter (scipy, `sigma` Pixel) geglättet. σ = 0 bedeutet keine Glättung.
`Cobb-regional`, `Cobb-segmental` und `Spline` werden vom API gespeichert und sind
σ-invariant; nur `reconstructed_*` und `inflection_*` ändern sich.

Jeder σ-Datensatz wird per `@st.cache_data` gecacht: die erste Berechnung dauert ~15 s,
danach ist der Wert sofort verfügbar.

### Deskriptive Vergleichstabelle (`make_sigma_table`)

Für jeden σ-Wert: Mittelwert, Std, Median pro cobbType. Spalten: `σ=X Ø`, `σ=X Std`,
`σ=X Md`. Zeilen: cobbType (optional + weitere Gruppierungsspalten).

### Sigma-Boxplot (`make_sigma_boxplot`)

- Ein Subplot pro cobbType, X = σ-Werte, Boxen farblich (Viridis-Skala, σ-abhängig)
- `Cobb-regional`, `Cobb-segmental`, `Spline` erhalten den Hinweis „API-gespeichert,
  σ-invariant" als X-Achsenbezeichnung
- `sharey=True`: alle Subplots teilen die Y-Achse für direkte Vergleichbarkeit

### ICC nach σ (`make_sigma_icc` + `make_sigma_icc_chart`)

**Berechnung:** Identisch zum ICC-Tab — kein Pooling über Regionen.

- Regionale cobbTypes → `compute_icc_table` mit `breakdown_col="img_type"`
- Segmentale cobbTypes → `compute_icc_by_segment`, gefiltert auf `LUMBAR_SEGMENT_PAIRS`

Ausgabe: lange Tabelle mit Spalten `σ, img_type, cobbType, Bereich, ICC, CI95, Median Std, n Bilder`.

**3D-Balkendiagramm (`make_sigma_icc_chart`):**

| Achse | Bedeutung |
|-------|-----------|
| X (Breite) | Spine-Region bzw. Vertebra-Segment (anatomisch sortiert) |
| Y (Tiefe) | Gauss-σ |
| Z (Höhe) | ICC(2,1) |
| Farbe | cobbType (Tab10-Palette) |

Für jeden `img_type` ein eigener 3D-Subplot. Mehrere cobbTypes an derselben X-Position
werden horizontal versetzt (`offset = (j − n_ct/2 + 0.5) · dx`).

Halbtransparente Referenzebenen:
- Z = 0.75 (grün, α = 0.10): Schwelle „gut"
- Z = 0.50 (orange, α = 0.10): Schwelle „moderat"

Blickwinkel: `elev=22°, azim=−50°`. Legende oben zentriert.

Zwei getrennte Charts werden erzeugt (falls Daten vorhanden):
1. Regionale cobbTypes — X-Ordnung: `CT → PT → MT → TL → U-LU → L-LU → LU`
2. Segmentale cobbTypes — X-Ordnung: `T12/L1 → L1/L2 → … → L5/S1`

---

## Anhang: Warum max-Aggregation statt mean?

Für `Cobb-regional` und `Spline` kann ein Rater für dieselbe Spine-Region mehrere Winkel
gemessen haben (z.B. zwei getrennte Kurven im MT-Bereich). Das größte gemessene Krümmungs-
maß ist klinisch das relevanteste (maximale Fehlstellung). Die max-Aggregation ist daher
nicht nur technisch notwendig für die ANOVA-Voraussetzungen, sondern auch inhaltlich
begründet.

Für `reconstructed_regional` gilt dasselbe: der Spline kann mehrere Bogen erzeugen;
der größte wird verglichen.

---

## Anhang: Verfehlte Alternative — Pooling über Regionen

Ein früherer Ansatz berechnte den ICC über alle Regionen gemeinsam, ohne vorher pro
Region zu aggregieren. Das führte zum ursprünglichen Problem zurück: verschiedene Rater
verglichen implizit verschiedene anatomische Bereiche miteinander (z.B. Rater A misst den
MT-Bogen, Rater B den TL-Bogen beim gleichen Bild). Das vermindert den ICC artefaktisch,
weil inter-regionale Winkelunterschiede als Rater-Varianz fehlinterpretiert werden.

Die aktuelle Implementierung vermeidet das durch die explizite Region-Aggregation vor
dem ICC-Schritt.
