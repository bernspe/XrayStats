# Umsetzungsplan: Verlaufsmodell degenerative Skoliose (Streamlit, Phasen 0–5)

Grundlage: `docs/SPEC_degenerative_skoliose_verlaufsmodell.md`. Entscheidungen: `DECISIONS.md`.

## Architektur

```
spine_geom/                   reine Bibliothek (numpy/scipy, kein Streamlit/Django)
  frames.py                   Bild→Patient (y_pat = −y_img, Anterior-Normalisierung), Sakrumsystem,
                              Längennormierung px → WKD (calibrationLine, Level-Umrechnung auf L4)
  sagittal.py                 SS, PT, PI, τ_L1, LL, VPA_i, L1PA/T4PA/T1PA, Mismatch, SVA-Surrogat
  coronal.py                  centroid_cobb (a) zentrale Differenzen, (b) Glättungsspline s=fix; Apex, Konvexität
  descriptors.py              Formvektor (Sakrumsystem), Haltungsvektor, Segmenthöhen (kalibriert)
  landmarks.py                Messung → {label: C_i}, H, S1-/L1-Deckplatte (Cobb-Linie, Fallback Tangente)
  qc.py                       Monotonie, Ausreißer-z, Flags
  uncertainty.py              Landmarken-Kovarianz aus Multi-Rater, Monte-Carlo, SEM/MDC95
  norms.py + norms.yaml       Normwerte mit Quelle/Population/n/Alter; GAP, L1PA-Ziel, Roussouly
  change.py                   Δx, Σ_Δ, D², RCI, u_norm-Projektion, Typologie
  tests/test_*.py             pytest, synthetische Geometrie
degen_scoliosis_app.py        Streamlit: Token → Fetch → Cache → Tabs
```

**Wiederverwendung:** `prepare_outlier_data.fetch_data()`, `annotation_analysis.build_df4d / detect_label_errors / apply_label_corrections`, `bspline_python.interpolate_bspline / map_nodes_to_spline_and_keys / calculate_tangents` (für den Tangenten-Fallback).

**Token-Handling** wie in `clinical_timecourse_app.py`: Passwortfeld in der Sidebar und Button „Daten laden“. Geladen werden `/api/imagedata/?detailed=True&complete=True` und `/api/clinicalbasedata/` (Geschlecht, Geburtsdatum). Die Pipeline schreibt nach `data/degen_*.pkl`, die App liest danach nur noch den Cache.

## Abgeleitete Tabellen (Cache)

| Tabelle | Zeile = | Inhalt |
|---|---|---|
| `degen_landmarks.pkl` | Messung × Label | x, y (Patientenkoordinaten), img_type, rater, Flags |
| `degen_measure.pkl` | Messung | H, S1-/L1-Deckplatte + Quelle, alle Parameter pro Rater |
| `degen_image.pkl` | Bild × Parameter | Mittelwert, Rater-SD, n_rater |
| `degen_visit.pkl` | Patient × Datum | Kombination a.p. + seitlich, Alter, Geschlecht |

## Streamlit-Tabs

1. **Abdeckung.** Beantwortet: *Sind die globalen sagittalen Parameter bei allen Patienten abgebildet?*
   - Heatmap Patient × Visite × Parameter. Zellwert „messbar“, „nur via Tangente“ oder „fehlt“, dazu der Grund (keine Hüfte, Range endet unter T4, kein L1 …).
   - Parameter: PI/PT/SS, LL, L1PA, T4PA, T1PA, T4-L1-Mismatch, Global Tilt, SVA-Surrogat (C7), centroid_cobb.
   - Zusammenfassung „mind. 1×“, „Baseline“, „≥ 2 Visiten“ sowie die Liste der Patienten **ohne** Hüfte bzw. ohne T4.
   - Export als CSV, um gezielt Nachannotation oder Ganzaufnahmen anzufordern.
2. **Visite.** Bild mit Overlay (plotly): Nodes, H, Deckplatten, Sakrumsystem, VPA-Linien. Dazu die Parametertabelle pro Rater. Zum Vergleich die gespeicherten Frontend-`globalParams`.
3. **PI pro Patient** (Phase 3).
   - Präzisionsgewichtetes Mittel mit SE; Flags bei mehr als 3 SD Abweichung.
   - Streuung innerhalb eines Patienten gegenüber dem Messfehler.
   - Vergleich eigene PI gegen Frontend-`pelvic_incidence` (Spline) und `pelvic_incidence_cobb` (Bland-Altman).
4. **Messunsicherheit** (Phase 2).
   - Landmarken-Kovarianz je Label und img_type.
   - Monte-Carlo-Fortpflanzung, Tabelle SEM/MDC95 je Parameter.
   - Bland-Altman `centroid_cobb` gegen den manuellen Cobb-regional.
5. **Normabweichung** (Phase 4).
   - `deviation_state(patient, visit)`: z-Werte nach Alter und Geschlecht, PI-abgeleitete Ziele (L1PA, GAP-SS/LL, Roussouly-Typ) mit Quelle.
   - Flag, wenn der Patient außerhalb der Normpopulation liegt.
   - LDI als Centroid-Surrogat, gekennzeichnet.
6. **Verlauf** (Phase 5).
   - Trajektorien je Parameter mit Unsicherheitsband.
   - Visitenpaar wählen → Δx (Form-/Haltungsvektor), D², p, RCI je Komponente, Projektion auf u_norm, Typologie-Quadrant.
   - Overlay mit Verschiebungspfeilen je Node im Sakrumsystem.
   - Slider für α und Σ_haltung-Platzhalter, mit Warnhinweis.

## Reihenfolge und Abnahme

Jede Phase endet mit grünen pytest-Tests und einem Eintrag in `CHANGELOG.md`/`DECISIONS.md`.

| Schritt | Inhalt | Abnahme |
|---|---|---|
| **0** | `docs/datenmodell.md`: API → Spec-Mapping, Vorschlag für Metadaten-/Schemafelder (Aufnahmetyp, Lagerung, Pixel Spacing, Übergangswirbel, beschriftete Deckplatten) ohne Umsetzung | Dokument und Liste zur Freigabe |
| **1a** | `frames`, `landmarks`, `sagittal` inkl. Tests (PI = PT + SS < 0,01°, VPA-Vorzeichen, Spiegelinvarianz) | Tests grün |
| **1b** | App-Gerüst mit Token/Fetch/Cache und **Tab Abdeckung** | Beantwortet die Abdeckungsfrage mit Live-Daten |
| **1c** | `coronal`, `descriptors`, `qc` und Tab Visite | Tests grün, Overlay plausibel an 5 Stichproben |
| **3** | PI als Patientenkonstante, Tab PI | Report: Streuung innerhalb Patient ≈ Messfehler, Ausreißer erklärt |
| **2** | `uncertainty`, Tab Messunsicherheit → `reports/messunsicherheit.md` | Tabelle SEM/MDC95 je Parameter |
| **4** | `norms.yaml` (Seed aus Spec §3.6), `norms`, Tab Normabweichung | `deviation_state()` liefert Werte mit Unsicherheit und Quelle |
| **5** | `change`, Tab Verlauf | Simulation: Fehlerrate unter H₀ = 0,05 ± 0,01; Beispielpatient visualisiert |

Phase 3 vor Phase 2, weil die PI-Konstanz bereits mit der Rater-SD als vorläufigem Messfehler auskommt und direkt auf die Abdeckungsfrage aufbaut.

## Vorläufiger Befund Abdeckung

Cache vom 03.08.2026, noch ohne neue Annotationen:

| | Patienten |
|---|---|
| gesamt | 149 |
| mit seitlichem Röntgen | 141 |
| PI/PT/SS mind. 1× (Hüfte + S1) | ~120 |
| L1PA | ~120 |
| T4PA | 24 |
| T1PA / Global Tilt / C7 | 19–23 (nur Xray-WS-lat) |

→ Die Beckenparameter sind breit abgedeckt. Die **globalen** Parameter (T4PA, T1PA, SVA) fehlen bei etwa 85 % der Patienten. Das spricht für die Imputation aus Phase 7 bzw. für gezielte Ganzaufnahmen.

## Risiken

- Die Identifikation der Deckplatten ist heuristisch, weil die Cobb-Linien nicht beschriftet sind. Das QC-Flag und Tab 2 machen Fehlgriffe sichtbar.
- Die k-Means-Labelkorrektur weicht von Spec §0.6 ab (siehe DECISIONS.md).
- Längen werden in Wirbelkörperdurchmessern (WKD) über die `calibrationLine` normiert, mit Umrechnung auf das Referenzlevel L4 (siehe DECISIONS.md). Die Rater-Streuung von etwa 4 % geht in die Unsicherheit ein. Werte in mm gibt es nur über eine Norm-Wirbeltiefe; ein Vergleich mit mm-Normwerten wie dem SVA bleibt daher näherungsweise.
- Σ_haltung und Q sind nicht identifizierbar. Die RCI-Ergebnisse sind deshalb eher zu liberal.
