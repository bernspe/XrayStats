# CHANGELOG

## 2026-10-02 – Winkelverlauf relativ zum Indexdatum

- Neuer Tab „Winkelverlauf (Index)“ in `degen_scoliosis_app.py`; Logik in `spine_geom/course.py` (Tests: `test_course.py`).
  - Anker `Patient.index_date`; Baseline = letzte Visite davor (Δ=0), Visiten ab Indexdatum = Veränderung.
  - Ansichten: LL / PI−LL (Xray-LWS-lat); LL, PI−LL + L-LU, U-LU, LU (Xray-LWS-ap, LU = max(U-LU, L-LU)); globale Parameter.
  - Methode wählbar: Cobb-regional | Spline (gilt für LL, PI−LL, L-LU, U-LU, LU). LL = zusammengesetzte Spanne T12→S1 aus `build_angles_df(lateral_spans=…)` (Port aus xray_api/api/helpers, wie ICC-Snapshot); Mittel je Visite über Untersucher/Aufnahmen.
  - Filter: therapy, therapy_specification, fused_segments, fused_to_pelvis (Index-Ereignis = Eintrag am Indexdatum, sonst nächster danach, sonst erster) sowie complication, long_term_complication (alle Einträge).
- `degen_pipeline.run` lädt zusätzlich `/api/clinicalentrydata/` → `data/degen_entries.pkl` (Fallback: `clinical_entries.pkl`).

## 2026-09-29 – Schritt 1a/1b: sagittale Geometrie + Abdeckungs-App

- `spine_geom/` (reine Bibliothek, numpy):
  - `frames`: Bild→Patient, Blickrichtung seitlich (S1-Deckplatte, Lordose-Kontrolle), Sakrumsystem, WKD-Normierung mit Level-Faktoren
  - `landmarks`: Nodes→Label, Hüftachse aus `hipLine`, S1-/L1-Deckplatte aus Cobb-Linien bzw. Tangenten-Fallback, Kalibrierziel geometrisch
  - `sagittal`: SS, PT, PI, LL, L1PA/T4PA/T1PA, T4-L1PA-Mismatch, Global Tilt, SVA (WKD); Voraussetzungstabelle `REQUIRES` mit Fehlgründen
- `spine_geom/tests/test_sagittal.py`: 16 Tests grün
  - PI = PT + SS < 0,01°
  - bekannte PT, SS, LL, L1PA
  - VPA-Vorzeichen
  - Spiegelinvarianz
  - Orientierung
  - Tangenten-Fallback
  - Kalibrierfaktoren
  - Sakrumsystem
- `degen_pipeline.py`: API → `data/degen_{measure,params,visits,calib}.pkl`; offline: `python degen_pipeline.py`
- `degen_scoliosis_app.py`: Token-Laden, Tab **Abdeckung** und Tab **Kalibrierung & QC**
  - Abdeckung: Parameter × „mind. 1×“/„Baseline“/„≥ 2 Visiten“, Hauptgründe, Patientenmatrix, Detail je Visite, CSV-Export

## 2026-09-29 – Schritt 1c + Phase 3: koronar, Formvektor, QC, Visite, PI als Patientenkonstante

- `spine_geom/coronal.py`: `centroid_cobb`
  - (a) zentrale Differenzen
  - (b) natürlicher Glättungsspline mit festem λ = 0,1
  - Endwirbel über Neigungsextrema mit Zusammenlegen von Rausch-Extrema < 5°
  - Apex und Konvexität aus Patientensicht
- `spine_geom/descriptors.py`: sagittaler Formvektor im Sakrumsystem (WKD), Segmenthöhen
- `spine_geom/qc.py`: kraniokaudale Monotonie, robuste z-Werte
- `spine_geom/patient.py`: PI als Patientenkonstante
  - Zufallseffektmodell mit σ_m (Rater am selben Bild) und τ (zwischen Bildern)
  - präzisionsgewichtet
  - sequentieller Leave-one-out-Ausschluss bei |z| > 3
- Tests: 27 grün
  - S-Kurve analytisch
  - Kreisbogen exakt
  - Rauschen teilt Kurve nicht
  - Formvektor spiegel- und skaleninvariant
  - PI-Schätzer: SE-Abdeckung ~95 % in Simulation, Ausreißerausschluss
- Pipeline:
  - speichert `degen_raw.pkl` (label-korrigierte Rohmessungen) und `degen_base.pkl`
  - `python degen_pipeline.py` rechnet ohne Token neu
  - neue Tabellen `degen_curves.pkl` und `degen_shape.pkl`
  - neuer Parameter `cCobb_max`
  - neues Flag `non_monotonic`
  - Alter < 18 J. wird verworfen (Testeinträge)
- App:
  - Tab **Visite**: Bild-Overlay im Konva-Stage-Raum, Parameter je Rater, Frontend-globalParams, QC
  - Tab **PI (Patientenkonstante)**: σ/τ, Streuung innerhalb Patient, Verlauf je Patient, Bland-Altman gegen Frontend (Spline/Cobb), L5/S1-Abhängigkeit, zeitliche Instabilität
- pytest ins App-venv installiert, weil scipy in Anaconda defekt ist: `…/ICFXStats/venv/bin/python -m pytest spine_geom`

## 2026-09-29 – Phase 2 (Messunsicherheit) + Korrekturen

- `spine_geom/uncertainty.py`: `pooled_cov` (robust, mit Rate grober Fehler), `perturb`, `mc_sd`, `mdc95`
- `spine_geom/landmarks.py`: Deckplatten-Mittelpunkt = Schnittpunkt der Cobb-Linie mit der Centroid-Sehne statt Linienmitte (Abweichung von Spec §4.2, siehe DECISIONS.md)
- `spine_geom/patient.py`: sequentieller Leave-one-out-Ausschluss
- PI (wie alle sagittalen Parameter) nur noch aus seitlichen Röntgen-Stehaufnahmen: keine Funktions-, keine MRT-Aufnahmen
- Pipeline:
  - Monte-Carlo-SD je Messung und Parameter (`sd`)
  - `degen_uncert.pkl`
  - `reports/messunsicherheit.md` wird automatisch erzeugt
- App: Tab **Messunsicherheit**; Werte im Visiten-Tab mit ± SD; Overlay auf die annotierten Punkte zugeschnitten
- `reports/pi_patientkonstante.md` (Phase-3-Bericht)
- Tests: 33 grün

## 2026-09-29 – Phase 4 (Normabweichung) + Phase-0-Dokument

- PI und alle sagittalen Parameter nur aus seitlichen Röntgen-Stehaufnahmen (auch kein MRT mehr); die Lagerung wird vor den Landmarken geprüft
- `spine_geom/norms.yaml`: Quellen mit Population, n, Altersbereich und `verified`; Populationsnormen; PI-Ziele; Zielbereiche; Klassifikationen
- `spine_geom/norms.py`: `population_reference`, `deviation_state` (Population, PI-Ziel, Zielbereich, Klassifikation mit Unsicherheit und Quelle)
- L4-Deckplatte aus den Cobb-Linien; neuer Parameter **LDI**
- App: Tab **Normabweichung** (Visite: Tabellen je Art; Kohorte: PI gegen LL mit GAP- und Schwab-Linien)
- `docs/datenmodell.md`: Mapping API → Spec und Schemavorschlag (Phase-0-Abnahme)
- Tests: 40 grün

## 2026-09-29 – Normen aus Volltexten, koronare Normparameter

- `norms.yaml`:
  - Yukawa-Strata (Geschlecht × Dekade)
  - Wang 2025 (Altersstrata ≥ 50 J., inkl. LL L1–S1)
  - Gharbi 2024 (koronar sowie T1PA, GT, PI−LL mit SD)
- `norms.py`: halboffene Altersgrenzen; bei Gleichstand gewinnt der Eintrag mit SD; Alias C7-CSVL
- `coronal.py`: `tilt_to_horizontal`, `coronal_global` (CTPA, C7-CSVL)
- `landmarks.pick_endplate` öffentlich, auch für a.p.-Deckplatten genutzt
- Pipeline: neue a.p.-Parameter FHA, L4_tilt, L5_tilt, CTPA, C7_CSVL_wkd inkl. Abdeckung, MC-SD und SEM
- Tests: 43 grün
- `patient.robust_sd` + Test; SEM-Tabelle und Bericht mit robuster SEM und MDC95
- Normtab: Parameter ohne MC-SD bekommen die robuste Rater-SEM als Unsicherheit
- Tests: 44 grün

## 2026-09-29 – Bugfix Cache

- `KeyError: 'LDI'` im Abdeckungs-Tab behoben. Ursache: `st.cache_data` hasht keine Argumente mit führendem Unterstrich; `load(_mtime)`, `pi_analysis(_mtime, …)` und `visit_values(_mtime, …)` wurden deshalb nie neu berechnet, eine laufende Instanz zeigte Daten von vor der Neuberechnung. Die Argumente heißen jetzt `mtime`; `load` bekommt zusätzlich die Parameterliste als Schlüssel.
- `spine_geom/change.py` + Tests (Phase 5, Kern): D², p, RCI, Projektion Richtung Ziel, Typologie. Abnahme erfüllt: Fehlerrate unter H₀ 5 % ± 1 % (Simulation n = 4.000).

## 2026-09-29 – Phase 5 (Veränderungsvektoren)

- `spine_geom/change.py`: `nearest_psd` + Test
- Pipeline `change_model`:
  - Fehlerkorrelation der Parameter und Formkoordinaten (Rater am selben Bild)
  - τ gleicher Tag (verschiedene Aufnahmen desselben Tages)
  - τ Tag-zu-Tag aus der PI: 4,2°, 294 Visitenpaare
  - gespeichert in `degen_uncert.pkl['change']`
- App: Tab **Verlauf**
  - Trajektorien mit 95 %-Intervall
  - Visitenpaar: D², p, RCI je Komponente, Δ/Jahr, Richtung Norm, Typologie, ΔPI-Kontrolle
  - Formvektor mit Verschiebungspfeilen (orange = jenseits des Fehlers)
  - Kohorte erste gegen letzte Visite (SS, PT, LL, L1PA) mit CSV-Export
- Tests: 49 grün
- Verlauf-Tab: Röntgenbilder t₁ | t₂ nebeneinander mit Overlay (seitlich/a.p. umschaltbar, Stehaufnahmen zuerst); `overlay_figure` mit einstellbarer Höhe, gemeinsamer Helfer `_image_panel`

## 2026-09-30 – Formvektor: Zwischenstationen und 3D

- `descriptors.coronal_shape` (koronares Beckensystem) + Test; Pipeline speichert a.p.-Formkoordinaten (`view`, `frame_ref` in `degen_shape.pkl`), `change_model` mit `.z`/`.c`
- App Verlauf → Formvektor: Darstellung 2D seitlich / 2D a.p. / 3D; alle Stationen mit Pfeilen zwischen aufeinanderfolgenden Visiten; Schritt-Tests (inkl. Gesamt) über die jeweils gemeinsamen Wirbel
- Tests: 50 grün
- Verlauf: Widget-Schlüssel je Patient (Stationsauswahl blieb nach Patientenwechsel leer); Schritttabelle in voller Breite unter der Formgrafik
- Formvektor: Pfeilspitzen in 3D (Kegel, einheitliche Größe); **Gesamtvektor** = Verschiebung des Schwerpunkts der gemeinsamen Wirbel (erste → letzte Station, magenta) mit eigenem Test (Σ = A·Σ_Δ·Aᵀ) und 95 %-Intervall je Achse in der Schritttabelle
- Formvektor-Darstellung nach Konvention seitlicher Bilder: anterior links, posterior rechts (2D seitlich und 3D; nur Achsenrichtung umgekehrt, Daten/Tests unverändert, intern weiter anterior = +x)
- `change.mean_shift` (mittlere Verschiebung einer Wirbelgruppe, optional relativ zu einem Referenzwirbel) + Test
- Formvektor: **Lumbalvektor** (L1–L5, Sakrumsystem, rot) und **Thorakalvektor** (BWS relativ zu L1, blau), je ≥ 3 gemeinsame Wirbel; Pfeile am Regionsschwerpunkt bei t₀ (2D und 3D), Komponenten ± 95 % und p je Region in der Schritttabelle; Gesamtvektor unverändert
- Tests: 51 grün

## 2026-09-30 – Maßstab: Kalibrier-Konsistenz und Maßstabsfehler

- Befund (Nutzerhinweis 2PVD): scheinbare Höhenänderungen im Formvektor durch inkonsistente Kalibrierlinien (WS-a.p. 06/2024: Linie 44 px vs. 32 px bei gleichem Wirbelabstand → −32 % Maßstab)
- `descriptors.lumbar_length` (L1→S1 in px) als kalibrierunabhängige Maßstabsreferenz
- Pipeline `calib_consistency`: Maßstab > 15 % vom Patientenmedian derselben Ansicht (≥ 3 Aufnahmen) → Patientenmaßstab, Flag `calib_inconsistent` (133 Messungen); bei 2 Aufnahmen nur Flag `calib_inconsistent_unresolved` (6); σ_s je Ansicht aus paarweisen Log-Differenzen minus Rater-Anteil (seitlich 4,2 %, a.p. 5,2 %) → `degen_uncert.pkl['change']['scale_sd']`
- App: Σ_Δ um den Maßstabsterm 2·σ_s²·x·xᵀ je Ansicht erweitert (radial, voll korreliert) für Formkoordinaten, SVA und C7-CSVL; Trajektorien-Intervalle ebenso; Hinweis im Verlauf-Tab
- Tests: 54 grün (u. a. Korrektur bei 3 Bildern, nur Flag bei 2 Bildern, σ_s-Schätzung)

## 2026-09-30 – Maßstab revidiert

- `calib_consistency`: **keine Korrektur mehr**, nur Flag `calib_inconsistent` (149 Messungen). Die Korrektur über die Lendenlänge hatte reale Höhenänderungen entfernt (2PVD: +3 cm laut cm-Lineal → < 1 cm).
- σ_s nur aus verschiedenen Aufnahmen desselben Tages (seitlich 8,0 %, a.p. 7,6 %); `MIN_SCALE_PAIRS` = 10
- App: Hinweis im Verlauf-Tab zur begrenzten Beurteilbarkeit absoluter Höhenänderungen (kleinste erkennbare relative Änderung ≈ 21 %)
- Tests: 53 grün (nur Flag, keine Korrektur; σ_s ignoriert echte Änderungen zwischen Visiten)

## 2026-09-30 – Maßstab aus Wirbelkörperhöhen

- `landmarks.vb_heights`: Wirbelkörperhöhe aus Cobb-segmental-Linien, gemessen als Abstand der Schnittpunkte mit der Wirbelsäulenachse. Unveränderte Modellvorschläge sind ausgeschlossen.
- `frames.vbh_factors`, `frames.patient_scale`: Maßstab je Messung aus Linie und Höhen mit patienteneigener Anatomie (gewichtete Kleinste Quadrate je Patient × Ansicht)
- `degen_pipeline.py`: `wkd_px` aus `patient_scale`; neue Spalten `calib_source`, `n_vbh`, `wkd_line_px`; Ausgabe `degen_vbh.pkl`
- App (Kalibrierung & QC): Tabellen „Höhe/WKD je Level“ und „Maßstabsquelle“
- σ_s seitlich 8,0 → 5,4 %, a.p. 7,6 → 7,1 %; Rater-CV des Maßstabs seitlich 4,7 → 3,4 %
- Tests: 57 grün (`test_vbh.py`: Neigung, Keilwirbel, Linienlänge, Modellvorschläge, Doppel-Linien, Patientenanatomie)

## 2026-09-30 – Formvektor: 3D-Richtung, höheninvariante Darstellung

- 3D: Die x-Achse zeigte andersherum als im 2D-Seitenbild, weil `autorange='reversed'` in 3D je nach Kamera die Seite umkehrt. Neu: Kamera blickt von Patient links, anterior erscheint links wie im 2D-Bild.
- Schalter **Höheninvariant**: Jede Station wird je Ansicht auf die mittlere Wirbelsäulenlänge skaliert (Polygonzug durch die Wirbel, die in allen Stationen vorhanden sind). Zeigt nur die Formänderung, ohne Höhenänderung und Maßstabsfehler. Pfeile und Tests nutzen die skalierten Werte.
