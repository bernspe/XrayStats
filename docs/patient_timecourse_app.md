# patient_timecourse_app — Zwischenstand (Prototyp Winkelverlauf pro Patient)

Dieses Dokument fasst den aktuellen Stand des Prototyps `patient_timecourse_app.py`
zusammen. Ziel der App: eine Vorstufe für die geplante Visualisierung des zeitlichen
Winkelverlaufs in **Patient.vue** (xray-annotator-Frontend, Daten aus xray_api) — erst
hier in Streamlit erprobt, bevor die endgültige Umsetzung in xray-annotator/xray_api
geplant wird.

Start:
```
cd /Users/peterbernstein/Python/XrayStats
/Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run patient_timecourse_app.py
```

---

## 1. Ausgangsfrage und Entwicklungsweg

Ursprüngliche Anforderung: Winkelverlauf (regional + segmental, a.p. und seitlich) über
`acquisition_date`, mit Mittelwert und Streubreite über alle Annotatoren, interaktiv
filterbar.

Verworfene Zwischenschritte (zur Einordnung, warum die App so aussieht wie sie aussieht):

1. **Generisches Liniendiagramm** (matplotlib) mit Region/Segment-Multiselect — funktional,
   aber unübersichtlich bei vielen Serien.
2. **Altair-Heatmap** (Zeile=Region/Segment, Spalte=Zeitpunkt, Farbe=Winkel) mit
   Hover-verlinktem Detail-Chart — technisch funktionsfähig, aber vom Nutzer verworfen:
   *„Ich hätte schon gern die SVG als Grundlage.“*
3. **Echtes Wirbelsäulen-Piktogramm** aus xray-annotator (`whole-spine-*-labels*.svg`) als
   visuelle Basis — das ist der aktuelle, beibehaltene Ansatz (siehe §3).

---

## 2. Datenbasis

| Quelle | Inhalt |
|---|---|
| `data/df4d_corrected_spine_measurements.pkl` | gecachte, label-korrigierte Messungen (siehe Hauptprojekt-`CLAUDE.md`) |
| `data/img_urls.pkl` | `image_id → signierte GCS-URL`, **unabhängig** von obigem Pickle über den Sidebar-Token-Dialog auffrischbar (24h-Gültigkeit) |
| `/Users/peterbernstein/Vue/xray-annotator/public/whole-spine-ap-labels2.svg`<br>`.../whole-spine-lateral-labels2.svg` | Piktogramm-Vorlagen, **direkt aus dem xray-annotator-Repo** referenziert (kein Import in dieses Projekt) |

Kein Live-Fetch der Winkeldaten selbst — nur `img_urls.pkl` hat einen eigenen
Refresh-Mechanismus (`prepare_outlier_data.fetch_img_urls`, gleiches Muster wie
`subluxation_app.py`). Für eine produktive Nutzung müsste auch das Haupt-Pickle
regelmäßig neu erzeugt werden (`prepare_outlier_data.py`).

### Patientenliste (Sidebar)

Nur Patienten mit **≥2 unterschiedlichen Aufnahmedaten mit messbaren Cobb-Winkeln** in
mind. einem Bildtyp (a.p. oder seitlich) werden gelistet (`patients_with_timecourse()`) —
ohne zweiten Zeitpunkt gibt es nichts, das als „Verlauf“ darstellbar wäre. Reduziert die
Liste im aktuellen Datenstand von 99 auf 56 Patienten. Ein **„Nächster Patient“-Button**
erlaubt schnelles Durchklicken (Streamlit-`on_click`-Callback, da direktes Setzen von
`st.session_state` nach Widget-Instanziierung sonst einen `StreamlitAPIException` wirft).

---

## 3. Layout

Pro Bildtyp (a.p., seitlich) eine Zeile mit 3 Spalten (`st.columns([1, 1, 3])`):

```
[Erstbefund-Röntgenbild] [Letztbefund-Röntgenbild] [Piktogramm + 2 Liniendiagramme]
```

Die dritte Spalte ist **ein einziges** `st.iframe` (Piktogramm-SVG + zwei
Diagramm-Panels als Flex-Layout im selben HTML-Dokument) — bewusst nicht drei getrennte
Streamlit-Elemente, siehe §5 (Hover-Verlinkung).

Alle vier Elemente einer Zeile sind auf `ICON_MAX_HEIGHT_PX = 200`px Höhe normiert:
Röntgenbilder werden per PIL zugeschnitten+skaliert (§4), die zwei Diagramm-Panels sind
je `(200-8)/2 = 96`px hoch gestapelt (+8px Lücke), sodass die Gesamthöhe exakt der
Piktogramm-Höhe entspricht.

---

## 4. Röntgenbilder (Erstbefund/Letztbefund)

- `crop_black_border()`: entfernt das schwarze Letterboxing der Bild-Exporte. Ein naiver
  Bounding-Box-Ansatz („irgendein Pixel über Schwellwert“) scheitert, weil die Exporte
  einen 1px breiten hellen Rahmen an den äußersten Kanten haben, der jede Zeile/Spalte
  berührt und die Box am Schrumpfen hindert. Lösung: **größter zusammenhängender Lauf**
  von Zeilen/Spalten mit Mittelwert-Helligkeit über Schwellwert (numpy), nicht die reine
  Extremwert-Bounding-Box.
- `fit_height()`: skaliert exakt auf `ICON_MAX_HEIGHT_PX`. `st.image(..., width="content")`
  hat sich als unzuverlässig erwiesen (skalierte trotz bereits korrekter Pixelhöhe
  sichtbar hoch) — stattdessen wird die exakte Pixelbreite explizit übergeben.
- Bild-URLs kommen aus `img_urls.pkl`, **nicht** aus dem `img`-Feld in `df4d` (das ist von
  einem älteren Fetch-Zeitpunkt und daher meist bereits abgelaufen).

---

## 5. Piktogramm

`render_spine_icon()` lädt die echte SVG-Vorlage aus xray-annotator und färbt die darin
bereits vorhandenen Marker ein:

| Element | SVG-Struktur | Beschreibung |
|---|---|---|
| Regionen (MT, TL, U-LU, L-LU, LU) | `<g>` mit 2 Boundary-Line-`<path>`s (a.p.) **oder** einzelner `<path>` (seitlich) — **beide Formen werden gehandhabt** | Klammern am Wirbelsäulen-Icon |
| Segmente (L1/L2 … L5/S1) | `<ellipse>` | kleine Punktmarker auf Höhe der Bandscheibe |

„LU" (gesamte Lendenlordose) existiert als Element nur in einigen SVG-Versionen; fehlt es,
läuft der Lookup ins Leere (kein Fehler). Die zugrunde liegenden Winkel-Werte für „LU"
werden ohnehin **synthetisch** aus U-LU/L-LU gebildet (`with_synthetic_lu()`, gleiche
Konvention wie in `stats_app.py`).

Marker ohne Daten für den gewählten Patienten werden per `display:none` ausgeblendet
(nicht grau eingefärbt) — „nicht verwendete Spine-Regionen" sind schlicht unsichtbar.

Die Original-SVGs sind für ein sehr schmales, hohes Viewbox ausgelegt; bei
`ICON_MAX_HEIGHT_PX = 200` wird das Icon nur ~40–80px breit. Für Segment-Ellipsen wird
daher eine unsichtbare, 2.2× vergrößerte Duplikat-Ellipse als Hover-Ziel eingefügt; für
Regionslinien eine unsichtbare 4px-breite Parallel-Linie — beides nur für die
Hit-Detection, visuell unverändert.

### Farbkodierung: klinische Bewertung statt Absolutwert

Kernentscheidung (nach Nutzer-Feedback): Farbe kodiert **nicht** den Absolutwinkel,
sondern ob die **Veränderung seit dem Erstbefund** (`Δ = letzter Wert − erster Wert`)
klinisch günstig ist — grün=besser, rot=schlechter, weißlich-neutral=stabil, als
Farbverlauf (kein Stufenmodell).

**Bewertungsrichtung** (`trend_direction()`):

| Bildtyp | Region/Segment | Zunahme ist... |
|---|---|---|
| a.p. | alle | schlecht (jede Zunahme der Skoliose-Krümmung) |
| seitlich | MT, TL | schlecht (mehr Kyphose) |
| seitlich | U-LU, L-LU, LU, alle Segmente | gut (mehr Lordose) |

**Signifikanzschwellen** (`REGIONAL_SIGNIFICANCE_DEG = 5.0`, `SEGMENTAL_SIGNIFICANCE_DEG
= 3.0`): Änderungen unterhalb gelten unabhängig von der Richtung als „stabil"
(neutralgrau), da regionale Cobb-Winkel eine gröbere/streuungsanfälligere Messung sind
als segmentale. Farbsättigung ist erst `TREND_SCALE_DEG = 5.0`° **oberhalb** der
jeweiligen Schwelle voll erreicht (kontinuierlicher Verlauf, kein Sprung an der Schwelle).

> ⚠️ **Nicht klinisch validiert** — Schwellen/Skala sind Startwerte, keine
> literaturbasierten MDC-Werte (minimal detectable change). Ausdrücklich als
> nachjustierbar markiert (`ponytail:`-Kommentar im Code).

### Tooltip

Reiner Zahlen-Tooltip (keine eingebettete Sparkline mehr — redundant, da die
Liniendiagramme ohnehin permanent sichtbar sind): `Label Datum: Wert° ± Std° / Δ: ±X°
(±Y%) — Verdict`. Positionierung ist Viewport-bewusst (klappt bei Bedarf nach
links/oben, um nicht am Iframe-Rand abgeschnitten zu werden).

---

## 6. Liniendiagramme (Winkelverlauf)

`render_chart_panel()` zeichnet **zwei eigenständige SVG-Panels** (kein
Streamlit-natives `st.line_chart` — das unterstützt weder getrennte Y-Achsen noch
Std-Bänder, und ließe sich nicht mit dem Piktogramm-Hover verknüpfen):

- **„Regionale Winkel"** und **„Segmentale Winkel"** getrennt, **je eigene Y-Achse** —
  regionale Cobb-Winkel sind i.d.R. deutlich größer als segmentale; eine gemeinsame
  Skala würde die segmentalen Linien auf null zusammendrücken.
- Mittelwert-Linie **+ Std-Band** (schattiertes Polygon) pro Serie.
- Regionale Werte kombinieren `Cobb-regional` und `Spline` bereits vor dem Charting
  (`compute_label_stats()` gruppiert nur nach Region-Label, nimmt pro Rater/Bild/Datum
  das Maximum beider cobbTypes — vermeidet Doppelzählung, mirrors
  `annotation_analysis.aggregate_angles_by_region`).

### Hover-Verlinkung (Piktogramm ↔ Chart)

Piktogramm und beide Chart-Panels liegen **im selben HTML-Dokument** (ein `st.iframe`,
kein separates pro Element) — das ist der Grund für die Ein-iframe-Architektur in §3.
Cross-Iframe-`postMessage` wäre nötig gewesen, wenn Piktogramm und Charts in getrennten
Streamlit-Elementen lägen; das ist deutlich fragiler (Streamlit kontrolliert die äußeren
`<iframe>`-Attribute, eigene `name`-Zuweisung zum gezielten Ansprechen ist nicht möglich).
Stattdessen: gemeinsames JS, ein `data-key`-Attribut (z.B. `"TL"`, `"L2/L3"`) auf
Piktogramm-Marker, Chart-Linie *und* Chart-Legenden-Swatch — Hover auf **irgendeinem**
dieser drei Elemente zeigt denselben Tooltip und hebt die zugehörige Chart-Linie hervor
(`xrSetHighlight`/`xrClearHighlight`, Opacity-Dimmen der übrigen Linien).

**Für die spätere Vue-Implementierung nicht relevant**: In einer echten
Single-Page-Anwendung teilen Piktogramm und Chart ohnehin denselben DOM-Baum — der
iframe-Umweg ist eine Streamlit-Eigenheit (Components/HTML-Sandbox), keine
Architekturvorgabe für xray-annotator.

---

## 7. Offene Punkte für die xray-annotator/xray_api-Migration

Diese Liste ist der Ausgangspunkt für die geplante gemeinsame Planung (siehe
ursprüngliches Projektziel):

- **Datenquelle**: Prototyp liest ausschließlich gecachte Pickles; die echte Umsetzung
  braucht einen Live-Endpunkt in xray_api (oder zumindest einen Cache mit definierter
  Aktualisierungsstrategie) statt manuellem `prepare_outlier_data.py`-Lauf.
- **Bild-URL-Handling**: der 24h-Signatur-Refresh hier ist ein Workaround für den
  Streamlit-Prototyp (separates `img_urls.pkl`, manueller Token-Dialog) — wie
  xray-annotator/xray_api Bild-URLs tatsächlich ausliefert, ist nicht recherchiert;
  vor Übernahme in die Zielarchitektur dort prüfen, ob derselbe Ablauf-Workaround
  überhaupt nötig ist.
- **Schwellen/Farbskala** (§5) klinisch validieren oder mit Fachexpertise abstimmen,
  bevor sie in ein produktives UI übernommen werden.
- **Piktogramm-SVGs**: aktuell `whole-spine-*-labels2.svg` (zweite Iteration, vom Nutzer
  während der Prototyp-Entwicklung überarbeitet: breiteres viewBox, LU-Element in beiden
  Varianten ergänzt, Regionsmarker in a.p. weiterhin `<g>`+2 Pfade, in seitlich auf
  einzelne `<path>`-Elemente umgestellt). Die Doppel-Behandlung in `render_spine_icon()`
  (`tag == "g"` vs. Einzel-`<path>`) sollte bei künftigen SVG-Revisionen im Blick bleiben.
- **Layout-Robustheit**: feste Pixelbreiten für Chart-Panels (220px) und
  Spalten-Ratio (`[1,1,3]`) sind empirisch gegen die aktuelle Fensterbreite austariert,
  nicht responsiv. In Vue mit echtem CSS-Layout (Flexbox/Grid + Breakpoints) lösbar,
  ohne die iframe-Restriktionen dieses Prototyps.
- **Multi-Patient-Vergleich, Filter nach Rater/cobbType** o.ä. wurden im Rahmen des
  Prototyps bewusst nicht wiederaufgenommen (siehe verworfene Zwischenschritte in §1) —
  ggf. für xray-annotator erneut bewerten, falls klinisch gewünscht.
