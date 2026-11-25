# Beurteilung des Skolioseverlaufs in Reaktion auf Therapie
In der Xray-api liegen die MeasureData Verläufe der Patienten vor. In ClinicalEntries finden sich dort die entsprechenden Einträge zu Symptomen und Therapie zugeordnet.
Ziel dieser Auswertung ist die Analyse der segmentalen Veränderungen in Reaktion auf die jeweilige Therapie.

## Sepzifische Fragen
 - welches Therapieverfahren (konservativ, Dekompression, Fusion) erzielte welche Veränderungen in den Segmenten, Angabe pro Segmenthöhe
 - gab es eine Abhängigkeit der seitlichen Korrekturen, wenn die Skoliose korrigiert wurde? Ging eine Skoliosekorrektur zu Lasten der lumbalen Lordose?
 - hat die Richtung der Cage-Implantation etwas mit der Öffnungsrichtung der Skoliose zu tun?
 - Wurde durch eine langstreckige Fusion auch segmental eine größere Korrektur möglich?
 - wie wirkte sich die segmentale Korrektur auf die regionale Kurve und auf die globale Kurve aus? Wie veränderten sich durch eine kurzstreckige Fusion die Nachbarsegmente?
 - Ging die Korrektur im weiteren Verlauf wieder verloren? Wenn ja, in welchem Segment und in welchem Zeitraum?

## Datengrundlage aus xray-api (/Users/peterbernstein/Django/xray_api/api/models.py)
Einwahl in die Api vergleichbar mit den anderen streamlit apps, Token erforderlich
 - Patient: patient_id, image_groups (listet zusammengehörige Images auf, die an verschiedenen Tagen aufgenommen wurden, aber der gleichen Entität angehörigen zum Beispiel Rö+MRT präOP), index_date (Datum der ersten Therapie, alle weiteren Abstände müssen relativ zu diesem Datum angegeben werden)
 - ImageData: measure_dataset (beinhaltet die Messungen unterschiedlicher Annotatoren, hieraus müssen Mittelwerte gebildet werden)
 - MeasureData.data: Struktur siehe api_imagedata_structure.md. Die dort gespeicherten Messungen müssen möglicherweise noch korrigiert werden (s.unten)
 - ClinicalEntryData: Enthalt die Einträge zu Symptomen, MRT-Befunden und Therapie, die den jeweiligen Messungen zugeordnet werden können. 
 - Inferenz der Schrauben und Cages (wenn ClinicalEntryData.therapy = Fusion) aus dem Model /Users/peterbernstein/Python/Xray-Model/runs/implants-merge-20261120/implant-detector-implants-merge-20261120.onnx, einschliesslich der Richtung, aus der der Cage eingebracht wurde. 

## Erforderliche Korrekturen der Messwerte vor einer Analyse
 - Die Segmentalen Winkel müssen aus der Spline pro Segment berechnet werden (Senkrechte auf der Tangente am Centroid)
 - Es muss eine Codierung der Winkelöffnung erfolgen, um die Richtung der Öffnung zu bestimmen (Lordose/Kyphose und links/rechts)

---

# Umsetzungskonzept (Stand 24.11.2026)

## 1. Entscheidungen aus dem Interview

| Thema | Entscheidung |
|---|---|
| Zeitbezug | **Je Therapie-Episode.** Jedes Therapieereignis hat eine eigene Baseline (letzte Aufnahme davor) und einen eigenen Verlauf bis zum nächsten Ereignis. Die Zeit wird zusätzlich relativ zu `index_date` ausgewiesen. |
| Spline | **Frontend-Spline** wie xray-annotator ≥ 2.17.6 (`bspline.ts`/`measureCompute.ts`), in `spine_geom.segmental.frontend_spline` nachgebaut. `bspline_python.interpolate_bspline` ist der ältere, interpolierende Stand und wird hier nicht verwendet. Segmentwinkel = Winkel zwischen den Senkrechten auf der Tangente an zwei benachbarten Centroiden. |
| Vorzeichen | **Absolut.** Seitlich: + = Lordose (anterior offen), − = Kyphose. a.p.: + = Öffnung nach Patient links, − = nach rechts (Konvention von `spine_geom`). Gepoolt wird über Beträge bzw. die Korrektur `|α_prä| − |α_post|`, nicht über das Vorzeichen. |
| Ausgabe | **Pipeline + neue App:** `segmental_course_pipeline.py` → `data/seg_*.pkl`, `segmental_course_app.py` mit einem Tab je Fragestellung, Bericht `reports/segmental_therapieeffekt.md`. |
| Segmenthöhen | **Implantate + Text.** Bei Fusion kommen die Höhen primär aus der Schraubendetektion im postop. Bild, auf die Wirbel-Nodes gemappt; `therapy_specification` dient als Gegenprobe. Bei Dekompression werden die Segmente aus `therapy_specification` geparst. |
| Cage-Hypothese | **Konvex vs. konkav.** Wird der Cage von der offenen (konvexen) oder der geschlossenen (konkaven) Seite des Segments eingebracht, und korrigiert eine Seite mehr? |
| Therapiegruppen | **Mapping-Tabelle** (editierbares dict, im Bericht gelistet), siehe §3.6. |
| Statistik | **Deskriptiv + MDC.** Je Segment Median/IQR der Änderung, Bootstrap-CI über Patienten und Anteil der Segmente mit \|Δ\| > MDC95. Ein gemischtes Modell kommt erst, wenn n ausreicht (Richtwert ≥ 20 Episoden je Gruppe). |
| Kohorte | **Nur validierte Patienten:** `Patient.progresstag` enthält einen Tag mit `status` `VALID` (so heißt der Wert in der API; die Konstante `VALID_STATUS = {'VALID', 'VALIDATED'}` akzeptiert beide Schreibweisen). Für diese Patienten gelten ClinicalEntries und Annotationen als vollständig, es werden **keine Lücken gefüllt**. Was fehlt, schließt die Episode bzw. das Segment aus und wird nur gezählt. |
| Inkrementell | Ein späterer Lauf rechnet **nur neu hinzugekommene validierte Patienten**. Bereits berechnete Patienten bleiben unverändert im Cache (§3.8). |
| Seitliche Bilder | **Fest: links = anterior, rechts = posterior.** Die Blickrichtungs-Heuristik `facing_lateral` entfällt. |
| a.p.-Beckenlinie | `hipLine` auf a.p. ist teils an den Hüftkopfzentren, teils an den unteren Enden der Iliosakralfugen gesetzt. Sie wird je Messung klassifiziert (§3.4) und nie ungeprüft als Hüftkopfzentrum verwendet. |
| Slip | **Dritte Segmentgröße neben koronarem und sagittalem Winkel:** sagittaler Wirbelversatz aus den Hinterkanten-Ecken PS/PI, in % WKD, > 0 = Anterolisthese. Manuelle Ecken (`corner_annotations.csv`) haben Vorrang, sonst das Slip-Modell über `POST /api/suggestcorners/`. Nur seitliche Stehaufnahmen, keine Funktionsaufnahmen (§3.9). |
| Ziel | **Machbarkeitsnachweis.** Die Datenmenge reicht für belastbare Aussagen noch nicht. Die Pipeline soll zeigen, dass jede Frage aus den Daten beantwortbar ist; die Zahlen werden mit n und als vorläufig ausgewiesen. |

## 2. Datenlage

Im Cache `data/image_groups_cache.json` (Stand 22.10.2026) haben 21 von 334 Patienten einen `VALID`-Tag. Das ist die Startkohorte für den Machbarkeitsnachweis. Die Therapieverteilung unter diesen Patienten zählt Schritt 0. Im Gesamtbestand (172 Einträge) sind nur 9 Dekompressionen und 8 Fusionen; je Frage sind also eher Einzelfälle als Gruppen zu erwarten.

Schritt 0 ist deshalb **keine Lückenliste mehr**, sondern eine reine Zählung: Episoden je Gruppe, davon auswertbar (Baseline + Verlauf, a.p. und/oder seitlich, Nodes auf postop. Bildern), sowie der Ausschlussgrund für den Rest.

## 3. Datenaufbereitung

### 3.1 Quellen (Wiederverwendung)

| Quelle | Nutzung |
|---|---|
| `data/degen_raw.pkl` (aus `degen_pipeline.py`) | label-korrigierte Messungen (df4d) inkl. Rohannotation. Basis für die Neuberechnung, kein eigener Fetch der Messungen. |
| `data/degen_entries.pkl` | ClinicalEntryData (therapy, therapy_specification, fused_segments, fused_to_pelvis, complication, long_term_complication) |
| `data/degen_params.pkl`, `degen_curves.pkl` | regionale/globale Parameter (centroid_cobb-Hauptkurve, LL, L1PA, T1PA, SVA, C7-CSVL), nicht neu berechnet |
| `/api/imagedata/` (`referring_patient`) | `index_date`, `image_groups`, `progresstag[].status` (Kohortenfilter); Token wie in den anderen Apps: Sidebar-Passwortfeld bzw. `XRAY_TOKEN` |
| Bilder (signierte URLs, `prepare_outlier_data.fetch_img_urls`) | nur für die Implantat-Inferenz |

Aktualisiert wird mit `degen_pipeline.run(token)` und anschließend `segmental_course_pipeline.run()`.

### 3.2 Messungen, Rater, Visiten

1. **Aufnahmetypen:** Hauptanalyse nur mit Stehaufnahmen in Neutralstellung.
   - koronar: `Xray-LWS-ap`, `Xray-WS-ap`
   - sagittal: `Xray-LWS-lat`, `Xray-WS-lat`
   - `incl/recl` (Funktionsaufnahmen) und MRT (liegend) sind ausgeschlossen. MRT ist höchstens Sensitivitätsanalyse.
2. **Rater:** Winkel je Rater und Bild berechnen, danach mitteln. Je Bild werden Mittelwert, SD und n_rater gespeichert. Gemittelt wird über Winkel, nicht über Node-Positionen, weil die Rater unterschiedliche Wirbelbereiche (`vertebra_range`) setzen.
3. **Visite:** Patient × Aufnahmedatum.
   - Bilder einer `image_group` (z. B. `preop-img`: Rö + MRT an verschiedenen Tagen) bilden **eine** Visite. Ihr Datum ist das späteste Bilddatum der Gruppe (am nächsten an der OP).
   - Gibt es pro Visite und Ebene mehrere Bilder, wird gemittelt; LWS und WS werden gemeinsam gemittelt (Flag `view`).

### 3.3 Segmentwinkel (Korrektur laut Vorgabe)

Neues Modul `spine_geom/segmental.py` (reine Funktionen):

- `segment_angles(nodes, labels, plane) -> {seg: {angle, edge}}`, Ablauf wie das Frontend:
  - **Spline:** geklemmter kubischer B-Spline mit den Nodes als *Kontrollpunkten*. Er geht nur durch den ersten und letzten Node und glättet dazwischen; ausgewertet an 10 · n + 1 Punkten. Mindestens 4 Nodes.
  - **Zuordnung:** Jeder Node bekommt den Spline-Punkt mit der nächsten Höhe (floor(y)), höchstens den vorletzten.
  - **Tangente:** über ±win Spline-Punkte mit win = round(Splinelänge / n) ≈ ein Wirbel, angesetzt am Node.
  - **Winkel:** Der Winkel zwischen den Senkrechten ist gleich der Tangentendifferenz. Das Frontend rundet ihn ohne Vorzeichen auf ganze Grad; hier bleibt er vorzeichenbehaftet und ungerundet.
- **Vorzeichen:** Die Berechnung läuft in Patientenkoordinaten (`spine_geom.frames.to_patient`).
  - seitlich mit festem `facing` (Bild links = anterior, rechts = posterior; Nutzervorgabe), keine Heuristik. QC-Flag `lordosis_negative`, wenn L1–S1 als Kyphose herauskommt (Hinweis auf ein gespiegeltes Bild oder Labelfehler)
  - a.p. mit `facing_ap` (radiologische Konvention, Flag `side_assumed`)
  - Kodierung als Zahl mit Vorzeichen plus Spalte `open_side ∈ {ant, post}` bzw. `{L, R}`
- **Konsistenz:** Die Summe der Segmentwinkel zwischen zwei Wirbeln ist gleich dem regionalen Spline-Winkel zwischen ihnen (Identität der Tangentendifferenzen). Damit ist die Zerlegung in Frage 5 exakt.
- **Segmentbereich:** T10/T11 bis L5/S1, Kern T12/L1 bis L5/S1.
- **Randsegmente (Nutzerentscheidung 24.11.2026):** Der geklemmte Spline läuft an den Enden in Richtung der letzten Kontrollpolygon-Kante aus, die Tangente dort ist einseitig. Randwinkel sind deshalb strukturell zu klein: L5/S1 seitlich im Median 3,9° gegenüber 10° aus den gezeichneten Cobb-Linien.
  - Das **oberste Segment** jeder Messung entfällt, ebenso das unterste, wenn es nicht bei S1 endet.
  - **L5/S1** = Senkrechte auf der Spline-Tangente an L5 gegen die **gezeichnete S1-Deckplatte** (wie bei LL/PI).
    - Quelle der Deckplatte: zuerst die `lowerLine` eines gespeicherten Cobb-Winkels mit `vertebra_lower = S1` (vom Frontend beschriftet, Cobb-segmental vor Cobb-regional), sonst die geometrische Zuordnung `landmarks.pick_endplate`.
    - Abdeckung in der Kohorte: seitlich 311 von 316 Messungen (98 %), a.p. 135 von 285 (47 %).
  - **Ohne Deckplatte** bleibt der Spline-Wert mit `src = s1_spline` / `edge` stehen. Innerhalb einer Visite fällt er weg, sobald ein anderer Rater eine Deckplatte hat. Δ und K werden nie über einen solchen Wert gerechnet, sonst stünde Methode gegen Methode.

### 3.4 a.p.-Beckenlinie (Hüftkopf vs. ISG)

Die Klassifikation erfolgt je a.p.-Messung über Breite `w` und Höhe `dy` (Mittelpunkt unter dem S1-Centroid) der `hipLine`, beide in Einheiten des L5–S1-Centroidabstands.

| Klasse | Regel | Anteil im Cache (697 a.p.-Messungen) |
|---|---|---|
| `isg` (untere ISG-Enden) | `w < 4,5` und `0 < dy < 2,5` | ≈ 290 |
| `hip` (Hüftkopfzentren) | `w ≥ 5,5` und `dy ≥ 2,5` | ≈ 250 |
| `unklar` | dazwischen | ≈ 110 |
| `implausibel` | `dy ≤ 0` (Linie über S1) | 45 |

Die Schwellen sind aus der Verteilung abgelesen (zwei getrennte Häufungen) und als Konstanten in der Pipeline änderbar.

Verwendung:
- Die **Richtung** der Linie dient bei `hip` und `isg` gleichermaßen als Beckenhorizontale, weil beide Landmarkenpaare symmetrisch am Becken liegen. Der Beckenschiefstand wird mit der Klasse in der Spalte `pelvis_ref` ausgewiesen.
- Den **Hüftkopfwinkel FHA** aus `degen_params` gibt es nur bei `hip`. Bei `isg` heißt der Wert `ISG_tilt`, bei `unklar`/`implausibel` wird er verworfen.
- Die Segmentwinkel selbst brauchen keine Beckenlinie, weil S1 als Node vorliegt.

### 3.5 Messfehler je Segment

SEM je Segment × Ebene = gepoolte SD zwischen Ratern innerhalb eines Bildes; MDC95 = 1,96 · √2 · SEM. Das Flag „relevante Änderung“ wird **erst in der App** gegen die aktuelle MDC95 gesetzt, nicht im Cache. MDC95 wird bei jedem Lauf aus allen annotierten Bildern neu berechnet (nicht nur aus der Kohorte, damit die Schätzung stabil ist). Die gespeicherten Δ alter Patienten bleiben dadurch gültig.

Einschränkung (steht im Bericht): Die Rater-SD enthält keinen Lagerungs- und Aufnahmefehler zwischen den Visiten, deshalb ist MDC95 eher zu klein. Ergänzend wird die Streuung zwischen Visiten am selben Tag verwendet, sofern es solche Bilder gibt (wie σ_s in `degen_pipeline`).

### 3.6 Therapie-Episoden

`THERAPY_MAP` (dict in `spine_geom/segmental.py`, erweiterbar; Schlüsselwörter für den Fallback in `KEYWORDS`):

| Rohwert `therapy` | Gruppe |
|---|---|
| konservativ, konservativ multimodale Komplextherapie, multimodale Komplextherapie, Epidurale Injektion | konservativ |
| Dekompression | Dekompression |
| Fusion | Fusion |
| Konversion in 4-Stab | Fusion, `revision=True` |
| Indikation | keine Therapie (ausgeschlossen) |
| leer | wird aus `therapy_specification` abgeleitet (Schlüsselwörter Fusion/Spondylodese/TLIF → Fusion, Dekompression/Laminektomie → Dekompression, Infiltration/PRT/Physiotherapie → konservativ), sonst ausgeschlossen |

Eine **Episode** ist ein Therapieeintrag (Datum `referring_date`):

- **Baseline:** letzte Visite **bis einschließlich** Ereignistag (höchstens 12 Monate davor, sonst Flag `baseline_old`). Im Cache vom 02.10. liegen Visiten am Ereignistag ausschließlich bei konservativen Episoden (11 von 11), und dort ist es die Aufnahme beim Vorstellungstermin. Das weicht bewusst von `spine_geom.course` ab, wo der Indextag als post zählt.
- **Verlauf:** alle Visiten nach dem Ereignistag bis einschließlich des nächsten Ereignistags. Diese Aufnahme ist zugleich Baseline der Folge-Episode.
  - `early` = erste Visite 0–3 Monate post
  - `late` = letzte Visite
  - Zeitklassen wie in `spine_geom.course.BINS`
- **Zeit:** relativ zum Ereignis und zu `index_date`.
- Fallen mehrere Einträge auf denselben Tag, werden sie zu einer Episode zusammengefasst. Bei Gruppenkonflikt gilt die Rangfolge Fusion > Dekompression > konservativ.
- „bis Os ilium / Ilium / Becken“ im Freitext heißt Strecke bis L5/S1 und `fused_to_pelvis`.

### 3.7 Implantate und Segmentklassen (nur Fusion)

**Inferenz:**
- Modell: `../Xray-Model/runs/implants-merge-20261120/implant-detector-…onnx` mit `onnxruntime` (im venv installiert).
- Vor- und Nachverarbeitung (`content_box`-Crop, Letterbox 800, Decode `implant-centernet-v2`, `merge`) kommen 1:1 aus `train_implants.ipynb`, in ein kleines Modul `implant_infer.py` gehoben, sodass Notebook und Pipeline denselben Code nutzen.
- Ausgabe je Implantat: Klasse, Mittelpunkt, `ends[0]` (Start = Schraubenkopf bzw. Einbringseite des Cages), `ends[1]`, Konfidenz.
- Cache `data/seg_implants.pkl`.
- Ausgewertet werden **alle Stehaufnahmen der Kohorte** (a.p. und seitlich), mit Cache je Bild (`seg_implant_dets.json`). Das ist einfacher als eine Auswahl, liefert die präop. Negativkontrolle und Vor-Instrumentierungen mit, und die Schrauben werden auch für den Slip gebraucht (§3.9).

**Höhenzuordnung:**
- Schraube: Höhe = Wirbel, dessen Centroid (Nodes derselben Aufnahme, Rater-Mittel je Label) dem Schraubenmittelpunkt entlang der lokalen Wirbelsäulenachse am nächsten liegt. Liegt er mehr als 0,6 Wirbelabstände entfernt, gibt es keine Höhe (Schraube außerhalb des annotierten Bereichs).
- Cage: Höhe = Segment, dessen zwei Centroide den Cage-Mittelpunkt einschließen.
- Iliumschraube bedeutet `fused_to_pelvis`.
- Hat ein postop. Bild keine Nodes, wird es nicht ausgewertet (es zählt dann nicht als Visite).
- Seite a.p. (Patientenseite, radiologische Konvention): Schraube = Lage zum Centroid; Cage = Seite von `ends[0]`.

**Fusionsstrecke:**
- Strecke = [UIV, LIV] aus den Schraubenhöhen der **ersten Verlaufsvisite** (a.p. ∪ seitlich); Iliumschrauben verlängern bis S1 und setzen „Becken“. Schrauben an der Baseline-Visite werden als Vor-Instrumentierung ausgewiesen (`implant_pre`).
- Gegenprobe mit den aus `therapy_specification` geparsten Höhen und `fused_segments`; Abweichungen kommen in eine QC-Liste.
- **Regel nach QC (24.11.2026, `choose_fusion`):** Verwendet wird der Text, wenn seine Segmentzahl zu `fused_segments` passt, sonst die Implantat-Strecke (`fused_new`, `fused_src`). Insgesamt fusioniert ist `fused_new` vereinigt mit der schon präop. instrumentierten Strecke (`fused_total`); daraus kommen die Segmentklassen. Hintergrund: Die Implantat-Strecke ist an UIV/LIV um ±1 unsicher (die Rater zählen die Wirbel auf a.p. und seitlich teils verschieden), der Text ist teils unvollständig.

**Parser:** `parse_levels(text)` erkennt `L4/5`, `LWK 3/4`, `L5/S1`, `LWK 5/SWK 1`, `Th12/L1`, `BWK 12/LWK 1` sowie Aufzählungen und Bereiche („L2–L5“) und liefert Segmente im Format `L4/L5`.

**Segmentklasse je Episode** (Ebene Segment):

| Gruppe | Klassen |
|---|---|
| Fusion | `fusioniert`, `UIV+1`, `UIV+2`, `LIV−1` (falls nicht Becken), `fern` |
| Dekompression | `dekomprimiert`, `benachbart`, `fern` |
| konservativ | `alle` (Referenz für den Spontanverlauf) |

**Cage-Seite (a.p.):**
- Einbringseite = Seite von `ends[0]` relativ zum Cage-Mittelpunkt (L/R).
- Bezug = `open_side` des Cage-Segments präop. (a.p.-Segmentwinkel).
- Daraus folgt `cage_approach ∈ {konvex, konkav, neutral}`; neutral heißt \|α_prä\| < MDC95. Gespeichert wird nur die Cage-Seite. Der Vergleich mit `open_side` hängt von der aktuellen MDC ab und wird deshalb in der Auswertung gebildet (§3.8).
- Weil Cage und Segmentwinkel aus demselben Bildsystem stammen, ist das Ergebnis invariant gegenüber Spiegelung, auch wenn die Seitenkonvention falsch angenommen ist.
- Liegt nur ein seitliches Bild vor, gibt es keine Cage-Seite.

### 3.8 Inkrementelle Berechnung

- **Schlüssel ist der Patient (`uuid`).** Jeder verarbeitete Patient hat in `seg_episodes.pkl` mindestens eine Zeile, auch ohne auswertbare Episode (Ausschlussgrund z. B. `no_therapy`). Die Menge der fertigen Patienten ist damit `seg_episodes.patient.unique()`, eine eigene Statusdatei ist nicht nötig.
- **Ablauf eines Laufs:**
  1. `degen_pipeline.run(token)` holt den aktuellen Gesamtstand.
  2. Neu sind alle aktuell validierten Patienten ohne Eintrag im Cache.
  3. Nur für diese werden Segmentwinkel, Slip (Modellabfrage nur für ihre Messungen), Visiten, Episoden, Implantat-Inferenz (nur ihre Bilder) und Änderungen berechnet und an die `seg_*`-Tabellen angehängt.
- **Ein Patient gilt erst als berechnet, wenn alle Eingaben da sind.** Der Button „Nur neue Patienten rechnen (Cache)“ läuft ohne Token und kann das Slip-Modell nicht abfragen. Fehlen für einen neuen Patienten Modellwerte (keine manuellen Ecken, nichts in `seg_corner_suggestions.json`), wird er übersprungen und als „wartet auf Token“ gemeldet. So bleibt kein Patient dauerhaft ohne Slip im Cache.
- Später ergänzte manuelle Ecken bei einem schon berechneten Patienten übernimmt erst `--rebuild` (gewollt: alte Fälle werden nicht neu gerechnet).
- **Neu berechnet wird bei jedem Lauf nur `seg_mdc.pkl`**, weil es sich aus allen Bildern speist. Alles, was von MDC95 abhängt (Flag \|Δ\| > MDC95, `cage_approach = neutral`), wird erst in der App ausgewertet; die gespeicherten Werte alter Patienten bleiben gültig.
- **`python segmental_course_pipeline.py --rebuild`** verwirft die `seg_*`-Tabellen und rechnet alles neu. Das ist nur nach einer Codeänderung nötig; jede Zeile trägt `pipeline_version`, damit gemischte Stände sichtbar sind.
- **Status-Entzug:** Die App zeigt nur Patienten, die *aktuell* validiert sind. Verliert ein Patient den `VALID`-Tag, bleibt er im Cache, wird aber nicht angezeigt.

### 3.9 Segmentaler Slip (sagittal)

**Größe:**
- Slip je Segment T12/L1–L5/S1 in % WKD; > 0 = Anterolisthese, < 0 = Retrolisthese.
- Definition wie Ground Truth und Modell (`train_slip.ipynb`): Abstand der PI-Ecke des oberen Wirbels zur Hinterkantenlinie PS–PI des unteren (`posterior_wall.slip`), **auch für L5/S1**. Die Rater setzen bei S1 ebenfalls PS/PI an die Hinterkante. `sacral_slip` (Meyerding) nutzt nur die klassische Kantensuche und wird hier nicht verwendet. (Korrektur 24.11.2026 gegenüber dem ersten Entwurf.)
- Der Wert in px wird durch die WKD aus der `calibrationLine` der Messung geteilt. Das Modell bekommt dieselbe WKD als `wkd` übergeben, damit beide Quellen gleich normiert sind.
- Nur seitliche Stehaufnahmen (`Xray-LWS-lat`, `Xray-WS-lat`) mit fester Orientierung links = anterior; das entspricht auch der Annahme des Modells. MRT und incl/recl sind ausgeschlossen (Nutzerentscheidung).

**Quellen, je Messung × Segment, mit Spalte `slip_source`:**
1. **Manuell** aus `data/corner_annotations.csv` (Canvas-Koordinaten wie die Nodes). Gibt es Durchgang 1 und 2 (blinde Wiederholung), zählt der Mittelwert; die Differenz geht in den Messfehler ein.
2. **Modell** über `POST /api/suggestcorners/` mit den Nodes derselben Messung (Rater-Nodes, in Original-px umgerechnet) und der WKD.
   - Ergebnis je Segment: `slip`, `unc`, `listhesis`
   - Cache `data/seg_corner_suggestions.json`, Schlüssel `m_uuid`; abgefragt wird nur, was noch fehlt
   - Ein Aufruf je Messung; danach wird wie bei den Winkeln über die Rater gemittelt (SD zwischen Ratern bleibt erhalten)
3. Die klassische Kantensuche (`posterior_wall.py`, MAE 12,4 % WKD) wird nicht verwendet.

**Instrumentierte Segmente:**
- Sind beide Wirbel eines Segments instrumentiert, ist das Modell nachweislich wertlos (Ausgabe = Prior). Dann gilt nur der manuelle Wert, sonst entfällt das Segment (`slip_source = instrumentiert`).
- Instrumentiert heißt: Implantat-Detektion (§3.7) oder `instrumented` in `corner_annotations.csv`.
- Ist nur ein Wirbel instrumentiert, wird der Modellwert behalten, aber mit Flag `slip_instr_one` markiert.

**Messfehler (§3.5 analog):**
- manuell: SEM aus der Differenz Durchgang 1 vs. 2, SEM = SD(d)/√2
- Modell: SEM aus dem OOF-Fehler des Slip-Laufs, MAE ≈ 4,5 % WKD auf seitlichem Röntgen, SEM ≈ 1,25 · MAE (Normalannahme)
- Für Δ mit Quellenmix prä/post: MDC95 = 1,96 · √(SEM_prä² + SEM_post²), Flag `slip_source_mixed`

**Bekannte Modellverzerrung:**
- Große Slips werden auf etwa 60 % geschrumpft (Steigung Modell~GT 0,59), Δ bei großen Slips also unterschätzt. Es wird nicht nachkalibriert.
- Als Sensitivitätsanalyse rechnet jede Slip-Auswertung zusätzlich nur mit Segmenten, die prä und post manuell annotiert sind.

## 4. Auswertung je Fragestellung

Grundgröße je Episode × Segment × Größe (koronarer Winkel, sagittaler Winkel, Slip): `Δ = α_post − α_prä` (mit Vorzeichen) und `K = |α_prä| − |α_post|` (Korrektur, > 0 = verbessert; beim Slip = Reposition). Beides wird für `early` und `late` berechnet. Ausgewertet wird stets deskriptiv: Median/IQR, Bootstrap-CI über Patienten (1000 Ziehungen, Patient als Einheit) und Anteil \|Δ\| > MDC95.

| # | Frage | Umsetzung | Darstellung |
|---|---|---|---|
| 1 | Welche Veränderungen erzielte welches Verfahren, je Segmenthöhe? | Δ/K je Gruppe × Segmenthöhe × Größe (koronar, sagittal, Slip), zusätzlich je Segmentklasse; beim Slip zusätzlich der Anteil Segmente mit Listhese (\|Slip\| ≥ 10 % WKD) prä vs. post | Heatmap Segment (Zeilen) × Gruppe (Spalten), Farbe = Median Δ, Markierung für CI ohne 0; Tabelle mit n |
| 2 | Kopplung koronar ↔ sagittal: geht die Skoliosekorrektur zu Lasten der Lordose? | je Episode: koronare Korrektur (K der Hauptkurve `cCobb_max` und Σ K der fusionierten Segmente) gegen ΔLL (T12–S1 und L4–S1) und gegen Δ sagittal am selben Segment. Zusätzlich je Segment: Reposition des Slips (K_slip) gegen Δ sagittalen Segmentwinkel. Geht Reposition mit Lordosierung einher? | Streudiagramm + Spearman ρ mit Bootstrap-CI, getrennt nach Gruppe; Segmentebene als Facetten |
| 3 | Hängt die Cage-Richtung mit der Öffnungsrichtung zusammen? | Kreuztabelle Einbringseite × `open_side` präop. → Anteil konvex/konkav; K_koronar und Δ_sagittal am Cage-Segment konvex vs. konkav (Median-Differenz, Bootstrap-CI) | Kreuztabelle, Boxplot K nach `cage_approach` |
| 4 | Größere segmentale Korrektur bei langstreckiger Fusion? | kurz = 1–2 Segmente, lang = ≥ 3 oder Becken; mittlere K je fusioniertem Segment und K je Höhe; zusätzlich relative Korrektur K/\|α_prä\|, weil lange Fusionen größere Ausgangskurven haben (Confounding) | Boxplot kurz/lang, Streudiagramm K vs. Anzahl Segmente, eingefärbt nach \|α_prä\| |
| 5a | Segmental → regional → global | Δ der Hauptkurve = Σ Δ der Segmente in der Kurve (exakt, §3.3) → Anteil je Segment; regional: `cCobb_max`, LL; global: C7-CSVL, L1PA, T1PA, SVA aus `degen_params` (nur wenn messbar, meist nur WS) | gestapelte Balken „Beitrag je Segment“, Tabelle Δ regional/global |
| 5b | Nachbarsegmente bei kurzstreckiger Fusion | Δ in `UIV+1/+2`, `LIV−1` vs. `fern` und vs. denselben Höhen bei konservativen Episoden (Spontanverlauf), auf die Zeit normiert (°/Jahr bzw. % WKD/Jahr). Neu auftretende Listhese bzw. Retrolisthese im Nachbarsegment wird als eigenes Ereignis gezählt (Anschlussdegeneration) | Punktdiagramm je Klasse mit CI |
| 6 | Ging die Korrektur verloren? In welchem Segment, in welchem Zeitraum? | operative Episoden: `Verlust = α(t) − α_early`, als Betrag in Richtung der Ausgangsdeformität, für alle drei Größen (beim Slip: erneutes Abgleiten nach Reposition); je Segment und Zeitklasse; Ereignis = erstes Überschreiten von MDC95; Bezug zu `long_term_complication` (Stabbruch, Schraubenlockerung) und zu Implantatänderungen in der Folgedetektion | Verlaufskurven je Segment(-klasse), Tabelle „Anteil mit Verlust je Zeitklasse“, Liste der Ereignisse mit Patient/Segment/Monat |

Zur Einordnung werden Kurvenrichtung und Hauptkurve aus `spine_geom.coronal.curves` als Spalten mitgeführt; gepoolt wird trotzdem nur über Beträge bzw. K.

## 5. Module und Dateien

```
spine_geom/segmental.py         segment_angles, segment_slips_pct (nutzt posterior_wall.slip/sacral_slip), parse_levels,
                                segment_classes, cage_approach (rein, ohne I/O)
spine_geom/tests/test_segmental.py
implant_infer.py                ONNX-Vor-/Nachverarbeitung aus train_implants.ipynb (gemeinsam genutzt)
segmental_course_pipeline.py    degen_raw/entries/params → Tabellen unten; python segmental_course_pipeline.py = neu rechnen
segmental_course_app.py         Streamlit, Token in der Sidebar (nur für Implantat-Inferenz/Refresh)
reports/segmental_therapieeffekt.md
```

| Cache (`data/`) | Zeile = | Inhalt |
|---|---|---|
| `seg_angles.pkl` | Messung × Segment × Ebene | Winkel, `open_side`, Randflag, Rater, Bild, img_type |
| `seg_slip.pkl` | Messung × Segment | Slip (% WKD), `slip_source` (manuell/Modell/instrumentiert), unc (Modell), Durchgangsdifferenz (manuell), Flags |
| `seg_corner_suggestions.json` | Messung (`m_uuid`) | Rohantwort von `/api/suggestcorners/` (Ecken, Slip, unc); nur fehlende werden abgefragt |
| `seg_visits.pkl` | Patient × Visite × Segment × Größe | Mittel, SD, n_rater, n_bilder, view, Slip-Quelle |
| `seg_mdc.pkl` | Segment × Größe (× Slip-Quelle) | SEM, MDC95, df |
| `seg_episodes.pkl` | Episode | Patient, Datum, Gruppe, Revision, Baseline-/early-/late-Visite, Monate ab Ereignis/Index, Fusionsstrecke, Quelle (Implantat/Text), QC-Flags |
| `seg_implants.pkl` | Implantat | Bild, Klasse, Mittelpunkt, ends, Konfidenz, Höhe, Seite |
| `seg_changes.pkl` | Episode × Segment × Ebene × Zeitpunkt | α_prä, α_post, Δ, K, Segmentklasse, Cage-Seite (MDC-abhängige Flags erst in der App, §3.8) |
| `seg_pelvis.pkl` | a.p.-Messung | `pelvis_ref` (hip/isg/unklar/implausibel), w, dy, Beckenschiefstand |

Alle `seg_*`-Tabellen außer `seg_mdc` tragen `patient` und `pipeline_version` und werden nur angehängt (§3.8).

**App-Tabs:**
1. Abdeckung (Schritt 0: Episoden je Gruppe, auswertbar vs. ausgeschlossen mit Grund, neu im letzten Lauf)
2. Episode (Einzelfall: Bilder prä/post mit Nodes und Implantaten, Segmenttabelle; Plausibilitätsprüfung)
3. Verfahren × Segment (F1)
4. Kopplung (F2)
5. Cage (F3)
6. Fusionslänge (F4)
7. Segmental → regional/global (F5)
8. Korrekturverlust (F6)

## 6. Reihenfolge und Abnahme

| Schritt | Inhalt | Abnahme |
|---|---|---|
| 0 | Kohortenfilter (`VALID`), Episoden bauen (THERAPY_MAP, Parser), inkrementeller Cache, Abdeckungstab | Tabelle Episoden je Gruppe, auswertbar vs. ausgeschlossen mit Grund; zweiter Lauf ohne neue Patienten rechnet nichts (Log „0 neue Patienten“) |
| 1 | `spine_geom/segmental.py` + Tests | pytest grün: synthetischer Keil → richtiges Vorzeichen und Seite (seitlich mit fester Orientierung links = anterior); Σ Segmente = regionaler Spline-Winkel (< 0,01°); Parser an allen `therapy_specification`-Werten im Cache; Beckenlinien-Klassifikation an je einem synthetischen hip-/isg-Fall |
| 2 | `seg_angles`, `seg_visits`, `seg_mdc`; Tab Episode | 5 Stichproben visuell plausibel; Segmentwinkel vs. gespeicherte Frontend-Spline-Segmentwinkel (`cobbType` Spline, segmental): Bland-Altman ≈ 0 |
| 2b | `seg_slip` (manuell + `/api/suggestcorners/` mit Cache), Slip in `seg_visits`/`seg_mdc`, Ecken im Tab Episode | Test: synthetische Ecken → bekannter Slip und Vorzeichen (Antero > 0), L5/S1 über `sacral_slip`. Auf Segmenten mit beiden Quellen: Modell vs. manuell, MAE in der Größenordnung des OOF-Werts (≈ 4,5 % WKD) und keine Vorzeichenumkehr; sonst stimmt die Koordinaten- oder WKD-Umrechnung nicht |
| 3 | `implant_infer.py`, Höhenzuordnung, Fusionsstrecke, Cage-Seite | Übereinstimmung Fusionsstrecke Implantat vs. Text in der QC-Liste; Cage-Seite an 10 Fällen manuell geprüft |
| 4 | `seg_changes` + Tabs F1–F6 + Bericht | Machbarkeit: Jede der sechs Fragen liefert für die Startkohorte eine Tabelle/Abbildung mit n je Zelle, auch wenn n = 1. Bericht als „vorläufig, Machbarkeitsnachweis“ gekennzeichnet; CIs erst ab n ≥ 5, darunter nur Einzelwerte |

## 7. Risiken und Grenzen

- **Kleine Fallzahl** (§2): Ziel ist der Machbarkeitsnachweis, nicht die Antwort. Die Ergebnisse sind vorläufig und werden immer mit n angegeben. Belastbar wird es erst mit weiteren validierten Patienten, die per inkrementellem Lauf dazukommen.
- **Keine Lückenfüllung:** Fehlt bei einem validierten Patienten etwas (Therapiefeld leer und nicht ableitbar, postop. Bild ohne Nodes), fällt die Episode bzw. das Segment heraus. Das kann systematisch sein, z. B. wenn instrumentierte Bilder seltener annotiert werden; die Ausschlussgründe in Tab 1 machen das sichtbar.
- **Beckenlinien-Klassifikation** (§3.4) ist eine Schwellenregel; etwa 16 % der a.p.-Messungen bleiben `unklar` und fallen für FHA/ISG_tilt weg.
- **Confounding durch Indikation:** Lange Fusionen und Dekompressionen betreffen andere Ausgangsbefunde als konservative Verläufe. Konservative Episoden sind Referenz für den Spontanverlauf, keine Kontrollgruppe.
- **Annotation postop. Bilder:** Implantate verdecken Wirbel; die Rater-SD auf instrumentierten Bildern wird getrennt ausgewiesen.
- **Spline-Randeffekte:** Die Frontend-Spline interpoliert und ist am ersten/letzten Node einseitig, daher das Randflag. Der Bereich des Raters muss das Segment ±1 Wirbel umfassen, sonst wird das Segment nicht ausgewertet.
- **Seitenkonvention a.p.:** Sie ist angenommen (`side_assumed`). Für die Cage-Frage ist das irrelevant (§3.7), für die absolute Links/Rechts-Darstellung nicht.
- **MDC aus Rater-Streuung** unterschätzt den Messfehler zwischen Visiten (§3.5).
- **Slip-Modell:** Es schrumpft große Slips (§3.9), ist auf seitlichem Röntgen sein schwächster Bildtyp, und auf doppelt instrumentierten Segmenten ist es nicht nutzbar. Gerade postop. fusionierte Segmente haben deshalb nur dann einen Slip, wenn Ecken manuell annotiert sind. Die Sensitivitätsanalyse „nur manuell“ zeigt, wie stark das Modell die Ergebnisse treibt.
- **Kein koronarer Versatz:** Eine laterale Listhese auf a.p. ist mangels Ecken auf a.p.-Bildern nicht messbar. Nur der sagittale Slip geht ein.
- **Keine Instabilitätsmessung:** Funktionsaufnahmen bleiben draußen (Nutzerentscheidung). Ein im Stand reponierter, aber dynamisch instabiler Slip fällt nicht auf.

## 8. Umsetzungsstand

**Schritt 0 (24.11.2026), umgesetzt:**
- `spine_geom/segmental.py` (`parse_levels`, `therapy_group`, `therapy_events`, `episodes`) mit Tests in `spine_geom/tests/test_segmental.py`
- `segmental_course_pipeline.py` (Kohortenfilter, inkrementeller Cache `data/seg_episodes.pkl`, `--rebuild`)
- `segmental_course_app.py` mit dem Tab Abdeckung, Startbefehl in `.claude/launch.json`
- `degen_pipeline.RAW_COLS` enthält jetzt die Bild-`uuid`, die für `image_groups` gebraucht wird. Alte Caches haben sie nicht; dann werden die Gruppen ignoriert und es gibt einen Hinweis.

**Probelauf ohne Kohortenfilter** (alle Patienten, Cache 02.10.): 48 Therapie-Episoden, davon 15 auswertbar.

| Gruppe | Episoden | auswertbar | davon a.p. + seitlich |
|---|---|---|---|
| Fusion | 8 | 5 | 5 |
| Dekompression | 8 | 3 | 2 |
| konservativ | 32 | 7 | 6 |

Der Rest scheitert an fehlender Baseline (11) oder fehlendem Verlauf (22). Mit der alten Regel (Ereignistag = post) wären es 4 konservative Episoden mehr gewesen; deren einziger „Verlauf“ war die Aufnahme am Vorstellungstag.

**Erster Lauf mit Kohortenfilter** (API-Stand 24.11.2026): 46 validierte Patienten mit 66 Therapie-Episoden; 3 Patienten haben kein Therapieereignis.

| Gruppe | Episoden | auswertbar | davon a.p. + seitlich | kein Verlauf | keine Baseline |
|---|---|---|---|---|---|
| Fusion | 37 | 36 | 35 | 1 | 0 |
| Dekompression | 5 | 2 | 2 | 2 | 1 |
| konservativ | 24 | 8 | 8 | 14 | 2 |

- Die Fusionen tragen den Machbarkeitsnachweis.
- Bei vielen Fusionen ist der späteste Verlauf nur die frühe postop. Aufnahme (wenige Tage), Korrekturverlust (F6) ist also nur bei einem Teil auswertbar.
- 61 Episoden ordnet `THERAPY_MAP` zu, 5 der Schlüsselwort-Fallback.
- Am Ereignistag liegen 12 konservative und 3 operative Visiten. Die operativen sind zweimal der zweizeitig operierte Patient 4AEM (Aufnahme am Tag der 2. OP = Zustand nach der 1. OP) und einmal 27NG. Als Baseline zu zählen ist dort vertretbar, aber nicht eindeutig. Die Implantat-Detektion in Schritt 3 kann das prüfen.

**Schritt 1 (24.11.2026), umgesetzt:** `frontend_spline`, `node_tilts`, `segment_angles`, `pelvis_ref` in `spine_geom/segmental.py`, 74 Tests grün.

- **Frontend-Abgleich:** Bei 3.552 Stehaufnahmen mit gespeicherten `Spline-segmental`-Werten stimmt round(\|α\|) bei 85 % der Messungen in *allen* Segmenten exakt. Die übrigen 526 Messungen passen zu keiner Algorithmus-Variante: weder ohne Tangentenfenster, noch abgerundet, noch mit dem alten interpolierenden Spline. Sie stammen zu 78 % von zwei Ratern (IDs 20 und 8), mit einer Abweichung im Median von 2°. Wahrscheinlich sind dort die gespeicherten Winkel veraltet, z. B. weil Nodes verschoben wurden, ohne dass die Winkel neu berechnet wurden. Die Auswertung rechnet ohnehin aus den Nodes; als Paritätstest im Code gilt deshalb die Schwelle ≥ 80 %.
- Der erste Versuch mit `bspline_python.interpolate_bspline` traf nur 11 % und lieferte systematisch größere Winkel. Das war der Grund für den Nachbau.
- `MIN_NODES` in der Pipeline ist jetzt 4 (vorher 3), weil das Frontend ab 4 Nodes rechnet. Die Episoden wurden neu gerechnet (`pipeline_version = s1-2026-11-24`), die Zahlen sind unverändert.

**Schritt 2 (24.11.2026), umgesetzt:**
- Pipeline: `seg_angles.pkl` (5.786 Segmentwinkel aus 631 Messungen der Kohorte), `seg_visits.pkl` (Rater-Mittel je Bild, dann je Visite), `seg_mdc.pkl` (aus allen Stehaufnahmen, neu sobald `degen_raw.pkl` neuer ist; Randsegmente mit Spline-Fallback ausgenommen).
- `episode_angles` liefert Baseline/early/late je Ebene mit eigenen Zeitpunkten, dazu Δ und K.
- Tab **Episode**: Segmenttabelle mit MDC95-Flag, daneben Röntgenbild prä/post mit Nodes, Frontend-Spline und Senkrechten, auf die Wirbelsäule zugeschnitten.
- **Messfehler** (MDC95, seitlich/a.p.): Kernsegmente T12/L1–L4/L5 etwa 5,5–7,5°. L5/S1 seitlich 13,3°, a.p. 11,3°: Die Rater setzen die Deckplatte unterschiedlich, und die L5-Tangente hängt an der L5-Node.
- **Plausibilität:**
  - Seitlich ist die Summe L1–S1 bei 99,3 % der Visiten lordotisch. Der Median steigt mit der S1-Deckplatte von 37° auf 47°.
  - Overlay-Stichprobe (03KB): Schrauben L1–L3 postop. passen zu den Segmenten aus `therapy_specification`. Seitlich ist anterior links, und L3/L4 sowie L4/L5 sind lordotisch.
- **Vergleich mit den gezeichneten Cobb-segmental-Winkeln:**
  - L5/S1 (Deckplatte): Bias +2,6°, SD 5,2°, r = 0,62, n = 241.
  - L3/L4 und L4/L5 (Spline): systematisch 6–7° größer, r ≈ 0,4–0,5. Das ist erwartbar, weil Centroid zu Centroid halbe Wirbelkörperkeile mitmisst, die gezeichnete Linie dagegen nur den Bandscheibenraum.
- **Einzel-Rater-Visiten sind laut:** z. B. 03KB prä L5/S1 32,3° (ein Rater, seine eigene Cobb-Linie 11°). Die Overlay-Beschriftung zeigt „Rater x von n“.

**Schritt 2b (24.11.2026), umgesetzt:**
- **Rechnung:** `spine_geom.segmental.corner_slips`. Getestet: 10 px anterior bei 50 px WKD ergeben +20 %, posterior ergibt −20 %.
- **Pipeline:**
  - `manual_slips` liest die manuellen Ecken (`corner_annotations.csv`): 184 Bilder, 1.028 Segmente, 112 Segmente mit zwei Durchgängen. Fehlt die WKD, wird sie über das Median-Verhältnis WKD/Nodeabstand geschätzt.
  - `fetch_corner_suggestions` fragt `POST /api/suggestcorners/` ab. Nodes und WKD gehen in Original-px; der Faktor kommt aus dem Bild-Header, die Bild-URL wird ohne Token geladen. Cache `seg_corner_suggestions.json` je `m_uuid`. Inferenzfehler (z. B. kein aktives Modell für `Xray-WS-lat`) werden gecacht und liefern keinen Slip, Netzfehler nicht.
  - `slips_of`: Je Bild × Segment hat „manuell“ Vorrang. Ein manuell annotiertes Bild braucht kein Modell.
- **Tabellen:** `seg_slip.pkl`; Slip in `seg_visits` und `seg_mdc` als `plane = 'slip'`. Damit gelten `episode_angles`, Δ und K unverändert; K entspricht der Reposition.
- **Warten auf Token:** Ohne Token werden Patienten mit fehlenden Modellwerten nicht angelegt („wartet auf Token“). Stand nach `--rebuild` ohne Token: 3 von 46 Patienten vollständig manuell, 43 warten (286 seitliche Messungen ohne Modellwert).
- **Messfehler:**
  - SEM manuell (Durchgang 1 vs. 2) 3,7 % WKD.
  - Modell: angenommen 1,25 · 4,5 = 5,6 % WKD.
  - Je Segment gilt vorsichtig die größere Zahl: MDC95 = 15,6 % WKD. Nur große Slip-Änderungen gelten damit als echt.
- **Tab Episode:** dritte Tabelle „Slip“ mit Spalte „Quelle“ (prä → post); im Overlay die Hinterkanten (grün), manuell vor Modell.
- **Stichprobe PTZN** (Fusion L3–S1, beide Zeitpunkte manuell): L5/S1 von +19,1 % auf +8,4 % (Reposition +10,7 %), L4/L5 von +10,8 % auf −9,9 %.
- **Abnahme Modell vs. manuell** (`--check-slip`, 40 seitliche LWS-Bilder, 226 Segmente): MAE 2,1 % WKD, Bias +0,1, r = 0,97, Vorzeichen bei \|GT\| ≥ 10 % zu 100 % gleich. Die Umrechnung von Koordinaten und WKD stimmt damit.
  - Die MAE ist besser als der OOF-Wert (4,5), weil die manuell annotierten Bilder überwiegend Trainingsbilder des Modells sind. Für den Messfehler bleibt deshalb der OOF-Wert maßgeblich (SEM Modell 5,6 % WKD).
- **Modellabfrage für die Kohorte (24.11.2026):** 326 Messungen, 245 mit Antwort. 81 bekamen 404, weil es kein aktives Hinterkanten-Modell für `Xray-WS-lat` gibt; das Modell ist nur auf LWS/MRT trainiert. `seg_slip`: 1.224 Modell- und 85 manuelle Werte.
- **Slip-Änderung (Δ late) vorhanden** in 33 von 46 auswertbaren Episoden: Fusion 24 von 36, Dekompression 2 von 2, konservativ 7 von 8. Den übrigen Fusionen fehlt der Slip vermutlich wegen Ganzwirbelsäulen-Aufnahmen (WS-lat).
- **Noch offen:**
  - Slip auf `Xray-WS-lat`: Modell dafür freigeben bzw. trainieren, oder WS-lat in `corner_annotate_app.py` aufnehmen und manuell annotieren.
  - Das Verwerfen von Modellwerten an doppelt instrumentierten Segmenten folgt mit der Implantat-Detektion in Schritt 3. Bis dahin gilt `instr_both` nur für manuell annotierte Bilder, wo ohnehin der manuelle Wert zählt.

**Schritt 3 (24.11.2026), umgesetzt:**
- **Inferenz:** `implant_infer.py` (onnxruntime, ohne Torch; Konstanten aus `results.json`).
  - Abnahme gegen `implant_annotations.csv` (30 Bilder, überwiegend Trainingsbilder, daher optimistisch):

    | Klasse | Recall | Präzision |
    |---|---|---|
    | Pedikelschraube (152) | 99 % | 100 % |
    | Cage (75) | 100 % | 99 % |
    | Iliumschraube (14) | 100 % | 100 % |

    Mittelpunkt ≈ 1 Stage-px, Richtung Start → Ende zu 100 % richtig. Die Vorverarbeitung entspricht damit dem Notebook.
  - Prüfaufruf: `python implant_infer.py --check`.
- **Kohorte:** 306 Stehaufnahmen, 1.986 Implantate (1.437 Pedikel-, 146 Iliumschrauben, 403 Cages), 155 Bilder ohne Implantat. 24 Schrauben oder Cages bekamen keinen Wirbel.
  - Test: Höhe über die Schraubenspitze statt über den Mittelpunkt bringt nichts (a.p./seitl. stimmen in 67 % statt 71 % überein).
- **QC Fusionsstrecke** (37 Fusionen): 22 gleich, 10 abweichend, 3 nur Implantat, 2 nur Text. Die 10 Abweichungen:
  - 2× Revision/Verlängerung: Der Text nennt die neuen Segmente, das Bild zeigt die ganze Instrumentierung.
  - 4× UIV ±1.
  - 4× Implantat-Strecke länger, darunter K95A: Text unvollständig, Implantat passt zu `fused_segments`.
  - Nach `choose_fusion`: 26 × Text, 11 × Implantat.
- **Cage-Seite:** an 10 a.p.-Fällen visuell geprüft. Der Startpunkt liegt immer auf der Bildseite, die der ausgegebenen Patientenseite entspricht. Wo das Seitenzeichen „R“ erkennbar ist (8/10), steht es links im Bild; die radiologische Konvention (`side_assumed`) hält.
- **Slip:** Modellwerte an Segmenten mit Schrauben in beiden Wirbeln (gleiches Bild) werden verworfen. `seg_slip` hat jetzt 1.038 statt 1.309 Werte.
- **App:**
  - Tab Episode: Implantate im Overlay (Start als Kreis), Zeile Fusionsstrecke (Implantat/Text/QC/Cages/präop. Schrauben) und Spalte „Klasse“ (fusioniert, UIV+1, LIV−1, …).
  - Tab Abdeckung: QC-Liste.

**Schritt 4 (25.11.2026), umgesetzt – Machbarkeitsnachweis erbracht:** Jede der sechs Fragen liefert für die Kohorte (46 auswertbare Episoden: 36 Fusion, 8 konservativ, 2 Dekompression) Tabellen mit n und, ab 5 Patienten, Bootstrap-CI.
- **Pipeline:**
  - Regionale/globale Parameter aus `degen_params` je Visite (`plane = 'regional'`).
  - `seg_changes.pkl`: je auswertbarer Episode × Größe × Segment × Verlaufsvisite mit Δ, K, Korrekturverlust und Segmentklasse.
  - Korrekturverlust = (α(t) − α_early) · sign(α_prä − α_early); > 0 heißt zurück in Richtung Ausgangswert. So gilt eine Definition für a.p., seitlich und Slip.
- **`segmental_analysis.py`:** F1–F6 als reine Pandas-Funktionen; `python segmental_analysis.py` schreibt `reports/segmental_therapieeffekt.md`. Tests in `spine_geom/tests/test_segmental_analysis.py` (CI erst ab 5 Patienten, MDC-Anteil, Sortierung, F6-Ereignis).
- **App:** Tabs F1–F6 (Heatmap Segment × Gruppe, Streudiagramme, Cage-Kreuztabelle, Verlaufslinien), Zeitpunkt early/late in der Sidebar, Button „Bericht schreiben“. Die erste Auswertung nach einem Pipeline-Lauf dauert ≈ 35 s (Bootstrap) und ist danach gecacht.
- **Erste Beobachtungen (vorläufig, nicht interpretieren):**
  - Fusion korrigiert die Hauptkurve (cCobb_max Δ Median −8,5°, CI −12,0; −4,4); konservativ nimmt sie zu (+9,7°).
  - Δ je fusioniertem Segment liegt um 0, weil sich links- und rechtskonvexe Segmente aufheben. Aussagekräftig ist K.
  - F2: Koronare Korrektur und ΔLL korrelieren bei Fusion positiv (ρ ≈ 0,5–0,6), d. h. kein Hinweis auf Lordoseverlust durch Skoliosekorrektur.
  - F3: 50 von 66 Cages sitzen an Segmenten mit |α_prä| < MDC95 („neutral“). Konvex/konkav ist deshalb nur bei wenigen Cages unterscheidbar.
  - F4: kein Zusammenhang zwischen Fusionslänge und mittlerer Segmentkorrektur (ρ = 0,09).

**Fehlerbehebungen beim ersten Lauf:**
- `prepare_outlier_data.fetch_data`: Timeout 120 s → (10 s, 600 s); die Antwort braucht inzwischen über 2 min.
- `spine_geom.coronal.tilts_spline`: Mindestzahl Punkte 4 → 5, weil scipys `make_smoothing_spline` mindestens 5 braucht (a.p.-Messung mit 4 Nodes in den neuen Daten); Test ergänzt.
