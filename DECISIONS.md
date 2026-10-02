# DECISIONS

Entscheidungen zur Umsetzung von `docs/SPEC_degenerative_skoliose_verlaufsmodell.md`.
Format: Datum · Entscheidung · Begründung/Quelle.

## 2026-09-29 – Phase 0 (Bestandsaufnahme, Interview)

### Aus Code/Daten beantwortet (Spec §8)

| # | Frage | Antwort | Quelle |
|---|---|---|---|
| 1 | Hüftmittelpunkte | `stateObject.hipLine` = Linie zwischen **beiden Hüftkopfzentren** (Endpunkte). H = Mittelpunkt. Vorhanden in a.p. **und** seitlich. Nur ein Hüftkopf ist nicht kodierbar (Linie braucht 4 Koordinaten). | `xray-annotator/src/helpers/measureCompute.ts` `globalParams()` |
| 2 | Deckplattenlinien | **Keine beschrifteten L1/S1-Linien.** `cobbLines` sind unbeschriftete Linien (px, Ursprung oben links, y nach unten, Endpunkte **ungeordnet**). Das Label ergibt sich aus den nächstgelegenen Nodes. S1-Deckplatte im Frontend = unterste `Cobb-regional`-Linie, deren Mitte unterhalb des L5-Nodes liegt. `suggestedLines` (paarweise INF/SUP) erst ab 09/2026. | `measureCompute.ts`, `docs/api_imagedata_structure.md` §7.4 |
| 3 | Blickrichtung seitlich | Nicht dokumentiert. Das Frontend leitet anterior aus dem Vorzeichen der Deckplattenneigung ab. | `bspline.ts` `spinopelvic()` |
| 4 | Annotierte Höhen | Meist T10/T11 → S1, Ganzaufnahmen ab C7. Übergangswirbel nicht kodiert (kein L6 im Label-Array). | Cache-Auswertung |
| 5 | Multi-Rater | Ja, 1–4 Messungen pro Bild, je mit `rater_id`. | `build_df4d()` |
| 6 | Kalibrierung | Kein Pixel Spacing. Nur `calibrationLine` über einem Referenzwirbel (px-Länge) → Distanzen in **Wirbeltiefe-Einheiten**, nicht mm. Stehend/liegend nicht vorhanden. | API-Doku §7.2 |
| 7 | Koronare Beckenreferenz | `hipLine` in 499/602 LWS-a.p.-Messungen. | Cache |
| 8 | Manuelle Cobb-Winkel | Ja, `Cobb-regional`-Linien in `angles`. | API |

### Im Interview entschieden

- **Umfang v1:** Phasen 0–5 als Streamlit-App. Phasen 6–8 später.
- **Deckplatten:** S1 = unterste `Cobb-regional`-Linie unterhalb von L5 (wie Frontend `*_cobb`). L1 = `Cobb-regional`-Linie, die dem L1-Node am nächsten liegt. Fallback ist jeweils die Spline-Tangente am Node mit QC-Flag `endplate_from_tangent`.
- **Abdeckung:** Matrix Patient × Visite × Parameter, dazu Zusammenfassungen „mind. 1×“, „an Baseline“ und „an ≥ 2 Visiten“.
- **Multi-Rater:** Jede Größe wird pro Rater berechnet. Pro Bild werden Mittelwert und Rater-SD ausgegeben. Die SD dient als vorläufige Messunsicherheit.
- **Labels:** Es werden die **k-Means-korrigierten Labels** verwendet (`detect_label_errors` / `apply_label_corrections`).
  ⚠ Das weicht bewusst von Spec §0.6 ab („nie über y-Sortierung“). Die Korrektur verschiebt nur das gesamte Label-Band um eine ganze Zahl. Die Identität bleibt also label-basiert, und die Messung wird nicht umsortiert. Korrigierte Messungen tragen das Flag `label_shift_corrected`.
- **Lagerung:** Xray-* gilt als stehend (Haltungsvektor erlaubt), MRI-* als liegend (nur Formvektor). `Xray-LWS-incl/recl` sind Funktionsaufnahmen und fließen nie in den Haltungsvektor ein, sondern nur in ein separates Flexibilitätsmaß.
- **Übergangswirbel:** vorerst ignoriert. Bekannte Lücke: Ausreißer bei der PI können daher rühren.
- **Reliable Change:** α = 0,05 ohne Korrektur für multiples Testen. Σ_haltung = 0 als **Platzhalter**, per Slider als skalare SD einstellbar. Jede Ausgabe trägt den Hinweis „Haltungsvariabilität nicht identifiziert“.

### Längennormierung über die Calibration Line (2026-09-29)

Alle Distanzen (SVA-Surrogat, Formvektor, Segmenthöhen, Hüfte–S1) werden in **Wirbelkörperdurchmessern (WKD)** angegeben: px / Länge der `calibrationLine`. Das gleicht Vergrößerung und Resizing zwischen Visiten aus.

Befund im Cache (1940 Messungen):
- Vorhanden in 96–98 % der LWS-Messungen, in 89–93 % der Ganzaufnahmen. Es fehlt also nicht nirgends, aber auch nicht überall.
- Die Linie liegt annähernd horizontal (Median 6–12°). Seitlich entspricht sie der **Tiefe**, a.p. der **Breite**. Die beiden Einheiten sind nicht austauschbar, a.p. und seitlich werden getrennt normiert.
- Die Längen verschiedener Rater streuen um etwa 4 % (Median-CV). Das geht als Unsicherheit in jede normierte Distanz ein (Phase 2).
- **Messanweisung:** L3 oder L4, je nachdem, welcher besser sichtbar ist. Der Wechsel zwischen den Visiten (51 % der Kombinationen Patient × img_type) ist also so gewollt und wird umgerechnet. Referenz ist L4.
- **Zielwirbel geometrisch bestimmen,** nicht über `calibration.target`. Maßgeblich ist der Node, der der Mitte der Linie am nächsten liegt, mit den k-Means-korrigierten Labels. Grund: `correct_measurement_label_shift()` verschiebt `calibration.target` nicht mit, bei Label-Shift-Messungen ist das gespeicherte Ziel deshalb falsch.
- **Faktor L3/L4** aus 201 Bildern, an denen Rater verschiedene Zielwirbel gewählt haben (Cache 03.08.):
  - seitlich: Median 0,996 (n = 88), also **kein Faktor**
  - a.p.: Median 0,969 (n = 113), also **Faktor 0,97**, anatomisch plausibel, weil die Breite nach kaudal zunimmt
  - Die Faktoren werden bei jedem Fetch neu geschätzt, in der App angezeigt und in `norms.yaml` als „intern geschätzt“ abgelegt.
- **Abweichungen von der Anweisung:** 23 % der Kalibrierungen liegen auf anderen Levels (L5 298×, L2 81×, L1 47×, thorakal/S1 ca. 40×).
  - Für L5, L2 und L1 wird der Faktor nach derselben Logik geschätzt, sofern es mindestens 20 Bildpaare gibt. Sonst gilt die Kalibrierung als ungültig und der Fallback greift. Flag `calib_off_protocol`.
  - Thorakale Level und S1 sind immer ungültig und gehen in den Fallback.
  - Diese Liste ist auch für die Rater-Schulung nützlich.
- Fehlt die Linie, gibt es einen patientenspezifischen Fallback: das Verhältnis aus mittlerem lumbalen Node-Abstand und WKD aus anderen Visiten desselben img_type. Flag `calib_imputed`. Der Node-Abstand enthält die Bandscheibenhöhe und driftet deshalb bei Degeneration.
- Eine Umrechnung in mm ist nur optional möglich, über eine Norm-Wirbeltiefe nach Level und Geschlecht aus `norms.yaml`. Sie ist gekennzeichnet und nur für den Vergleich mit Normwerten wie dem SVA gedacht.

### Umsetzung Schritt 1a/1b (2026-09-29)

- **Deckplatten-Zuordnung** (`landmarks._pick_endplate`):
  - Das Fenster für die Linienmitte reicht von der Hälfte des Abstands zum kranialen Nachbar-Centroid bis zum Centroid. Bei S1 reicht es bis ½ Abstand darunter.
  - Horizontal darf die Linienmitte weniger als 1 Abstand vom Centroid entfernt sein.
  - Bei S1 wird die tiefste Linie genommen, bei L1 die dem Centroid nächste.
  - Der Tangenten-Fallback legt M auf den Centroid. PI und SS sind dadurch systematisch verschoben: Die Median-Differenz zur gespeicherten PI beträgt +5° gegenüber −2° bei Cobb-Linien.
- **Lageabhängigkeit:** ~~Nur die PI stammt aus jeder Seitaufnahme.~~ Revidiert am 29.09.: *alle* sagittalen Parameter, auch die PI, kommen nur aus seitlichen Röntgen-Stehaufnahmen (siehe Phase 2).
- **QC Hüfte:** Liegt PT außerhalb von −30…60°, wird die Hüfte verworfen. Flag `hip_implausible`, Grund „Hüfte unplausibel“. Das betrifft 7 Messungen, typisch ist eine `hipLine` auf S1-Höhe.
- **Hauptgrund** für eine fehlende Abdeckung, nach Priorität: fehlender Bildausschnitt bzw. Aufnahmetyp (keine seitliche Aufnahme, keine Stehaufnahme, C7/T1/T4/L1 fehlt) vor fehlender Hüfte vor fehlender Deckplatte.
- **Erste Level-Faktoren** (Cache 03.08., geometrisch bestimmte Level):

  | Level | seitlich | a.p. |
  |---|---|---|
  | L3 | 1,007 (n = 88) | 0,961 (n = 104) |
  | L2 | 0,974 (n = 27) | < 20 Paare, ungültig |
  | L5 | 0,942 (n = 58) | < 20 Paare, ungültig |

  L1 hat in beiden Ansichten weniger als 20 Paare und ist damit ungültig.

### Schritt 1c / Phase 3 (2026-09-29)

- **centroid_cobb (b):** `make_smoothing_spline` (natürliche Randbedingungen) mit festem λ = 0,1 auf y in WKD.
  - `UnivariateSpline` wurde verworfen, weil es an den Rändern einer 30°-S-Kurve um ~8° überschoss.
  - λ ist ein Platzhalter bis Phase 2.
- **Endwirbel:** Extrema der Neigung. Benachbarte Extrema mit |Δθ| < 5° werden zusammengelegt, damit Rauschen eine Kurve nicht teilt. Bei Neigungsgleichstand zweier Wirbel wird der kraniale genommen.
- **cCobb_max** ist der Abdeckungs- und Übersichtsparameter: größte Kurve nach Methode (a), nur aus a.p.-Stehaufnahmen. Die Längsschnittauswertung mit Endwirbeln aus der Baseline folgt in Phase 5.
- **PI-Modell:** Bild-PI = Rater-Mittel. Varianz je Bild = σ_m²/n + τ².
  - σ_m wird gepoolt über Bilder mit ≥ 2 Ratern geschätzt, τ² per Momentenschätzer.
  - Auffällig ist ein Bild bei Leave-one-out-|z| > 3.
  - Ausgeschlossen wird sequentiell das jeweils schlimmste Bild, solange ≥ 2 Bilder bleiben. Bei 2 widersprüchlichen Bildern wird keins ausgeschlossen, beide gelten als auffällig.
- **Befund (alter Cache 08/2026, vorläufig):** σ_m je nach Teilmenge

  | Teilmenge | σ_m |
  |---|---|
  | alle | 9,0° |
  | nur Cobb-Deckplatte | 7,7° |
  | Stehaufnahmen | 8,0° |
  | Cobb + Stand | 6,5° |

  - SS und PT liegen einzeln bei je ~4°.
  - Die Funktionsaufnahmen erhöhen die PI-Streuung. PI bleibt dort trotzdem zulässig, weil es morphologisch ist. Zu prüfen ist, ob incl/recl ausgeschlossen werden sollen (offene Frage).
  - Die eigene PI ist zeitlich stabiler als das Spline-Surrogat: Median-SD je Patient 6,4° gegenüber 9,3°.

### Phase 2 + Korrekturen (2026-09-29, frische API-Daten)

- **PI nur aus Röntgen-Stehaufnahmen** (Entscheidung des Nutzers):
  - incl/recl liefern keine PI, Grund „Funktionsaufnahme“.
  - MRT ebenfalls nicht: Aus MRT lässt sich die PI nicht ableiten. Grund „keine Stehaufnahme“.
  - Die Lagerung wird vor den Landmarken geprüft, damit der Grund stimmt.
  - Damit weicht die Umsetzung von Spec §3.5 ab („PI aus jedem CT/MRT mit Hüftköpfen“).
- **Abweichung von Spec §4.2 – M ist nicht die Linienmitte:**
  - Befund: Die gezeichneten S1-Linien sind Cobb-Linien, im Median 2,1 WKD lang (P90 4,3), die Deckplatte selbst ist etwa 1 WKD tief. Die Linienmitte lag bis 0,4 WKD neben der Wirbelsäule.
  - Lösung: Die Linie liefert nur die **Richtung**. M ist der Schnittpunkt mit der Centroid-Sehne des kranialen Nachbarn (L5–S1 bzw. T12–L1). Das Segment wird auf 1 Centroid-Abstand um M normiert, P = M − ½ Abstand · e dient als Surrogat für die hinterer Deckplattenecke (SVA).
  - Auch die Auswahl der Linien läuft jetzt über den Schnittpunkt statt über die Linienmitte.
  - Ergebnis: Gegen das Frontend (`*_cobb`, Spline-Schnittpunkt) liegt die Median-Differenz bei 0°; die Rater-SD für PT sank von 4,1° auf 2,6°.
  - Test: `test_long_offcenter_cobb_line_gives_same_result`.
- **Landmarken-Kovarianz:**
  - Nur Stehaufnahmen und Bilder mit ≥ 2 Ratern, Positionen in Patienten-WKD mit **einem WKD pro Bild**. Ein WKD je Rater erzeugte eine Scheinstreuung von ~0,5 WKD.
  - Robust: Abweichungen > 0,5 WKD bzw. > 15° gelten als grobe Fehler (Label- bzw. Deckplattenverwechslung), werden verworfen und als Rate ausgewiesen.
  - Bei weniger als 20 Freiheitsgraden wird die gepoolte Centroid-Kovarianz der Ansicht verwendet.
- **Monte-Carlo:** 100 Ziehungen je Messung, Landmarken unabhängig, Ergebnis als Spalte `sd` in `degen_params.pkl`. Tangenten-Deckplatten haben eigene Kovarianz-Kinds (`_t`). Laufzeit der Pipeline etwa 1,5 min.
- **Befunde Phase 2** (reports/messunsicherheit.md):
  - Die Centroid-Platzierung streut um 0,04–0,06 WKD (≈ 1,5–2 mm), zervikothorakal bis 0,3 WKD.
  - Grobe Fehler bei der Hüfte: 14 % seitlich, 27 % a.p.
  - Monte-Carlo- und empirische SEM stimmen innerhalb von ~1° überein (PI 4,6 gegenüber 5,7°, PT 2,5 gegenüber 2,6°, LL 4,5 gegenüber 4,7°).
  - centroid_cobb gegen manuellen Cobb: Bias ≈ 0°, LoA ±12–14°. Bei Kurven bis S1 Bias +2–3°.
  - Rater-SEM bei festen Endwirbeln: manuell 4,5°, centroid a 4,3°, centroid b 3,1°. **Methode b ist genauer als der manuelle Cobb.**
- **Befund Phase 3** (reports/pi_patientkonstante.md): τ zwischen Aufnahmen (6,5–8,1°) ist größer als σ Rater (5,7°), getrieben durch den Wechsel des Aufnahmetyps (LWS-lat ↔ WS-lat: SD 10,2° ohne Versatz). Die PI dient damit als Schätzer des Aufnahmefehlers für Σ_Δ in Phase 5.

### Phase 4 – Normabweichung (2026-09-29)

- **Referenz:** Standard ist die MEANS-Gesamtnorm (multiethnisch). Passende Alters- und Geschlechtsstrata haben Vorrang: zuerst Yukawa, dann reine Altersstrata (Nutzerentscheidung). Die Rangfolge steht in `norms.population_reference`.
- **Verifizierte Normwerte** (Abstracts über Europe PMC):
  - MEANS spinopelvin: PI 52,0 ± 10,7, PT 12,5 ± 7,3, SS 39,5 ± 8,2, LL L1–S1 57,4 ± 11,3 (im Original negativ kodiert), PI−LL −5,4 ± 10,7
  - MEANS global: T1PA 5,0° (junge Erwachsene) bzw. 13,7° (> 64 J.), SVA −14,2 bzw. 17,0 mm, jeweils ohne SD
  - Yukawa gesamt: PI 53,7 ± 10,9, PT 14,5 ± 8,4, SS 39,4 ± 8,0, SVA 3,1 ± 12,6 mm
- **Yukawa-Strata** nach Dekade und Geschlecht stehen nur im Volltext (Springer, Paywall). In `norms.yaml` ist die Struktur leer angelegt, bis der Nutzer die Tabelle liefert. Werte werden nicht geschätzt.
- **Yukawa-LL** (T12–S1) ist nicht mit unserer LL (L1–S1) vergleichbar und wird weggelassen.
- **„young adults“ bei MEANS** ist im Abstract nicht abgegrenzt; als Altersgrenze sind 18–35 J. angenommen.
- **Hills, GAP, Schwab, Roussouly** sind aus der Spec übernommen und nicht gegen die Originale geprüft (`verified: false`).
- **SVA:** Unsere SVA ist in WKD, die Normen in mm. Es gibt deshalb keine Abweichung oder z-Wert, nur einen Hinweis. Eine Umrechnung in mm bräuchte eine belegte Normtiefe des Wirbelkörpers.
- **PI-abhängige Ziele** verwenden die Patienten-PI aus Phase 3. Die Unsicherheit ist SD_Abweichung = √(SD_Wert² + (a·SE_PI)²). Ein z-Wert entsteht nur, wenn die Residual-SD bekannt ist (Hills-Zielbereich ± 2°).
- **Klassifikationen** werden als „unsicher“ markiert, wenn eine Schwelle innerhalb ±1,96·SD liegt.
- **LDI** wird über die echte L4-Deckplatte berechnet: Die segmentalen Cobb-Linien enthalten sie meist, ein Centroid-Surrogat ist daher nicht nötig. Der LDI entfällt bei LL < 10°, weil der Quotient dann instabil ist. SEM 7,9 Prozentpunkte.
- **Visitenwert** = Mittel aller Messungen der Visite, SD = √(mittlere MC-Varianz / n). Die Messungen werden dabei als unabhängig angenommen; Rater am selben Bild teilen aber den Aufnahmefehler (τ). Das wird in Phase 5 ersetzt.

### Normen aus Volltexten + koronare Parameter (2026-09-29, PDFs vom Nutzer)

- **Yukawa 2018:** Tabellen 1 und 4–8 aus dem Volltext eingetragen (PI, PT, SS, SVA je Geschlecht × Dekade 20–79, mit n). Die LL (T12–S1, Tab. 4) bleibt weg, weil sie nicht vergleichbar ist.
- **Wang 2025** (Spine 50:E431, n = 701, ≥ 50 J., chinesisch, Tab. 2): PI, PT, SS, LL L1–S1 und SVA als 5-Jahres-Altersstrata 50–55 … > 75, ohne Trennung nach Geschlecht und ohne n je Gruppe.
- **Gharbi 2024** (Brain & Spine 4:103917, EOS, N-Gruppe n = 236):
  - koronar: Hüftkopfwinkel, L4-/L5-Tilt, CTPA, C7-CSVL mit Altersstrata < 20 / 20–50 / > 50
  - sagittal: T1PA, GT und PI−LL mit SD, jeweils nach Alter
- **Rangregel erweitert:**
  - Alters- und Geschlechtsstratum (Yukawa) > reines Altersstratum (Wang, Gharbi, MEANS-T1PA) > Gesamtnorm (MEANS).
  - Bei Gleichstand gewinnt ein Eintrag mit SD, dann die Reihenfolge in der Datei.
  - Altersgrenzen sind halboffen.
  - Folge: Ab 80 J. greift Wang (> 75) statt MEANS gesamt. LL L1–S1 hat ab 50 J. Wang als Referenz. T1PA und GT haben Gharbi als Referenz, weil MEANS dort keine SD liefert.
  - ⚠ Das geht über die Nutzerentscheidung „MEANS Standard, Yukawa für Alter/Geschlecht“ hinaus und folgt deren Logik, geschichtete Normen vor der Gesamtnorm. Zur Bestätigung vorgelegt.
- **Koronare Parameter aus a.p.-Stehaufnahmen** (`CORONAL` in der Pipeline):
  - **FHA:** Winkel der `hipLine` (beide Hüftkopfzentren) zur Horizontalen.
  - **L4-/L5-Tilt:** a.p.-Cobb-Linie der kranialen Deckplatte (Auswahl wie sagittal über den Schnitt mit der Centroid-Sehne) zur Horizontalen.
  - **CTPA:** Vertikale gegen die Linie S1-Mitte → T1-Centroid. Die S1-Mitte ist der Schnitt der S1-Deckplattenlinie mit der L5–S1-Sehne; fehlt die Linie, dient der S1-Centroid als Ersatz („≈“).
  - **C7-CSVL:** in WKD, deshalb nicht mit der mm-Norm vergleichbar.
  - Alle als **Beträge** wie bei Gharbi, damit geht die Seite verloren. Die Verteilung ist rechtsschief, z-Werte sind daher nur grob.
  - **Nicht messbar** (Landmarken fehlen): Beckenschiefstand und Maloney-Winkel (Beckenkamm), Coracoid-Winkel und -Distanz, OD-CSVL.
- **MC-SD koronar:** CTPA, C7-CSVL und cCobb_max aus verrauschten Centroiden. FHA und L4-/L5-Tilt haben keine MC-SD, weil Linien nicht verrauscht werden. Im Normtab dient dort die **robuste Rater-SEM** als Unsicherheit.
- **Robuste SEM** = 1,4826·Median(|paarweise Rater-Differenz|)/√2, zusätzlich zur klassischen SEM.
  - Beispiel FHA: klassische SEM 5,2°, robust 0,97°; nur 3 % der Bilder haben eine Rater-Spanne > 5°.
  - Der Unterschied zwischen beiden SEM zeigt den Anteil grober Fehler (Hüftlinie bzw. Label verwechselt) gegenüber dem Platzierungsrauschen.
- **Befunde koronar** (Live-Daten, Median gegen Gharbi-Norm):

  | Parameter | Median (IQR) | Gharbi-Norm | robuste SEM | Patienten |
  |---|---|---|---|---|
  | FHA | 1,4° | 1,6 ± 1,2 | 0,97° | – |
  | L4-Tilt | 7,6° (3,6–11,9) | 3,0 ± 2,3; leichte Skoliose 6,3 ± 5 | 1,6° | – |
  | L5-Tilt | 6,4° | – | 2,5° | – |
  | CTPA | 1,9° | 1,1 ± 1,0 | – | 40 |
  | C7-CSVL | 0,32 WKD | – | – | 36 |

  Für CTPA und C7-CSVL gibt es noch keine Bilder mit ≥ 2 Ratern und gezeichneter S1-Deckplatte.

### Phase 5 – Veränderungsvektoren (2026-09-29)

- **Komponenten:**
  - Skalarvektor: Parameter nach Wahl; Standard SS, PT, LL, L1PA, cCobb_max, L4-Tilt. PI ist ausgenommen und dient als Kontrolle: ΔPI sollte 0 sein.
  - Formvektor: Centroide im Sakrumsystem (WKD), gemeinsame Wirbel beider Visiten, nur Stehaufnahmen.
- **Σ_Δ = D₁RD₁ + D₂RD₂ + 2·diag(τ_gleicherTag² + τ_Tag²)**, auf positiv semidefinit projiziert (`nearest_psd`).
  - D: Visiten-SD, d. h. MC-SD bzw. robuste Rater-SEM geteilt durch √n.
  - R: **Fehlerkorrelation** aus Rater-Abweichungen am selben Bild, paarweise, ≥ 30 Paare. Sie ist stark: PI–SS 0,85, PT–L1PA 0,86, L1.x–L3.x 0,93. Ein diagonales Σ wäre falsch.
  - **τ_gleicherTag** je Komponente aus verschiedenen Aufnahmen desselben Patienten am selben Tag, robust über MAD, mindestens 10 Paare, sonst 0 mit Hinweis.
    - ≈ 0 für PI, SS, L1PA, cCobb_max und FHA
    - PT 2,6°, LL 2,0°, LDI 7,4 Prozentpunkte, L4-Tilt 4,5°, L5-Tilt 1,8°
    - Formkoordinaten bis 0,2 WKD
  - **τ_Tag** (Slider): voreingestellt mit der Tag-zu-Tag-Schwankung der PI, 4,2° aus 294 Visitenpaaren. Weil die PI biologisch konstant ist, misst das Positionierungs- und Projektionsfehler zwischen Visiten. Die echte Haltungsschwankung (Σ_haltung) steckt nur teilweise darin, die Ergebnisse sind daher eher liberal. Für Längen und Form ist der Wert ein Platzhalter mit Voreinstellung 0.
- **Test:** D² ~ χ²(p), α wählbar (Standard 0,05, ohne Korrektur für Mehrfachtests).
  - Abnahme per Simulation erfüllt: Fehlerrate unter H₀ 5 % ± 1 %.
  - Der Omnibus-Test verliert Power, wenn sich nur eine Komponente ändert. Bei 4 Komponenten braucht er für 80 % Power etwa 4 σ_Δ. Deshalb zeigt die App auch die RCI je Komponente.
- **Richtung Norm:**
  - u = (Ziel − x₁) in Einheiten der Populations-SD, Ziel bevorzugt das PI-Ziel (Hills bzw. GAP), sonst das Populationsmittel.
  - Die Projektion hat eine SD √(uᵀΣu); positiv heißt Annäherung.
- **Typologie:**
  - „Deutlich abweichend“: D² der Abweichung zum Ziel bei t₁ (Norm-SD, unkorreliert) mit p < α.
  - „Driftend“: Veränderungs-D² mit p < α.
  - Die Kohorte wird mit einem festen Kernsatz (SS, PT, LL, L1PA) verglichen, erste gegen letzte Visite mit allen vier Parametern.
- **Offen:** Eine Dekompensationsrichtung u_dek aus den Verläufen progredienter Patienten zu lernen, braucht ein Ereignis- oder Outcome-Label (Phase 8).

### Formvektor: Stationen und 3D (2026-09-30, Nutzerwunsch)

- **Stationen:** Alle Visiten mit Formdaten werden gezeigt, nicht nur t₁/t₂.
  - Getestet wird jeder aufeinanderfolgende Schritt plus erste → letzte Station, jeweils über die in *beiden* Stationen vorhandenen Wirbel. So bleiben Zwischenaufnahmen mit größerem Ausschnitt, z. B. eine Ganzaufnahme zwischen zwei LWS-Aufnahmen, erhalten.
  - Die Tests sind nicht für Mehrfachtests korrigiert (wie Phase 5).
- **Koronarer Formvektor (a.p.):**
  - Ursprung ist die S1-Mitte (Schnitt der S1-Deckplatte a.p. mit der L5–S1-Sehne, sonst S1-Centroid).
  - Die x-Achse ist die Beckenquerachse zum Patienten links. Priorität: S1-Deckplatte a.p. > `hipLine` > Bildhorizontale (Spec §4.6: Beckenreferenz verwenden), gespeichert als `frame_ref`.
  - Einheit WKD der a.p.-Kalibrierung, also Wirbel*breite*. Koordinaten `.z` (links) und `.c` (kranial).
  - Die Fehlermodelle (SEM, Korrelation, τ gleicher Tag) werden wie für die seitlichen Koordinaten geschätzt.
- **3D (näherungsweise, Spec §4.7):**
  - Die Punkte werden über das Label zusammengeführt: anterior und kranial aus der seitlichen Aufnahme, links aus der a.p.-Aufnahme desselben Tages. Die kraniale a.p.-Koordinate entfällt, weil sie redundant ist.
  - Für die Darstellung wird a.p. mit dem Median-Verhältnis der Segmentlängen auf Wirbeltiefe skaliert. Der Test nutzt die Originaleinheiten, jede Komponente hat ihre eigene Varianz.
  - Die Fehlerkorrelation zwischen seitlichen und a.p.-Koordinaten wird mit 0 angenommen (verschiedene Bilder).
  - Axiale Rotation ist nicht bestimmbar.
- **Keine OP-Markierung** (Nutzerentscheidung 2026-09-30).
- **Darstellungskonvention seitlich** (Nutzervorgabe 2026-09-30): anterior links, posterior rechts, wie auf den seitlichen Röntgenbildern. Umgesetzt als umgekehrte x-Achse (`autorange='reversed'`) in 2D seitlich und 3D. Intern bleibt anterior = +x, deshalb bleiben Vorzeichen in Tabellen („ant +0,12“ = nach anterior), Tests und Pipeline unverändert.
- **Gesamtvektor** = mittlere Verschiebung der gemeinsamen Wirbel (Schwerpunkt) relativ zum Sakrumsystem, g = A·Δx mit A = Mittelungsmatrix, Cov = A·Σ_Δ·Aᵀ, Test χ²(Achsenzahl).
  - Er fasst die netto Verlagerung zusammen: anterior etwa bei Lordoseverlust, kaudal bei Höhenverlust. Eine reine Formänderung ohne Schwerpunktverlagerung, z. B. eine Umverteilung der Lordose, zeigt er nicht; dafür gibt es den Omnibus-D² und die Pfeile je Wirbel.
  - Er hängt vom gemeinsamen Wirbelsatz ab. Die Schrittvektoren addieren sich deshalb nur bei gleichem Satz zum Gesamtvektor.
  - Er bildet tₙ − t₀ direkt ab (erste und letzte *ausgewählte* Station), nicht als Summe der Schritte. Der Schwerpunkt ist das ungewichtete Mittel der Centroide der in t₀ und tₙ gemeinsamen Wirbel.
- **Regionale Vektoren** (Nutzerentscheidung 2026-09-30), jeweils mit `change.mean_shift`, Cov = A·Σ_Δ·Aᵀ:
  - **Lumbalvektor** = mittlere Verschiebung L1–L5 im Sakrumsystem.
  - **Thorakalvektor** = mittlere Verschiebung der BWS-Wirbel minus Verschiebung von L1. Das ist der Eigenbeitrag der BWS ohne die über die Gliederkette mitgetragene LWS-Veränderung.
  - Grenze: Nur die Translation von L1 wird herausgerechnet; eine Kippung am thorakolumbalen Übergang wirkt weiter über den Hebel. Die Alternative, ein lokales System an der L1-Deckplatte, gibt es seitlich, a.p. aber meist nicht.
  - Je Region sind ≥ 3 gemeinsame Wirbel nötig, thorakal zusätzlich L1. Sonst wird die Region nicht angezeigt.
  - Datenlage: Ein Thorakalvektor zwischen erster und letzter seitlicher Station ist bei 41 Patienten möglich. In 3D braucht er BWS-Wirbel auch in a.p. und ist deshalb selten.
- Alternative, nicht umgesetzt: der Vektor des obersten gemeinsamen Wirbels, analog SVA bzw. C7-CSVL, aber je nach Bildausschnitt ein anderer Wirbel.

### Maßstab (2026-09-30, Nutzerentscheidung: Punkte 1 und 2, keine maßstabsfreie Anzeige)

- **Problem:** Die Normierung jeder Aufnahme auf ihre eigene Kalibrierlinie (WKD) erzeugt zwischen Aufnahmen Maßstabsfehler. Sie wachsen mit dem Abstand zu S1 und wirken wie Höhen- oder Formänderungen.
  - Kohorte: Die Lendenlänge L1–S1 in WKD weicht zwischen Aufnahmen desselben Patienten typischerweise 3–4 % ab, bei 16–18 % der Aufnahmen > 10 %, bei 3–6 % > 20 %.
  - Beispiel 2PVD, WS-a.p. 06/2024: −32 %.
- **1. Maßstabsfehler im Fehlermodell:** σ_s je Ansicht (seitlich 4,2 %, a.p. 5,2 %).
  - Schätzung: paarweise Log-Differenzen der Lendenlänge in WKD zwischen Aufnahmen je Patient, robust (MAD), nach Korrektur der groben Fehler, minus Rater-Anteil (Rater-CV 5,1 % / n̄ Rater).
  - Im Test: Term 2·σ_s²·x·xᵀ je Ansicht. x ist das Mittel der Positionen beider Visiten; seitliche und a.p.-Längen sind getrennt, weil es verschiedene Aufnahmen sind.
  - Mögliche Doppelzählung: τ_gleicherTag der Formkoordinaten enthält Maßstabsunterschiede am selben Tag schon teilweise. Das ist konservativ und bewusst in Kauf genommen.
- **2. Grobe Kalibrierfehler korrigieren:**
  - Weicht das Verhältnis Lendenlänge/WKD mehr als 15 % vom Patientenmedian derselben Ansicht ab und liegen ≥ 3 Aufnahmen vor, wird WKD = Lendenlänge / Median gesetzt (Flag `calib_inconsistent`).
  - Bei genau 2 Aufnahmen ist nicht entscheidbar, welche falsch ist: nur Flag `calib_inconsistent_unresolved`.
  - Ein echter lumbaler Höhenverlust > 15 % würde mitkorrigiert, bleibt über das Flag aber sichtbar.
- **Nicht umgesetzt** (Nutzerentscheidung): maßstabsfreie Formdarstellung.
- **Datenqualität:** 2 Kombinationen Bild × Rater haben Mehrfachmessungen (8 Messungen, z. B. 2PVD Bild 2794 mit 6 Messungen derselben Raterin). Sie verkleinern die Visiten-SD leicht; nicht bereinigt.

### Maßstab – Revision (2026-09-30, nach Nutzerbefund 2PVD)

- **Befund:** An #2790 (06.06.2024) und #2796 (16.09.2024), beide WS-a.p., zeigt das cm-Lineal real T1–S1 = 45 → 48 cm (+3 cm, OP-bedingt).
  - Beide Bilder haben identische 8,07 Stage-px/cm, die rohen Pixel geben +6,7 % also korrekt wieder.
  - Die Kalibrierlinien messen aber 5,5 cm (L3) gegenüber 4,0 cm (L4): 37 % Unterschied bei konstanter Anatomie.
  - Mit der Korrektur über die Lendenlänge blieben < 1 cm übrig, mit der Roh-WKD-Normierung wurden es scheinbar +50 %.
- **Falsche Annahme zurückgenommen:** „Lendenlänge L1–S1 ist über die Zeit konstant“ gilt nicht bei OP-Distraktion oder Kollaps. Kein anatomisches internes Maß taugt als Referenz, wenn es Teil der gemessenen Veränderung sein kann.
- **Umsetzung (Nutzerentscheidung):**
  1. Keine Korrektur mehr, nur QC-Hinweis `calib_inconsistent`, wenn der Maßstab > 15 % vom Patientenmedian abweicht. Betrifft 149 Messungen.
  2. σ_s nur aus verschiedenen Aufnahmen **desselben Tages**, weil dort keine biologische Änderung möglich ist. Ergebnis: **seitlich 8,0 % (57 Paare), a.p. 7,6 % (102 Paare)**, abzüglich Rater-Anteil. Das ist größer als die frühere Schätzung von 4–5 %; die tägliche Referenz enthält auch Wechsel zwischen LWS- und WS-Format.
- **Konsequenz:** Die kleinste sicher erkennbare relative Längenänderung liegt bei etwa 1,96·√2·σ_s ≈ **21–22 %**. Absolute Höhenänderungen wie 3 cm auf 45 cm sind mit der WKD-Normierung **nicht zuverlässig beurteilbar**.
  - Winkel (PI, SS, LL, VPAs, centroid_cobb) sind maßstabsfrei und nicht betroffen.
  - Formvektoren sind in der Krümmung aussagekräftig, im Betrag nur grob.
  - Grob falsche Kalibrierlinien wie bei #2790 bleiben unkorrigiert. Sie sind über das Flag sichtbar, können aber einzelne Visitenpaare als „verändert“ erscheinen lassen.
- **Geprüft und verworfen:** Stage-Pixel als Maßstab innerhalb eines Aufnahmetyps sind kaum stabiler. Die robuste SD der Lendenlänge über Visiten beträgt bei LWS 6–7 % (px) gegenüber 6–7 % (WKD), bei WS 11–12 % gegenüber 13–15 %. Gleiche px/cm wie bei 2PVD sind nicht die Regel.
- **Externe Kalibrierung:** Das cm-Lineal ist nur auf einem Teil der Aufnahmen sichtbar, eine sichere Kalibrierung ist damit nicht erreichbar (Nutzer). Offen als Möglichkeit: maßstabsfreie Verhältnisse innerhalb eines Bildes, etwa die Segmenthöhe relativ zur Summe der lumbalen Segmenthöhen, für lokale Kollaps- oder Umverteilungsmuster.

### Maßstab aus Wirbelkörperhöhen (2026-09-30, Nutzervorschlag)

- **Idee (Nutzer):** Wirbelkörperhöhen sind knöchern. OP-Distraktion, Cages und Degeneration verändern sie nicht, nur die Bandscheibe. Sie taugen daher als Maßstab, wo die Lendenlänge scheitert (2PVD).
- **Quelle:** nur `cobbLines` mit `mode = Cobb-segmental`.
  - Unveränderte Modellvorschläge (`src = model` und identisch mit `suggestedLines`) sind ausgeschlossen, weil die Inferenz noch unzuverlässig ist (Nutzer). Vom Rater verschobene Modell-Linien zählen.
  - Die Länge der Endplattenlinien geht nicht ein, weil sie so ungenau ist wie die calibrationLine (Nutzer).
- **Messung** (`landmarks.vb_heights`): Die Höhe ist der Abstand der Schnittpunkte von Deck- und Grundplattenlinie mit der Wirbelsäulenachse am Centroid.
  - Schräge Linien, Keilwirbel und Überstand der Linie gehen nicht ein.
  - Die Achse ist die Tangente am Centroid (zentrale Differenz), nicht die Spline selbst. Der Krümmungsfehler über eine Wirbelhöhe ist 2. Ordnung.
  - Zuordnung: Schnittpunkt innerhalb ±½ Nachbarabstand. Genau eine Linie je Seite, sonst kein Wert.
- **Plausibel:** Die Höhe beträgt im Median 0,77 WKD seitlich und 0,55 WKD a.p., das entspricht etwa 27/35 mm bzw. 27/48 mm. Höhe/Centroid-Abstand liegt bei 0,82, die Bandscheibe also bei ≈ 18 %.
- **Populationsfaktoren reichen nicht:**
  - Maßstab aus Höhen und Maßstab aus Linie weichen je Messung um 14 % voneinander ab (robuste SD). Im Median ist der Bias 0.
  - Der Grund ist vor allem Anatomie: Das Verhältnis Höhe/Tiefe streut zwischen Patienten um ≈ 12 %.
  - Wechselt zwischen Visiten die Quelle oder das Level, springt der Maßstab. Mit Populationsfaktoren wurde die SEM des SVA schlechter (0,38 → 0,42), σ_s seitlich stieg auf 10,9 %.
- **Umsetzung** (`frames.patient_scale`): Gewichtete Kleinste Quadrate in log je Patient × Ansicht. Die Unbekannten sind der Maßstab s je Messung und die **patienteneigene** Höhe/WKD a[L] je Level.
  - Beobachtungen: Linie (σ 5 %), Wirbelhöhe (σ 4 %), Prior a[L] = Populationsfaktor (σ 12 %).
  - Absolute Einheit bleibt der Mittelwert der Kalibrierlinien des Patienten.
  - Fallback ohne jede Beobachtung: Node-Abstand (`calib_imputed`) wie bisher.
  - Neue Spalten: `calib_source` (vbh/line/imputed), `n_vbh`, `wkd_line_px`; Faktoren in `degen_vbh.pkl`.
- **Ergebnis** (Cache 30.09., alt → neu):
  - Rater-CV des Maßstabs am selben Bild: seitlich 4,7 → 3,4 %, a.p. 4,5 → 4,0 %
  - σ_s aus Aufnahmen desselben Tages: seitlich 8,0 → 5,4 %, a.p. 7,6 → 7,1 %
  - SD Lendenlänge/WKD zwischen Aufnahmen je Patient: 5,9 → 4,3 %
  - `calib_inconsistent`: 149 → 122
  - SEM SVA: 0,38 → 0,31 WKD
  - Abdeckung: 2.111 von 2.758 Messungen (77 %) haben ≥ 1 Wirbelhöhe, aber meist nur 1–2 Wirbel. Mit ≥ 3 Wirbeln sinkt der Rater-CV auf 2,8 %.
- **2PVD (Wahrheit: gleiche px/cm):** Der scheinbare Maßstabssprung sinkt von 42 % auf 17 %, ist aber nicht beseitigt.
  - Die Linie ist grob falsch (37 %), und der Rater hat nur 1–2 Wirbel markiert (L3: 24,0 → 22,0 px).
- **Geprüft und verworfen:** Maßstab je Messung nur aus Höhen, Linie nur zur Verankerung von a[L]. Bei 2PVD besser (−11 %), insgesamt schlechter (Rater-CV seitlich 4,2 %, σ_s 7,7 %).
- **Offen:**
  - Robuste Gewichtung, damit grob falsche Linien wie bei 2PVD nicht mitziehen, etwa Huber.
  - Messanweisung: segmentale Linien für alle Lendenwirbel setzen. Das ist der größte Hebel.

### Annahmen (ohne Rückfrage, revidierbar)

- **Visite** = (patient_id, acquisition_date). a.p. und seitliche Aufnahme am selben Datum bilden eine Visite.
- **Koronare Seite:** Radiologische Konvention. Bild-links entspricht Patient-rechts. Flag `side_assumed`.
- **Datenstand:** Es wird immer frisch über die API geladen. Die gespeicherten `globalParams` gehören je nach Stand zum Batch vor oder nach dem Recompute (`xray-annotator/RECOMPUTE.md`). Sie dienen nur zum Vergleich, eigene Werte werden aus dem `stateObject` neu berechnet.
