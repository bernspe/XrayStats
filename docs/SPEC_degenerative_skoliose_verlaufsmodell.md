# Spezifikation: Verlaufsmodell degenerative Skoliose auf Basis von Wirbel-Centroiden

**Zweck:** Übergabedokument für Claude Code. Es enthält (A) den fachlichen Hintergrund und die Designentscheidungen aus einer Recherchesitzung vom 28.09.2026 und (B) ein phasenweises Arbeitsprogramm mit Abnahmekriterien.

**Status:** Forschungssoftware. Keine klinische Entscheidungsunterstützung. Eine spätere klinische Nutzung wäre voraussichtlich Medizinprodukte-Software nach MDR und ist nicht Gegenstand dieser Spezifikation.

---

## 0. Arbeitsanweisungen für Claude Code

1. **Zuerst das Repository sichten.** Relevant sind das bestehende Annotationsschema (Nodes/Centroiden, Deckplattenlinien, Hüftmittelpunkte), die Django-Modelle, die Django-Q-Tasks und die API. Das bestehende Backend ist `xray_api` (Django REST Framework, Django-Q). Module sind an die vorhandene Struktur anzupassen, nicht parallel neu zu erfinden.
2. **Vor Phase 1 die offenen Fragen aus Abschnitt 8 klären**, soweit sie sich nicht aus Code oder Daten beantworten lassen. Getroffene Annahmen im Code und in `DECISIONS.md` dokumentieren.
3. **Geometrie-Kern als reine Python-Bibliothek** (numpy/scipy, typisiert, ohne Django-Abhängigkeit). Django-Q-Tasks und API-Endpunkte sind nur dünne Hüllen darum.
4. **Tests zuerst:** Jede Formel wird mit synthetischer Geometrie getestet, deren Ergebnis analytisch bekannt ist (pytest).
5. **Keine DB-Migrationen und keine Änderungen am Annotationsschema ohne Rückfrage.**
6. **Harte Regeln aus früheren Fehlern im Umfeld:**
   - Keine fest verdrahtete Wirbelanzahl.
   - Wirbelidentität immer über das anatomische Label, **nie** über Sortierung nach y-Koordinate.
   - Keine Verzerrung des Seitenverhältnisses beim Resizing.
   - Übergangswirbel (L6, sakralisierter L5) explizit behandeln.
7. **Normwerte und Schwellen nie im Code hart kodieren.** Sie gehören in eine Datendatei (`norms.yaml`) mit Quelle, Population, n und Altersbereich.
8. **Jede abgeleitete Größe mit Unsicherheit ausgeben** (Standardfehler bzw. Intervall), sobald Phase 2 steht.

---

## 1. Ziel des Gesamtvorhabens

Ein Modell für die **degenerative (De-novo-)Skoliose**, das drei Dinge leistet:

1. Es erfasst die **strukturelle Deviation dreidimensional**: sagittal, koronar, axial (soweit messbar) sowie die Balance.
2. Es bezieht die **Vulnerabilität** ein: Knochenqualität/Osteoporose, erhaltene segmentale Beweglichkeit, Muskulatur.
3. Es ermöglicht aus **unvollständigen Röntgenaufnahmen** (LWS-Aufnahmen statt Ganzwirbelsäule, Hüftköpfe oft nicht abgebildet) eine belastbare Inferenz. Gefragt ist nach der Abweichung vom Normalprofil und perspektivisch nach der Dekompensationswahrscheinlichkeit.

**Grundsatzentscheidung:** Solange ein individuelles Normalprofil nicht vorhersagbar ist, wird Veränderung primär **intraindividuell als Vektor** formuliert (der Patient ist seine eigene Referenz). Die Populationsnorm dient ergänzend als vorläufige Zustandsachse.

---

## 2. Datenlage (Ist-Stand laut Nutzer)

- **Pro Aufnahme:**
  - Wirbel-Centroiden („Nodes“) mit anatomischem Label, a.p. und/oder seitlich.
  - **Sagittal zusätzlich die Deckplattenlinien von L1 (kranial) und S1 (kranial).**
  - **Sporadisch Hüftmittelpunkte.** Ob beide Hüftkopfzentren einzeln oder nur deren Mittelpunkt vorliegen, ist offen (siehe Abschnitt 8).
- **Koronare Cobb-Winkel** werden bisher aus den Centroiden rekonstruiert (Spline).
- **Die PI** wurde bisher über Hüfte → S1-Centroid und die Richtung des Centroid-Splines rekonstruiert. Das wird ersetzt (siehe Abschnitt 4.2).
- **Ziel:** Diese Daten sind die Basis der Verlaufsbeobachtung.

---

## 3. Fachlicher Hintergrund (verdichtete Erkenntnisse)

### 3.1 Reporting-Standards für den klinischen Verlauf

- **SRS-Standard-Outcome-Set (COSSCO; de Kleuver et al., Spine Deform 2021):**
  - 25 Outcome-Domänen nach WHO-ICF.
  - PROMs: ODI v2.1a, SRS-22r, EQ-5D-3L, NRS (0–10) Rücken und Bein.
  - Verpflichtend sind Baseline und 1 Jahr (einschließlich 30-Tage-Ereignissen); 2 Jahre werden für die Forschung empfohlen.
  - Gilt für operierte ASD-Patienten. **Für den konservativen Verlauf gibt es kein Äquivalent.**
- **AO Spine ASD Patient Profile (Naresh-Babu 2023):** Baseline-Beschreibung in vier Domänen: allgemeine Gesundheit, wirbelsäulenspezifische Gesundheit, Bildgebung (Röntgen und MRT), Deformitätstyp.
- **DWG-Register (DACH):**
  - Kompatibel mit Spine Tango.
  - Eigener Zusatzbogen „adulte degenerative Deformität“.
  - COMI, zusätzlich ODI und EQ-5D.
  - Pflicht für DWG-zertifizierte Zentren.
- **Komplikationsberichterstattung nach den Martin-Kriterien (Ann Surg 2002), zehn Punkte:**
  1. Erhebungsmethode (prospektiv/retrospektiv)
  2. Beobachtungsdauer
  3. Daten nach Entlassung / ambulant
  4. Definitionen der Komplikationen
  5. Mortalität und Todesursachen
  6. Morbiditätsrate und Gesamtzahl der Komplikationen
  7. Prozedurspezifische Komplikationen
  8. Schweregrad-Graduierung
  9. Liegedauer
  10. Risikofaktoren in der Analyse

### 3.2 Radiologische Messstandards und Messfehler

- **Messmethodik:** SDSG Radiographic Measurement Manual (O'Brien et al., SRS; frei als PDF).
- **SRS-Schwab-Klassifikation:**
  - Kurventyp T, L, D oder N.
  - Modifier PI−LL: 0 (<10°), + (10–20°), ++ (>20°).
  - Modifier PT: 0 (<20°), + (20–30°), ++ (>30°).
  - Modifier SVA: 0 (<4 cm), + (4–9,5 cm), ++ (>9,5 cm).
  - AO Spine (2025) empfiehlt sie bedingt als Baseline-Klassifikation.
- **Messfehler Cobb-Winkel:**
  - Klinisch akzeptiert sind ±5°.
  - Bei einer Interobserver-Variabilität von 6–7° braucht es mehr als 10° Differenz für 95 % Sicherheit.
  - Die größte Fehlerquelle ist die Wahl der Endwirbel.
  - Liegend gemessene Werte sind etwa 5° kleiner als stehend gemessene.
- **Natürlicher Verlauf der De-novo-Kurven:**
  - Im Mittel etwa 2° pro Jahr, nicht linear.
  - Schneller bei über 69-Jährigen, bei Levoskoliose und bei lateraler Listhese über 5 mm.
  - Stärkster Prädiktor bei früher degenerativer Skoliose: die **asymmetrische Keilform der apexnahen Bandscheiben**.
- **Konsequenz:** Messrauschen und jährliches Signal sind gleich groß. Zwei-Punkt-Vergleiche sind unzureichend; nötig sind Steigungsschätzung, Unsicherheitsfortpflanzung und Fehlermodelle.

### 3.3 Komplikationsdefinitionen (für spätere operative Verläufe)

- **Klassifikation:** ISSG-AO Multi-Domain Complication Classification.
  - Trennt medizinisch und chirurgisch.
  - Bewertet neurologische Auswirkung, Zeitpunkt, Intervention und Rückbildung.
  - Validiert.
  - Radiologische und implantatbezogene Komplikationen treten spät auf.
- **PJK (Glattes):**
  - Winkel von der Unterkante des UIV zur Oberkante von UIV+2 von mindestens 10° **und** mindestens 10° größer als präoperativ.
  - Die Definitionen variieren stark: Dieselbe Kohorte hatte je nach Definition 46,2 %, 22,9 % oder 8,7 % PJK.
- **PJF:**
  - Radiologisch nach Lafage: PJA über 28° oder Zunahme über 22°.
  - Klinisch/strukturell nach Hart: Zunahme über 10° plus Fraktur, dorsale Bandruptur oder Implantatausriss.
  - Schweregrad: Hart-ISSG PJKSS mit sechs Komponenten.
- **DJK:** analog zur PJK an LIV/LIV+1; die Definitionen sind uneinheitlich.
- **Schraubenlockerung:**
  - Lysesaum von mindestens 1 mm (Sandén 2004).
  - In einer Studie erreichte das CT gegen die Referenz Ausdrehmoment nur 22 % Sensitivität bei 96 % Spezifität.
- **Stabbruch:** ISSG prospektiv 9 % insgesamt und 22 % nach PSO, im Mittel nach 14,7 Monaten.
- **Osteoporotische Frakturen:** OF-Klassifikation der DGOU (OF1–OF5, κ = 0,63). Genant eher für die Epidemiologie.

### 3.4 Sagittales Zielprofil aus der PI

- **PI allein reicht nicht.** Sie erklärt nur etwa 30 % der Varianz der L1–S1-Lordose (Hills 2022). Die GAP-Formel ideale LL = 0,62·PI + 29 hat R² = 0,39.
- **T4-L1-Hip-Achse (Hills et al., Spine 2022):**
  - Der **Vertebral-Pelvic-Angle (VPA)** ist der Winkel am Hüftachsenmittelpunkt zwischen der Linie zur S1-Deckplattenmitte und der Linie zum **Wirbel-Centroid**.
  - Normal gilt **L1PA = 0,5·PI − 21°** (r² = 0,58; in einer AIS-Kohorte 0,5·PI − 19 ± 7°).
  - PI und L1PA zusammen erklären die LL mit R² = 0,74.
  - T4PA ≈ L1PA (r = 0,9), also ein T4-L1PA-Mismatch von etwa 0°.
  - Das Risiko mechanischen Versagens war minimal bei L1PA = 0,5·PI − (19 ± 2)° und einem Mismatch zwischen −3° und +1°.
  - AO Spine 2025/26: bedingte Empfehlung; weniger validiert; nur für lange Fusionen untersucht.
  - **→ Das Konzept ist centroidbasiert und passt direkt zu den vorhandenen Daten.**
- **GAP (Yilgor 2017):**
  - Ideal SS = 0,59·PI + 9.
  - Ideal LL = 0,62·PI + 29.
  - LDI = L4–S1 / L1–S1 × 100, Ziel 50–80 %.
  - Hinzu kommen RSA über den Global Tilt und ein Altersfaktor.
  - Die prädiktive Validität für mechanische Komplikationen ist uneinheitlich.
- **Roussouly:**
  - Der untere Lordosebogen entspricht dem Sacral Slope; der obere Bogen liegt bei Asymptomatischen bei etwa 20°.
  - Typen über SS: <35° (Typ 1/2), 35–45° (Typ 3), >45° (Typ 4).
  - Der Inflektionspunkt liegt nur bei Typ 3 am thorakolumbalen Übergang. In einer ASD-Kohorte lag er bei 60 % außerhalb von T12/L1.
- **Speziell degenerative Lumbalskoliose (AO Spine 2026, PJK-Prävention):** Hyperlordose vermeiden. Die Lordoseverteilung ist womöglich wichtiger als das exakte Erreichen idealer Zahlenwerte.
- **Altersadjustierung:** Altersadjustierte Konzepte lockern die Ziele mit dem Alter; in der T4-L1-Analyse war das Alter nicht mit dem Versagensrisiko assoziiert. **Ungeklärt.**

### 3.5 Inferenz aus unvollständigen Aufnahmen

Was eine Teilaufnahme hergibt, lässt sich in drei Klassen einteilen:

| Klasse | Beispiele | Konsequenz |
|---|---|---|
| Direkt messbar (LWS a.p./seitlich) | LL (L1–S1), SS, segmentale Winkel, lumbaler Cobb, laterale Listhese, Frakturen | messen |
| Konstant und aus Fremdbildgebung übertragbar | **PI** (lageunabhängig; aus jedem CT/MRT mit Hüftköpfen oder älteren Aufnahmen) | einmal gut messen, dann fixieren |
| Positionell/global, nicht beobachtbar | PT, L1PA, T4PA, SVA, thorakale Kompensation | nur imputieren, mit Unsicherheit |

**Surrogate:**
- SIP (S1 zur Symphyse) korreliert mit R = 0,96 mit der PI, setzt aber die sichtbare Symphyse voraus.
- Der anatomische SS korreliert mit r = 0,73 mit der PI.
- PT ≈ 75 − SFP-Winkel (a.p.; r = 0,87 bei Männern, 0,67 bei Frauen).
- PT über die Beckenschaufelneigung: eher für Kohortenebene geeignet.

**Warnsignal:** Die direkte Vorhersage sagittaler Parameter aus koronaren Aufnahmen hatte begrenzte Genauigkeit und systematische Verzerrungen (Bassani 2026). Typischer Fehlermodus: Regression zum Populationsmittel, sodass gerade die dekompensierten Extremfälle verfehlt werden.

**Pflichten jedes Inferenzmodells:**
1. Eine Vorhersageverteilung ausgeben, keinen Punktwert.
2. Gegen eine Baseline nur aus Alter, Geschlecht und PI antreten.
3. Das Fehlen einer Region als Input kennen.
4. Die Kalibrierung prüfen (Abdeckung der Intervalle), nicht nur den mittleren absoluten Fehler.

### 3.6 Normwerte (Seed-Daten für `norms.yaml`)

| Gruppe | Quelle | Kernwerte |
|---|---|---|
| Sagittal, multiethnisch, EOS | MEANS (n = 468) | TK T4–T12 37,4 ± 10,9°; LL 57,4 ± 11,3° (dort negativ kodiert); T1PA 5,0° bei Jungen, 13,7° bei über 64-Jährigen |
| Sagittal nach Dekade und Geschlecht | Yukawa et al. 2018 (n = 626, Japan) | CL 4,1 ± 11,7°, TK 36,0 ± 10,1°, LL 49,7 ± 11,2°, PI 53,7 ± 10,9°, PT 14,5 ± 8,4°, SS 39,4 ± 8,0°, SVA 3,1 ± 12,6 mm; starke Veränderung zwischen 60er und 70er Jahren |
| Sagittal über 50 Jahre | Community-Kohorte (PMC12502937) | SVA von −20,2 ± 24,3 mm (50–55 J.) bis 8,8 ± 30,8 mm (über 75 J.) |
| Segmentale Neigung | n = 618, nach Alter und PI stratifiziert | siehe Originalarbeit |
| PI-bezogene Form | Hills 2022 | L1PA = 0,5·PI − 21°; T4-L1PA ≈ 0° |
| Koronar | Normstudie 2024 (PMC11736157) | C7-CSVL 8,3 ± 7,5 mm; koronarer T1PA 1,1 ± 1,0°; kein Zusammenhang koronar–sagittal |
| Koronar | OD-CSVL (n = 467) | 8,3 ± 6,5 mm; steigt mit Alter und BMI |
| Prävalenz | Thoraxaufnahmen 25–64 J. / ASD im Alter | 13,4 % thorakale Skoliose bei Asymptomatischen; ASD bei Älteren bis 68 % → **koronare Kurve im Alter ≠ per se Abweichung** |
| 3D/axial | Le Huec & Hasegawa 2016 (n = 268, EOS) | 3D-Normwerte; axiale Rotation ICC nur 0,51–0,88 (obere BWS schwach) |
| Segmentale Beweglichkeit | Staub et al. 2015 (n = 162, Flex/Ext im Sitzen, nicht degeneriert) | Obergrenze der a.p.-Translation 2,1–4,6 mm; der Rotation 16,3–23,5° je Segment |
| Knochen | WHO / Pickhardt 2013 / ISCD 2019 / Jang 2019 | T-Score ≤ −2,5; L1 ≤ 160 HU zu 90 % sensitiv, ≤ 110 HU zu über 90 % spezifisch; ISCD: < 100 HU Osteoporose wahrscheinlich, > 150 HU normal; Normwerte für über 20.000 Erwachsene |
| Muskulatur | Chinesische Erwachsene (PubMed 37456285); Dixon-MRT (PMC11320528) | Pathologische Verfettung gesamt 10–34 % je nach Alter und Geschlecht; Vorhersagegleichungen nach Alter, BMI, Geschlecht |
| Formmodell normal | J Biomech 2024 (n = 85) | Sechs Formmodi erklären 98 % der Varianz der globalen sagittalen Ausrichtung |
| Formmodell ASD | Sci Data 2024 (n = 42, EOS) | Statistisches Formmodell mit FE-Modellen |

**Grenzen der Normdaten:**
- „Asymptomatisch“ ist nicht gleich „ideal“.
- Für über 70-Jährige ist die Datenlage dünn.
- Viele Kohorten sind asiatisch.
- Publiziert sind fast nur Randverteilungen. Für multivariate Abweichungen braucht es Kovarianzen, also Rohdaten oder ein Formmodell.

### 3.7 Vektorformulierung der Veränderung

- **Zwei getrennte Vektoren:**
  - **Formvektor** im **Sakrum-Koordinatensystem**. Er ist intrinsisch und auf Teilaufnahmen messbar.
  - **Haltungs- bzw. Balancevektor** relativ zur Hüftachse bzw. zum Lot (SS, PT, VPAs). Er ist nur messbar, wenn die Hüfte abgebildet ist bzw. die Aufnahme im Stehen erfolgte.
- **Nicht redundante Basis:** PI, PT und SS nicht gemeinsam aufnehmen (PI = PT + SS). Bevorzugt ein Verschiebungsfeld pro Node, später per Formmodell/PCA auf wenige Modi komprimiert.
- **Metrik:** multivariater Reliable Change Index.
  - Δx = x(t₂) − x(t₁)
  - Σ_Δ = Σ_mess(t₁) + Σ_mess(t₂) + 2·Σ_haltung
  - D² = Δxᵀ Σ_Δ⁻¹ Δx ~ χ²(p) unter H₀
  - Je Komponente: RCI_k = Δx_k / √Σ_Δ,kk
- **Σ_haltung (Haltungsvariabilität zwischen Visiten)** ist nicht aus Multi-Rater-Daten schätzbar. Dafür braucht es echte Wiederholungsaufnahmen in kurzem Abstand. Bis dahin als explizit markierten Platzhalter führen.
- **Richtung zerlegen:**
  - Komponente in Richtung Norm bzw. von ihr weg: u_norm = normierter Vektor (x_Ziel − x(t₁)).
  - Komponente entlang einer „Dekompensationsrichtung“ u_dek, später aus den Veränderungsvektoren progredienter Patienten gelernt.
  - Koronar die Vorzeichen relativ zur Konvexität kodieren.
- **Zustand und Veränderung kombiniert (Typologie):**

| | stabil (D² n.s.) | driftend (D² sig.) |
|---|---|---|
| nahe Norm | unauffällig | beginnende Dekompensation |
| deutlich abweichend | kompensiert | dekompensierend |

### 3.8 Statistische Auswertung des Verlaufs (für spätere Phasen)

- **Ereignisse** (Fraktur, Revision, Dekompensation) als Zeit-bis-Ereignis-Analyse.
  - Bei älteren Kollektiven ist der Tod ein **Competing Risk**: Kaplan-Meier überschätzt dann; stattdessen kumulative Inzidenzfunktion bzw. Fine-Gray.
  - Mechanische Komplikationen häufen sich in den ersten zwei Jahren. Mindestens 2 Jahre, besser 5 Jahre nachbeobachten.
- **Joint Models:**
  - Ein Längsschnitt-Teilmodell (gemischtes Modell mit fehlerbereinigter Trajektorie) wird gemeinsam mit einem Hazard-Modell geschätzt.
  - Verknüpfung über aktuellen Wert, Steigung, kumulative Größe oder gemeinsame Zufallseffekte.
  - Vorteile: korrigiert Messfehler und informativen Dropout (z. B. durch Operation) und liefert dynamische individuelle Prognosen.
  - Software: R-Pakete **JMbayes2**, JM, joineRML. In Python gibt es keine ausgereifte Implementierung, daher Export nach R vorsehen.
- **Informative Untersuchungszeitpunkte in Routinedaten:** Geröntgt wird, wenn es schlechter wird. Dagegen hilft die Modellierung des Besuchsprozesses, z. B. durch Inverse-Intensity-Gewichtung.
- **Berichtsrahmen:** STROBE bzw. RECORD (bei Registerdaten), Martin-Kriterien, COSSCO.

---

## 4. Designentscheidungen für die vorhandenen Centroid-Daten

### 4.1 Koordinatensysteme und Vorzeichen (verbindlich)

**Patientenkoordinaten sagittal:**
- x = anterior (+), y = kranial (+).
- Einheit mm, sofern Pixel Spacing bzw. Kalibrierung vorhanden ist, sonst px. Winkel sind davon unabhängig.
- Bildkoordinaten haben typischerweise y nach unten, deshalb gilt y_pat = −y_img.
- Die Blickrichtung des Patienten auf Seitaufnahmen wird normalisiert, sodass anterior immer +x ist.
  - Liegen die Deckplattenlinien geordnet vor (anterior/posterior), wird diese Ordnung verwendet.
  - Sonst gilt als Heuristik für **Stehaufnahmen**: Das tiefer liegende Ende der S1-Deckplatte ist anterior (SS > 0). Dann wird ein QC-Flag gesetzt.

**Patientenkoordinaten koronar:**
- x = Patient links (+), y = kranial (+).
- Die Seitenzuordnung über DICOM Patient Orientation bzw. Metadaten klären (siehe Abschnitt 8).

**Winkel** in Grad nach außen, intern in Radiant.

### 4.2 PI, SS, PT, LL: jetzt über die echten Deckplatten

Die L1- und S1-Deckplattenlinien liegen sagittal vor. **Das Spline-PI-Surrogat wird damit abgelöst.** Es wird nur noch einmalig als Validierungsvergleich berechnet (siehe Phase 3).

**Notation:**
- S1-Deckplatte mit anteriorem Punkt A und posteriorem Punkt P.
- Mitte: M = (A + P)/2.
- Richtung: e = (A − P)/|A − P|.
- Kaudale Normale: n = (e_y, −e_x). Sie zeigt nach unten-posterior, also in das Sakrum.
- H = Hüftachsenmittelpunkt (Mittelpunkt beider Hüftkopfzentren).

**Formeln:**
- **SS** = atan2(−e_y, e_x). Positiv, wenn das anteriore Ende tiefer liegt.
- **PT** = atan2(−(M−H)_x, (M−H)_y). Positiv, wenn M posterior von H liegt.
- **PI (direkt)** = arccos( n · (H−M) / |H−M| ).
- **Pflichttest:** PI_direkt = PT + SS (Toleranz < 0,01° bei synthetischer Geometrie).
- **L1-Neigung** τ_L1 = atan2(−e_y, e_x) mit e aus der L1-Deckplatte (posterior → anterior).
- **LL (L1–S1)** = SS − τ_L1. Lordose positiv. Bei der Übernahme von Normwerten beachten, dass MEANS die LL negativ kodiert.

### 4.3 PI als Patientenkonstante

- Die PI ist morphologisch konstant. Sie wird deshalb **nicht pro Visite neu verwendet**, sondern **einmal pro Patient geschätzt**: als präzisionsgewichtetes Mittel über alle Aufnahmen mit Hüfte und S1-Deckplatte, mit Standardfehler.
- **Konsistenzprüfung:** Einzelmessungen, die mehr als 3 Mess-SD vom Patientenwert abweichen, werden geflaggt. Das deutet auf Annotationsfehler, Beckenrotation, Übergangswirbel oder eine falsche Seitenzuordnung hin.
- **Ausnahmen, bei denen die PI sich ändern kann:** Fusion über das Iliosakralgelenk oder das Becken, sakrale Frakturen.

### 4.4 VPAs aus Centroiden (wenn die Hüfte vorliegt)

- Für jeden Node i mit Centroid C_i:
  - u = M − H, v = C_i − H
  - **VPA_i = −atan2(u_x v_y − u_y v_x, u·v)**
  - Positiv heißt: Der Centroid liegt anterior der Linie H → M.
- Abgeleitet: **L1PA**, **T4PA** (falls T4 annotiert), **Mismatch = T4PA − L1PA**.
- Die Ziele (siehe Abschnitt 3.4) stehen in `norms.yaml`, nicht im Code.
- **Pflichttest:** synthetische Wirbelsäule, bei der der Centroid auf der Linie H → M liegt → VPA = 0. Ein anteriorer Versatz ergibt einen positiven Wert.

### 4.5 Koronare „Cobb“-Winkel aus Centroiden

- **Benennung:** Centroidbasierte Winkel heißen im Code und in der Ausgabe `centroid_cobb`, nie nur `cobb`. Sie messen die Lage der Wirbelkörper, nicht die Neigung der Deckplatten, ähnlich der Ferguson-Methode, und liefern tendenziell kleinere Werte.
- **Zwei Methoden implementieren:**
  - (a) Diskrete Tangenten über zentrale Differenzen an festen Endwirbeln.
  - (b) Glättungsspline mit **festem Glättungsparameter und festen Randbedingungen** für alle Visiten.
- **Längsschnitt:** Die **Endwirbel werden aus der Baseline fixiert.** Eine freie Endwirbelsuche pro Visite ist nur eine Sekundärauswertung.
- **Bekannte Schwächen, die dokumentiert und im QC berücksichtigt werden:**
  - Translation und Kippung sind nicht trennbar; laterale Listhese erscheint als Krümmung.
  - Die Keilform von Bandscheibe und Wirbel wird nur indirekt erfasst.
  - Kurze Kurven (fraktionell L4–S1) werden von den Randbedingungen dominiert.
- **Fehlerfortpflanzung:** Die Tangentenrichtung über eine Sehne der Länge L hat einen Fehler von etwa √2·σ_Landmarke / L. Kurze Sehnen verrauschen, lange Sehnen glätten lokale Effekte weg.
- **Kalibrierung:** Bland-Altman `centroid_cobb` gegen den manuellen Cobb an einer Teilmenge (Voraussetzung dafür siehe Abschnitt 8).
- **Optionale Schema-Erweiterung (nur nach Rückfrage):** Deckplattenpunkte an End- und Apexwirbeln, um Kippung und Translation zu trennen; Pedikel-Landmarken a.p. für die axiale Rotation.

### 4.6 Form- und Haltungsdeskriptoren

**Sakrum-Koordinatensystem (sagittal):**
- Ursprung M (S1-Deckplattenmitte).
- x-Achse entlang e (anterior).
- y-Achse als kraniale Normale.
- Da die S1-Deckplatte vorliegt, ist es **exakt definiert**; eine Procrustes-Registrierung ist dafür nicht nötig.

**Formvektor (sagittal):**
- Node-Koordinaten im Sakrumsystem.
- Sehnenwinkel relativ zur S1-Deckplatte.
- LL (L1–S1).
- Abstände benachbarter Nodes als Segmenthöhen; deren Abnahme zeigt Bandscheiben- bzw. Wirbelkollaps an und ist ein Vulnerabilitätsmarker. **Nur mit Maßstabskalibrierung** verwendbar.

**Haltungsvektor:**
- SS (nur bei Stehaufnahmen mit echter Bildvertikalen).
- PT, VPAs (nur mit Hüfte).

**Koronar:**
- Ohne S1-Referenzlinie in der a.p.-Aufnahme: Procrustes auf die lumbosakrale Kette (rigide, bei kalibriertem Maßstab ohne Skalierung).
- Liegt eine Beckenreferenz vor, diese verwenden (siehe Abschnitt 8).

**Liegend-Aufnahmen:** Nie in Haltungsdeskriptoren. Die Differenz liegend/stehend wird separat als Flexibilitätsmaß geführt.

**Flags pro Aufnahme:**
- Fraktur (Centroid verschiebt sich bei Kollaps)
- Fusion bzw. Implantat
- Übergangswirbel
- Liegend
- Keine Kalibrierung

### 4.7 3D

- Konventionelle a.p.- und Seitaufnahmen sind nicht simultan. Eine 3D-Fusion der Centroiden ist deshalb nur **näherungsweise** möglich (Zuordnung über das Label, gemeinsame kraniokaudale Achse). Sie wird als solche gekennzeichnet.
- Axiale Rotation ist aus Centroiden **nicht** bestimmbar.

---

## 5. Zustandsraummodell (Kern der Verlaufsbeobachtung)

Jede Visite beobachtet eine andere Teilmenge: anderer Bildausschnitt, Hüfte nur sporadisch. Ansatz:

```
x_t = x_{t-1} + b·Δt + w_t,     w_t ~ N(0, Q·Δt)      # latente vollständige Konfiguration (Form + Haltung)
y_t = H_t·x_t + v_t,            v_t ~ N(0, R_t)       # H_t wählt die sichtbaren Komponenten aus
```

**Die Komponenten:**
- **x_t:** Node-Koordinaten im Sakrumsystem plus Haltungsparameter (SS, PT).
- **PI:** feste Patientenkonstante (siehe Abschnitt 4.3), kein Zustand.
- **R_t:** Landmarkenunsicherheit, aus Multi-Rater-Daten oder einem probabilistischen Ground Truth. Fehlen sie, gilt ein dokumentierter Platzhalter.
- **Q:** biologische Variabilität einschließlich der Haltungsschwankung.
- **b:** patientenindividuelle Drift, also der Veränderungsvektor.

**Umsetzung:**
- Kalman-Filter plus RTS-Glätter mit unregelmäßigem Δt und fehlenden Komponenten, als eigene kleine Implementierung in numpy.
- Alternativ `statsmodels.tsa.statespace`, sofern variable H_t sauber abbildbar sind.
- Q und b per EM bzw. Maximum Likelihood mit schwachen Priors.
- Später hierarchisch über Patienten: b_i ~ N(μ_b, Σ_b).

**Ausgaben:**
- Geglättete vollständige Konfiguration je Zeitpunkt mit Kovarianz, auch für nicht abgebildete Teile.
- D²-Test auf reliable Veränderung zwischen beliebigen Zeitpunkten.
- Drift b mit Konfidenzbereich.

**Identifizierbarkeit:** Um Q von R zu trennen, braucht es Test-Retest-Aufnahmen. Solange sie fehlen, ist Q nur zusammen mit dem Platzhalter für R schätzbar. Das muss im Report stehen.

---

## 6. Arbeitsprogramm

Jede Phase endet mit grünen Tests, einem kurzen Eintrag in `CHANGELOG.md` und aktualisierten `DECISIONS.md`.

### Phase 0: Bestandsaufnahme und Datenmodell

**Aufgaben:**
- Repository sichten, das Annotationsschema auf die Abschnitte 2 und 4 abbilden, die offenen Fragen aus Abschnitt 8 beantworten oder stellen.
- Einen Vorschlag für fehlende Metadatenfelder erstellen, **ohne ihn umzusetzen**:
  - Aufnahmetyp (LWS, Ganzwirbelsäule, Becken)
  - Stehend/liegend
  - Pixel Spacing bzw. Kalibrierung
  - Hüfte vorhanden (ein- oder beidseitig)
  - Rater-ID
  - Flags für Fraktur, Fusion und Übergangswirbel
  - Aufnahmedatum

**Abnahme:** Mapping-Dokument `docs/datenmodell.md` und eine Liste der Schemaänderungen zur Freigabe.

### Phase 1: Geometrie-Kern

**Aufgaben:**
- Paket (Arbeitsname `spine_geom`) mit:
  - `frames.py`: Normalisierung der Bild- in Patientenkoordinaten, Sakrumsystem
  - `sagittal.py`: SS, PT, PI, τ_L1, LL, VPAs, Mismatch
  - `coronal.py`: `centroid_cobb` nach beiden Methoden, Apex, Konvexität
  - `descriptors.py`: Form- und Haltungsvektor, Segmenthöhen
  - `qc.py`: Label-Konsistenz, kraniokaudale Monotonie, robuste Ausreißer-z-Werte, Flags
- Synthetische Testgeometrien (parametrisierte Wirbelsäule mit bekanntem PI, SS, PT, LL und bekannten VPAs sowie koronaren Kurven).

**Abnahme:**
- Alle Formeltests grün, insbesondere PI = PT + SS, das VPA-Vorzeichen und die Invarianz gegen Bildspiegelung nach Normalisierung.
- Keine Django-Abhängigkeit im Paket.

### Phase 2: Messunsicherheit

**Aufgaben:**
- Kovarianz pro Landmarke aus Multi-Rater-Daten schätzen (falls vorhanden).
- Monte-Carlo-Fortpflanzung auf alle abgeleiteten Größen.
- SEM und MDC95 je Parameter ausgeben.
- Bland-Altman `centroid_cobb` gegen den manuellen Cobb.

**Abnahme:** Report `reports/messunsicherheit.md` mit Tabellen je Parameter.

### Phase 3: PI als Patientenkonstante

**Aufgaben:**
- Präzisionsgewichtete PI pro Patient mit Standardfehler.
- Konsistenz-Flags.
- Einmaliger Validierungsvergleich des alten Spline-PI-Surrogats gegen die deckplattenbasierte PI: systematische Abweichung, Abhängigkeit von der Degeneration bei L5/S1, zeitliche Instabilität.

**Abnahme:** Report mit der Verteilung der Streuung innerhalb eines Patienten. Die Erwartung ist eine Streuung in der Größenordnung des Messfehlers; Ausreißer sind erklärt.

### Phase 4: Normabweichung (Zustandsachse)

**Aufgaben:**
- `norms.yaml` mit den Seed-Daten aus Abschnitt 3.6 anlegen, inklusive Quellen.
- Aus der PI abgeleitete Ziele: L1PA-Ziel, ideale SS/LL nach GAP, theoretischer Roussouly-Typ aus dem idealen SS.
- Alters- und geschlechtsbezogene z-Werte, sofern Strata vorliegen.
- Hinweis-Flag, wenn die Norm außerhalb von Alter oder Population des Patienten liegt.
- **LDI** (L4–S1 / L1–S1) braucht die L4-Deckplatte und ist nicht verfügbar. Entweder als Centroid-Surrogat, klar gekennzeichnet, oder eine Schema-Erweiterung vorschlagen.

**Abnahme:** Funktion `deviation_state(patient, visit)` gibt die Abweichungen mit Unsicherheit und Quellenangabe zurück.

### Phase 5: Veränderungsvektoren

**Aufgaben:**
- Δx zwischen zwei Visiten, getrennt nach Form- und Haltungsvektor.
- Σ_Δ nach Abschnitt 3.7, mit Platzhalter für Σ_haltung.
- D², p-Wert, RCI je Komponente.
- Projektion auf u_norm.
- Typologie nach der Tabelle in Abschnitt 3.7.
- Frontend: Overlay mit Verschiebungspfeilen pro Node sowie Trajektorien-Plots, an das bestehende Frontend angepasst.

**Abnahme:**
- Auf synthetischen Daten mit bekannter Drift wird die Fehlerrate unter H₀ eingehalten (Simulation, α = 0,05 ± Toleranz).
- Visualisierung für einen Beispielpatienten.

### Phase 6: Zustandsraummodell

**Aufgaben:**
- Kalman-Filter und RTS-Glätter mit fehlenden Komponenten.
- EM- bzw. ML-Schätzung von Q und b.
- Per Django-Q-Task pro Patient ausführbar.
- Später ein hierarchisches Modell über Patienten.

**Abnahme:**
- Validierung per Weglassen einzelner Beobachtungen: Vorhersage des weggelassenen Nodes bzw. der Hüfte.
- Die Abdeckung der 90-%-Intervalle liegt zwischen 85 und 95 %.
- Test mit simulierten Patienten (bekannte Drift, zufällige Bildausschnitte, sporadische Hüfte).

### Phase 7: Maskierungsstudie (Inferenz aus Teilaufnahmen)

**Voraussetzung:** Ein Satz vollständiger Referenzaufnahmen (Stehaufnahme der Ganzwirbelsäule mit Hüftköpfen, ideal EOS) und einige echte Bildpaare (LWS- und Ganzaufnahme innerhalb weniger Wochen) gegen Domain Shift.

**Aufgaben:**
- Systematisch maskieren: nur LWS, mit oder ohne Hüfte, mit oder ohne PI aus Fremdbildgebung.
- Imputationsmodell für L1PA, PT und T4PA mit Vorhersageverteilung.
- Vergleich gegen die Baseline nur aus Alter, Geschlecht und PI.
- Kalibrierung prüfen.

**Abnahme:** Report „Mindestanforderung an die Bildgebung“: Restunsicherheit je Bildausschnitt. Das ist als eigenständige Publikation denkbar.

### Phase 8: Klinische Verknüpfung und Ereignismodell (nur Spezifikation, keine Umsetzung)

**Aufgaben:**
- Datenfelder nach COSSCO und Komplikationsfelder nach ISSG-AO mit den Definitionen aus Abschnitt 3.3 spezifizieren.
- Vulnerabilitätsfelder: DXA (Hüfte bevorzugt), L1-HU, VBQ, prävalente Frakturen (OF/Genant), Funktionsaufnahmen, Differenz liegend/stehend, Ankylose bzw. DISH, Muskelverfettung, Hüft- und Kniestatus, Frailty.
- „Dekompensation“ operational definieren, z. B. als Komposit: globale Achse jenseits einer Schwelle bei ausgeschöpfter PT-Reserve, oder neue Fraktur, oder klinische Verschlechterung über der MCID mit bildgebendem Korrelat.
- Exportformat für R (JMbayes2 / Fine-Gray) festlegen.

**Abnahme:** `docs/klinisches_datenmodell.md` und ein Entwurf des Analyseplans.

---

## 7. Datenquellen für spätere Trainings- und Validierungsschritte

| Datenblock | Zweck | Quelle |
|---|---|---|
| Vollständige Stehaufnahmen der Ganzwirbelsäule mit Hüfte (ideal EOS mit 3D) | Ground Truth für Imputation und Maskierung | eigenes PACS, EOS-Zentren, ESSG/ISSG |
| Echte Bildpaare LWS + Ganzaufnahme | Domain Shift (Zentrierung, Kollimation, Parallaxe) | retrospektiv |
| Normativdaten | Zustandsachse | publizierte Tabellen (Abschnitt 3.6); Rohdaten z. B. beim MEANS-Konsortium anfragen |
| CT mit daraus berechneten Röntgenprojektionen (DRR) | 3D/Rotation aus 2D lernen (nicht für die Haltung, da liegend) | VerSe, CTSpine1K |
| Vortraining der Landmarkenerkennung | Detektion der Nodes | BUU-LSPINE (3.600 Patienten a.p. + seitlich; öffentlich bisher 400 bzw. 2.000 Aufnahmepaare) |
| Längsschnitt mit Ereignissen | Zustandsraummodell, Joint Model | Registerkooperation (DWG), Populationskohorten (z. B. Wakayama/ROAD: MRT + DXA, überwiegend liegend, prüfen), PACS |
| Test-Retest-Aufnahmen in kurzem Abstand | Σ_haltung bzw. Q identifizieren | prospektiv, kleine Stichprobe |

---

## 8. Offene Fragen (vor bzw. in Phase 0 klären)

1. **Hüftmittelpunkte:** Liegen beide Hüftkopfzentren einzeln vor oder nur deren Mittelpunkt? In welcher Ansicht? Wie wird mit nur einem sichtbaren Hüftkopf umgegangen?
2. **Deckplattenlinien:** Sind die Endpunkte geordnet (anterior/posterior)? In welchem Koordinatensystem (px, Ursprung, y-Richtung)?
3. **Seitaufnahmen:** Ist die Blickrichtung des Patienten dokumentiert? Gibt es a.p./p.a. und die Seitenzuordnung (DICOM Patient Orientation)?
4. **Annotierte Höhen:** Welche Wirbelhöhen sind typischerweise annotiert (T10–S1? L1–S1)? Wie werden Übergangswirbel kodiert?
5. **Multi-Rater:** Gibt es Mehrfachannotationen bzw. einen probabilistischen Ground Truth? Wie sind sie gespeichert?
6. **Metadaten:** Sind Pixel Spacing bzw. Kalibrierung und stehend/liegend pro Aufnahme verfügbar?
7. **Koronare Referenz:** Gibt es a.p. eine Beckenreferenz (S1, Beckenkämme, Hüftköpfe)?
8. **Kalibrierung:** Gibt es manuelle Cobb-Messungen für eine Kalibrierungsstichprobe?
9. **Entscheidungsschwelle:** Welches α soll für die reliable Veränderung gelten, und ist eine Korrektur für multiples Testen über Visitenpaare nötig?

---

## 9. Referenzen (Auswahl, mit Links)

**Reporting und Klassifikationen**
- de Kleuver M et al. SRS ASD Standard Outcome Set (COSSCO). Spine Deform 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8363534
- Naresh-Babu J et al. AO Spine ASD Patient Profile. Global Spine J 2023. https://pubmed.ncbi.nlm.nih.gov/34402318/
- DWG-Register, Teilnahmebedingungen. https://www.akademie-der-dwg.de/fileadmin/scripts/dwg/Mitgliederbereich/Teilnahmebedingungen_24.07.2021_Version_1.4_final.pdf
- SDSG Radiographic Measurement Manual. https://www.srs.org/Files/Research/Manuals-and-Publications/sdsg-radiographic-measuremnt-manual.pdf
- ISSG-AO Complication Classification (Validierung). https://www.ovid.com/jnls/spinejournal/fulltext/10.1097/brs.0000000000003766~development-and-validation-of-a-multidomain-surgical
- Martin RC et al. Quality of complication reporting. Ann Surg 2002. https://pubmed.ncbi.nlm.nih.gov/12035036/
- Varianz der PJK-Definitionen. https://pmc.ncbi.nlm.nih.gov/articles/PMC12347835/
- OF-Klassifikation (DGOU). https://journals.sagepub.com/doi/full/10.1177/2192568217717972

**Sagittale Zielwerte**
- Hills J et al. The T4-L1-Hip Axis. Spine 2022. https://pubmed.ncbi.nlm.nih.gov/35867583/
- T4-L1-Hip Axis: Realignment Targets. https://pure.johnshopkins.edu/en/publications/the-t4-l1-hip-axis-sagittal-spinal-realignment-targets-in-long-co/
- AO Spine Clinical Practice Recommendations: Sagittal Alignment (2025/26). https://pmc.ncbi.nlm.nih.gov/articles/PMC13216116/
- AO Spine Recommendations: PJK/PJF Prevention (2026). https://journals.sagepub.com/doi/full/10.1177/21925682261482063
- RLL und LDI (GAP-Komponenten). https://thejns.org/focus/view/journals/neurosurg-focus/43/6/article-pE5.xml
- Roussouly: Spinal Lordosis Ratio. https://journals.sagepub.com/doi/10.1177/21925682221133748

**Messfehler und Verlauf**
- Disc Wedging als Progressionsprädiktor. https://thejns.org/spine/view/journals/j-neurosurg-spine/34/4/article-p557.xml
- Progression kleiner De-novo-Kurven. https://pubmed.ncbi.nlm.nih.gov/19809605/
- Cobb-Winkel liegend vs. stehend. https://journals.lww.com/md-journal/fulltext/2016/02090/adult_degenerative_scoliosis__can_cobb_angle_on_a.57.aspx
- Competing Risks in Registerdaten. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3214742/

**Surrogate und Inferenz**
- Sagittal aus koronar: begrenzte Genauigkeit (Bassani 2026). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13429776/
- Sacral incidence to pubis. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7903762/
- Sacro-femoral-pubic angle. https://link.springer.com/article/10.1007/s00586-011-2061-6

**Normwerte**
- MEANS. https://pubmed.ncbi.nlm.nih.gov/35797648/
- Yukawa et al. (626 japanische Freiwillige). https://link.springer.com/article/10.1007/s00586-016-4807-7
- Asymptomatische über 50-Jährige. https://pmc.ncbi.nlm.nih.gov/articles/PMC12502937/
- Segmentale Neigung nach Alter und PI. https://pubmed.ncbi.nlm.nih.gov/39451141/
- Koronare Normwerte. https://pmc.ncbi.nlm.nih.gov/articles/PMC11736157/
- OD-CSVL-Normwerte. https://link.springer.com/article/10.1007/s00586-021-06873-6
- Segmentale Beweglichkeit (Staub 2015). https://thejns.org/spine/view/journals/j-neurosurg-spine/23/6/article-p731.xml
- Opportunistisches CT (Pickhardt 2013). https://pubmed.ncbi.nlm.nih.gov/23588747/
- L1-HU multizentrisch, ISCD. https://link.springer.com/article/10.1007/s00198-026-08212-1
- Muskelverfettung, Normschwellen. https://pubmed.ncbi.nlm.nih.gov/37456285/
- Formmodell normale sagittale Ausrichtung. https://www.sciencedirect.com/science/article/pii/S0021929024002057
- Formmodell ASD (EOS). https://www.nature.com/articles/s41597-024-03351-8

**Datensätze**
- BUU-LSPINE. https://www.mdpi.com/2076-3417/13/15/8646
- Wakayama Spine Study. https://www.nature.com/articles/s41598-024-64706-0
