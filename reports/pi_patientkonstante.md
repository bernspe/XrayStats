# PI als Patientenkonstante (Phase 3)

Stand 29.09.2026. Datenbasis: API-Abruf vom selben Tag, 219 Patienten, 2.758 Messungen. Tabellen und Grafiken stehen live im App-Tab „PI (Patientenkonstante)“.

## Methode

- **PI je Messung:** Die Hüftachse H ist die Mitte der `hipLine`. Die S1-Deckplatte liefert die Richtung der gezeichneten Cobb-Linie; ihr Mittelpunkt M ist der Schnittpunkt dieser Linie mit der Centroid-Sehne L5–S1 (siehe DECISIONS.md).
- **Nur seitliche Röntgen-Stehaufnahmen** gehen ein: keine Funktionsaufnahmen (incl/recl), kein MRT. Hüften mit PT außerhalb von −30…60° werden verworfen.
- **Modell je Bild:** PI_i = PI_pat + b_i + ē_i mit Var = σ²/n_i + τ².
  - σ ist die gepoolte SD zwischen Ratern am selben Bild.
  - τ ist die zusätzliche Streuung zwischen Aufnahmen und wird per Momentenschätzer bestimmt.
  - Die Patienten-PI ist präzisionsgewichtet.
  - Auffällig ist ein Bild bei Leave-one-out-|z| > 3; auffällige Bilder werden sequentiell ausgeschlossen.

## Ergebnisse

| Größe | nur Cobb-Deckplatte | inkl. Tangenten-Fallback |
|---|---|---|
| Patienten mit PI | – | 159 |
| σ (Rater, gleiches Bild) | 5,7° | 6,4° |
| τ (zwischen Aufnahmen) | 6,5° | 8,1° |
| Median SE der Patienten-PI | – | 7,3° |
| auffällige bzw. ausgeschlossene Bilder | – | 5 / 5 |

**Streuung innerhalb eines Patienten**
- **τ nach Aufnahmetyp:** 3,6° nur zwischen LWS-lat-Aufnahmen, 5,1° nur zwischen WS-lat-Aufnahmen.
- **LWS-lat gegen WS-lat bei denselben Patienten** (n = 31): Die mittlere Differenz beträgt −1,2°, es gibt also keinen systematischen Versatz. Die SD liegt aber bei 10,2°.
- **Rater-σ nach Aufnahmetyp:** 5,5° bei LWS-lat, 6,5° bei WS-lat. Das erklärt die Streuung zwischen den Aufnahmen nicht.
- **Paar-SD der Aufnahmen:** 5,8° für Paare vom selben Tag (n = 37), 8,1° für Paare von verschiedenen Tagen (n = 298). Die robuste SD für verschiedene Tage (MAD) liegt bei 6,5°, die Verteilung hat also schwere Ränder.

**Vergleich mit dem Frontend**
- **Gegen `pelvic_incidence_cobb`:** Bias +0,7°, LoA ±10,9°. Die Konstruktion ist gleich; die Reste stammen aus der Sehne statt dem Spline und aus der ganzzahligen Speicherung.
- **Gegen das Spline-Surrogat `pelvic_incidence`:** Bias −7,0°, LoA ±19,8°. Das Surrogat liegt systematisch zu hoch.
- **Zusammenhang mit der L5/S1-Höhe:** keiner (r = 0,06).

**Zeitliche Instabilität** (SD der Bild-PI je Patient, Median über 86 Patienten)

| Methode | Median-SD |
|---|---|
| eigene PI | 5,1° |
| Spline-Surrogat | 6,3° |

## Bewertung gegen das Abnahmekriterium

Das Abnahmekriterium lautet: Die Streuung innerhalb eines Patienten liegt in der Größenordnung des Messfehlers.

- **Für Aufnahmen desselben Typs erfüllt:** Zwischen LWS-Aufnahmen gilt τ ≈ 3,6° < σ.
- **Über Aufnahmetypen hinweg nicht erfüllt:** Dort kommt eine Aufnahmekomponente hinzu, die größer ist als der Rater-Fehler. Die Ursache ist projektionsbedingt (Beckenrotation, Zentralstrahl, Positionierung) und hat keinen systematischen Versatz.
- **Folge für Phase 5:** Weil die PI morphologisch konstant ist, schätzt τ den **Aufnahmefehler** direkt. Das gilt vermutlich ähnlich für alle beckenbezogenen Parameter (PT, SS, VPAs). Σ_Δ sollte daher neben Σ_mess eine Aufnahmekomponente enthalten, getrennt nach Aufnahmetyp-Paar. Das ist ein belastbarerer Ersatz als der Platzhalter Σ_haltung = 0, bis Test-Retest-Daten vorliegen.

## Offene Punkte

- **Extreme PI-Werte** (z. B. Patient SX4G, PI ≈ 108°): im Visiten-Tab prüfen, ob die Deckplatte verwechselt oder ein Übergangswirbel vorliegt. Übergangswirbel sind bisher nicht kodiert.
- **PI-Modell:** σ je Aufnahmetyp und τ je Aufnahmetyp-Paar statt gepoolter Werte.
