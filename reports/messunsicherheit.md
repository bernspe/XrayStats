# Messunsicherheit (Phase 2)

Automatisch erzeugt von `degen_pipeline.py` am 02.10.2026 08:09. Datenbasis: 2819 Messungen, 1571 Bilder, 224 Patienten.

## Methode

- **Empirische SEM:** gepoolte SD zwischen Ratern am selben Bild (nur Deckplatte aus Cobb-Linie; zweite Spalte inkl. Tangenten-Fallback). Enthält Platzierungsrauschen *und* grobe Fehler.
- **Monte-Carlo-SEM:** Landmarken-Kovarianz je Ansicht und Landmarke aus Multi-Rater-Daten (robust: Abweichungen > 0.5 WKD bzw. > 15° als grobe Fehler verworfen), 100 Ziehungen je Messung, Landmarken unabhängig; Median über Messungen. Beschreibt reines Platzierungsrauschen.
- **Robuste SEM:** 1,4826·Median(|paarweise Rater-Differenz|)/√2 – unempfindlich gegen einzelne grobe Fehler; typische Messpräzision, wenn die Landmarken richtig identifiziert sind.
- **MDC95** = 1,96·√2·SEM: kleinste Veränderung zwischen zwei Messungen, die mit 95 % kein Messrauschen ist.
- **Nicht enthalten:** Haltungsvariabilität zwischen Visiten (Σ_haltung) – ohne Test-Retest-Aufnahmen nicht schätzbar. Für Visitenvergleiche ist die MDC95 daher eine **Untergrenze**.

## SEM und MDC95 je Parameter

| Parameter | Bilder ≥ 2 Rater | SEM emp. | SEM emp. inkl. Tangente | SEM robust | SEM MC | MDC95 emp. | MDC95 robust | MDC95 MC |
|---|---|---|---|---|---|---|---|---|
| PI (°) | 130 | 5.97 | 6.72 | 4.45 | 4.68 | 16.56 | 12.34 | 12.98 |
| SS (°) | 242 | 4.22 | 4.72 | 3.31 | 3.89 | 11.69 | 9.17 | 10.77 |
| PT (°) | 130 | 2.65 | 3.30 | 1.88 | 2.47 | 7.34 | 5.22 | 6.84 |
| LL L1–S1 (°) | 178 | 4.82 | 5.47 | 4.33 | 4.59 | 13.36 | 11.99 | 12.72 |
| LDI (%) | 141 | 7.94 | 11.41 | 6.88 | 7.84 | 22.00 | 19.06 | 21.74 |
| L1PA (°) | 130 | 2.55 | 3.20 | 1.87 | 2.15 | 7.07 | 5.18 | 5.95 |
| T4PA (°) | 12 | 2.36 | 3.83 | 2.48 | 2.27 | 6.54 | 6.86 | 6.28 |
| T1PA (°) | 7 | 2.63 | 4.85 | 2.43 | 2.20 | 7.30 | 6.73 | 6.10 |
| T4-L1PA-Mismatch (°) | 12 | 1.19 | 1.17 | 1.11 | 0.88 | 3.29 | 3.08 | 2.43 |
| Global Tilt (°) | 5 | 1.83 | 5.13 | 2.88 | 2.61 | 5.08 | 7.99 | 7.22 |
| SVA (WKD) | 9 | 0.32 | 0.45 | 0.23 | 0.20 | 0.90 | 0.64 | 0.55 |
| centroid_cobb max (°) | 260 | 5.01 | 5.01 | 4.40 | 6.25 | 13.87 | 12.20 | 17.33 |
| Hüftkopfwinkel a.p. (°) | 205 | 5.11 | 5.11 | 0.97 | – | 14.17 | 2.69 | – |
| L4-Tilt a.p. (°) | 207 | 1.84 | 1.84 | 1.58 | – | 5.09 | 4.37 | – |
| L5-Tilt a.p. (°) | 175 | 2.88 | 2.88 | 2.43 | – | 7.98 | 6.73 | – |
| koronarer T1PA (°) | 0 | – | 0.41 | – | 0.32 | – | – | 0.87 |
| C7-CSVL (WKD) | 0 | – | 0.14 | – | 0.04 | – | – | 0.10 |

## Landmarken-Streuung (Platzierungsrauschen, robust)

Positionen in WKD (Patientenkoordinaten: seitlich x = anterior, a.p. x = Patient links; y = kranial), Richtungen in Grad. „Grobe Fehler“ = Anteil verworfener Rater-Werte (Label-/Deckplattenverwechslung).

| Ansicht | Landmarke | SD x | SD y | FG | grobe Fehler |
|---|---|---|---|---|---|
| ap | C7 | 0.04 | 0.21 | 26 | 6.1 % |
| ap | H | 0.05 | 0.13 | 204 | 27.3 % |
| ap | L1 | 0.04 | 0.08 | 428 | 1.3 % |
| ap | L3 | 0.04 | 0.08 | 424 | 1.9 % |
| ap | L5 | 0.04 | 0.11 | 415 | 2.0 % |
| ap | S1 | 0.04 | 0.12 | 363 | 0.7 % |
| ap | T1 | 0.05 | 0.19 | 33 | 4.9 % |
| ap | T4 | 0.04 | 0.13 | 36 | 6.2 % |
| lat | C7 | 0.19 | 0.32 | 22 | 8.9 % |
| lat | H | 0.08 | 0.10 | 184 | 13.9 % |
| lat | L1 | 0.05 | 0.07 | 447 | 1.5 % |
| lat | L3 | 0.05 | 0.06 | 447 | 1.4 % |
| lat | L5 | 0.06 | 0.07 | 448 | 0.6 % |
| lat | M_L1 | 0.05 | 0.09 | 251 | 0.0 % |
| lat | M_S1 | 0.07 | 0.07 | 383 | 0.0 % |
| lat | S1 | 0.10 | 0.07 | 432 | 0.1 % |
| lat | T1 | 0.18 | 0.23 | 27 | 5.9 % |
| lat | T4 | 0.14 | 0.12 | 43 | 2.7 % |
| lat | ang_L1 | 2.50 | – | 251 | 0.0 % |
| lat | ang_S1 | 3.89 | – | 379 | 0.6 % |

## centroid_cobb vs. manueller Cobb (a.p., gleiche Endwirbel, gleicher Rater)

| Stratum | Methode | n | Bias (°) | LoA ± (°) |
|---|---|---|---|---|
| alle | a (zentrale Differenzen) | 1144 | 0.70 | 13.79 |
| alle | b (Glättungsspline) | 1143 | 0.59 | 13.20 |
| untere Endwirbel ≠ S1 | a (zentrale Differenzen) | 799 | -0.12 | 13.58 |
| untere Endwirbel ≠ S1 | b (Glättungsspline) | 799 | -0.04 | 12.20 |
| untere Endwirbel = S1 | a (zentrale Differenzen) | 345 | 2.61 | 13.55 |
| untere Endwirbel = S1 | b (Glättungsspline) | 344 | 2.06 | 14.90 |

Rater-SEM bei festen Endwirbeln (Gruppen = Bild × Endwirbelpaar, ≥ 2 Rater):

| Methode | SEM (°) | MDC95 (°) | Gruppen |
|---|---|---|---|
| manual | 4.81 | 13.34 | 70 |
| cc_a | 5.15 | 14.28 | 70 |
| cc_b | 3.04 | 8.43 | 70 |
