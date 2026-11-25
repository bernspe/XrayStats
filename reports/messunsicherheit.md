# Messunsicherheit (Phase 2)

Automatisch erzeugt von `degen_pipeline.py` am 24.11.2026 20:33. Datenbasis: 6024 Messungen, 3257 Bilder, 339 Patienten.

## Methode

- **Empirische SEM:** gepoolte SD zwischen Ratern am selben Bild (nur Deckplatte aus Cobb-Linie; zweite Spalte inkl. Tangenten-Fallback). Enthält Platzierungsrauschen *und* grobe Fehler.
- **Monte-Carlo-SEM:** Landmarken-Kovarianz je Ansicht und Landmarke aus Multi-Rater-Daten (robust: Abweichungen > 0.5 WKD bzw. > 15° als grobe Fehler verworfen), 100 Ziehungen je Messung, Landmarken unabhängig; Median über Messungen. Beschreibt reines Platzierungsrauschen.
- **Robuste SEM:** 1,4826·Median(|paarweise Rater-Differenz|)/√2 – unempfindlich gegen einzelne grobe Fehler; typische Messpräzision, wenn die Landmarken richtig identifiziert sind.
- **MDC95** = 1,96·√2·SEM: kleinste Veränderung zwischen zwei Messungen, die mit 95 % kein Messrauschen ist.
- **Nicht enthalten:** Haltungsvariabilität zwischen Visiten (Σ_haltung) – ohne Test-Retest-Aufnahmen nicht schätzbar. Für Visitenvergleiche ist die MDC95 daher eine **Untergrenze**.

## SEM und MDC95 je Parameter

| Parameter | Bilder ≥ 2 Rater | SEM emp. | SEM emp. inkl. Tangente | SEM robust | SEM MC | MDC95 emp. | MDC95 robust | MDC95 MC |
|---|---|---|---|---|---|---|---|---|
| PI (°) | 235 | 6.56 | 7.28 | 5.30 | 5.75 | 18.19 | 14.68 | 15.94 |
| SS (°) | 509 | 4.59 | 4.94 | 3.71 | 4.40 | 12.71 | 10.28 | 12.21 |
| PT (°) | 235 | 3.23 | 4.35 | 2.14 | 3.62 | 8.97 | 5.94 | 10.02 |
| LL L1–S1 (°) | 294 | 5.21 | 6.24 | 4.53 | 5.12 | 14.44 | 12.56 | 14.18 |
| LDI (%) | 249 | 8.96 | 11.62 | 6.88 | 8.42 | 24.85 | 19.06 | 23.35 |
| L1PA (°) | 235 | 3.17 | 3.97 | 2.10 | 3.39 | 8.78 | 5.83 | 9.39 |
| T4PA (°) | 43 | 2.54 | 3.65 | 2.41 | 3.33 | 7.03 | 6.67 | 9.22 |
| T1PA (°) | 27 | 2.54 | 3.89 | 2.36 | 3.30 | 7.03 | 6.55 | 9.15 |
| T4-L1PA-Mismatch (°) | 43 | 0.98 | 0.99 | 0.81 | 0.78 | 2.71 | 2.24 | 2.16 |
| Global Tilt (°) | 25 | 2.75 | 4.30 | 2.83 | 3.83 | 7.64 | 7.83 | 10.61 |
| SVA (WKD) | 40 | 0.36 | 0.38 | 0.23 | 0.19 | 0.99 | 0.64 | 0.52 |
| centroid_cobb max (°) | 593 | 5.01 | 5.01 | 4.29 | 6.47 | 13.89 | 11.90 | 17.93 |
| Hüftkopfwinkel a.p. (°) | 469 | 2.74 | 2.74 | 0.88 | – | 7.58 | 2.45 | – |
| L4-Tilt a.p. (°) | 450 | 2.32 | 2.32 | 1.68 | – | 6.42 | 4.67 | – |
| L5-Tilt a.p. (°) | 335 | 3.00 | 3.00 | 2.51 | – | 8.31 | 6.96 | – |
| koronarer T1PA (°) | 2 | 0.96 | 0.47 | 1.05 | 0.29 | 2.66 | 2.90 | 0.81 |
| C7-CSVL (WKD) | 1 | 0.35 | 0.12 | 0.52 | 0.04 | 0.96 | 1.43 | 0.12 |

## Landmarken-Streuung (Platzierungsrauschen, robust)

Positionen in WKD (Patientenkoordinaten: seitlich x = anterior, a.p. x = Patient links; y = kranial), Richtungen in Grad. „Grobe Fehler“ = Anteil verworfener Rater-Werte (Label-/Deckplattenverwechslung).

| Ansicht | Landmarke | SD x | SD y | FG | grobe Fehler |
|---|---|---|---|---|---|
| ap | C7 | 0.05 | 0.17 | 76 | 4.0 % |
| ap | H | 0.05 | 0.15 | 365 | 34.1 % |
| ap | L1 | 0.04 | 0.07 | 766 | 1.2 % |
| ap | L3 | 0.05 | 0.09 | 763 | 1.4 % |
| ap | L5 | 0.04 | 0.12 | 755 | 1.0 % |
| ap | S1 | 0.05 | 0.13 | 661 | 0.3 % |
| ap | T1 | 0.05 | 0.16 | 102 | 3.0 % |
| ap | T4 | 0.05 | 0.15 | 111 | 2.8 % |
| lat | C7 | 0.14 | 0.23 | 69 | 5.0 % |
| lat | H | 0.09 | 0.11 | 298 | 15.1 % |
| lat | L1 | 0.05 | 0.08 | 801 | 1.3 % |
| lat | L3 | 0.05 | 0.07 | 798 | 1.7 % |
| lat | L5 | 0.07 | 0.08 | 800 | 0.9 % |
| lat | M_L1 | 0.05 | 0.11 | 397 | 0.0 % |
| lat | M_S1 | 0.12 | 0.12 | 671 | 0.0 % |
| lat | S1 | 0.10 | 0.07 | 767 | 0.1 % |
| lat | T1 | 0.15 | 0.20 | 77 | 2.0 % |
| lat | T4 | 0.11 | 0.13 | 109 | 1.0 % |
| lat | ang_L1 | 2.60 | – | 397 | 0.0 % |
| lat | ang_S1 | 4.42 | – | 667 | 0.3 % |

## centroid_cobb vs. manueller Cobb (a.p., gleiche Endwirbel, gleicher Rater)

| Stratum | Methode | n | Bias (°) | LoA ± (°) |
|---|---|---|---|---|
| alle | a (zentrale Differenzen) | 2305 | 0.61 | 14.23 |
| alle | b (Glättungsspline) | 2304 | 0.71 | 13.51 |
| untere Endwirbel ≠ S1 | a (zentrale Differenzen) | 1695 | 0.09 | 13.85 |
| untere Endwirbel ≠ S1 | b (Glättungsspline) | 1695 | 0.12 | 12.74 |
| untere Endwirbel = S1 | a (zentrale Differenzen) | 610 | 2.04 | 14.90 |
| untere Endwirbel = S1 | b (Glättungsspline) | 609 | 2.36 | 15.02 |

Rater-SEM bei festen Endwirbeln (Gruppen = Bild × Endwirbelpaar, ≥ 2 Rater):

| Methode | SEM (°) | MDC95 (°) | Gruppen |
|---|---|---|---|
| manual | 5.46 | 15.12 | 96 |
| cc_a | 4.88 | 13.53 | 96 |
| cc_b | 3.32 | 9.19 | 96 |
