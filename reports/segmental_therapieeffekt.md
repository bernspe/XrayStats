# Segmentaler Therapieverlauf – Machbarkeitsnachweis (vorläufig)

Erzeugt mit `python segmental_analysis.py` aus den Caches der Pipeline (`s4-2026-11-25`).
Konzept: `docs/degen_skoliose_verlauf_segmental_bottom_up.md`. **Alle Zahlen sind vorläufig und dienen dem Nachweis,
dass jede Frage aus den Daten beantwortbar ist** – nicht der Antwort selbst.

- Kohorte: validierte Patienten; auswertbare Episoden je Gruppe: {'Dekompression': 2, 'Fusion': 36, 'konservativ': 8}.
- Zeitpunkt: `late` (letzte Visite vor dem nächsten Ereignis). Δ = post − prä, K = |prä| − |post| (> 0 = Korrektur,
  beim Slip Reposition). Einheiten: a.p./seitlich in °, Slip in % WKD.
- Median, IQR (Q25–Q75), CI = 95-%-Cluster-Bootstrap über Patienten, **erst ab 5 Patienten**, darunter nur
  Einzelwerte. `> MDC95 %` = Anteil der Werte über dem Messfehler (Rater-Streuung; Slip konservativ 15,6 % WKD).
- n = Anzahl Werte (Episode × Segment), Patienten = Anzahl Patienten.

## F1 – Welche Veränderungen erzielte welches Verfahren, je Segmenthöhe?

### Δ je Gruppe × Segment
| plane   | group         | seg     |   n |   Patienten |   Median |   Q25 |   Q75 |   CI_lo |   CI_hi |   > MDC95 % |
|:--------|:--------------|:--------|----:|------------:|---------:|------:|------:|--------:|--------:|------------:|
| ap      | Dekompression | T11/T12 |   1 |           1 |      1.0 |   1.0 |   1.0 |   nan   |   nan   |         0.0 |
| ap      | Dekompression | T12/L1  |   1 |           1 |      2.6 |   2.6 |   2.6 |   nan   |   nan   |         0.0 |
| ap      | Dekompression | L1/L2   |   2 |           2 |     -1.5 |  -3.4 |   0.5 |   nan   |   nan   |         0.0 |
| ap      | Dekompression | L2/L3   |   2 |           2 |      0.8 |   0.5 |   1.1 |   nan   |   nan   |         0.0 |
| ap      | Dekompression | L3/L4   |   2 |           2 |      0.7 |   0.2 |   1.2 |   nan   |   nan   |         0.0 |
| ap      | Dekompression | L4/L5   |   2 |           2 |      0.4 |  -0.6 |   1.4 |   nan   |   nan   |         0.0 |
| ap      | Dekompression | L5/S1   |   1 |           1 |     -2.5 |  -2.5 |  -2.5 |   nan   |   nan   |         0.0 |
| ap      | Fusion        | T1/T2   |   3 |           3 |      0.5 |  -2.9 |   1.0 |   nan   |   nan   |         0.0 |
| ap      | Fusion        | T2/T3   |   6 |           5 |     -0.3 |  -5.1 |   3.8 |    -8.3 |     4.9 |        16.7 |
| ap      | Fusion        | T3/T4   |   6 |           5 |     -0.2 |  -1.0 |   7.3 |    -1.2 |    11.3 |        33.3 |
| ap      | Fusion        | T4/T5   |   6 |           5 |      1.6 |  -0.8 |   2.9 |    -1.5 |     6.9 |        16.7 |
| ap      | Fusion        | T5/T6   |   6 |           5 |     -0.2 |  -2.2 |   2.8 |    -2.3 |     3.3 |         0.0 |
| ap      | Fusion        | T6/T7   |   6 |           5 |      2.9 |  -0.2 |   5.6 |    -1.6 |     5.8 |        16.7 |
| ap      | Fusion        | T7/T8   |   6 |           5 |      1.5 |   0.4 |   3.6 |    -0.9 |     4.1 |         0.0 |
| ap      | Fusion        | T8/T9   |   7 |           6 |      1.3 |  -0.5 |   1.6 |    -2.9 |     1.8 |         0.0 |
| ap      | Fusion        | T9/T10  |  10 |           7 |      1.6 |  -1.3 |   2.5 |    -1.6 |     2.6 |         0.0 |
| ap      | Fusion        | T10/T11 |  15 |          10 |      0.6 |  -0.9 |   2.9 |    -0.0 |     1.6 |        13.3 |
| ap      | Fusion        | T11/T12 |  28 |          21 |     -0.1 |  -1.8 |   1.7 |    -1.3 |     1.2 |        10.7 |
| ap      | Fusion        | T12/L1  |  33 |          25 |     -0.5 |  -2.3 |   1.3 |    -2.0 |     0.6 |        12.1 |
| ap      | Fusion        | L1/L2   |  36 |          26 |      0.1 |  -2.4 |   2.1 |    -1.3 |     1.3 |        19.4 |
| ap      | Fusion        | L2/L3   |  36 |          26 |      0.1 |  -1.6 |   2.4 |    -1.0 |     1.8 |        27.8 |
| ap      | Fusion        | L3/L4   |  36 |          26 |     -0.2 |  -2.4 |   4.5 |    -1.5 |     2.5 |        25.0 |
| ap      | Fusion        | L4/L5   |  36 |          26 |      0.2 |  -2.0 |   1.5 |    -0.5 |     0.6 |        16.7 |
| ap      | Fusion        | L5/S1   |  24 |          16 |      0.4 |  -1.5 |   3.2 |    -0.3 |     2.8 |         4.2 |
| ap      | konservativ   | C7/T1   |   1 |           1 |      0.6 |   0.6 |   0.6 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T1/T2   |   1 |           1 |     -2.0 |  -2.0 |  -2.0 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T2/T3   |   1 |           1 |      0.4 |   0.4 |   0.4 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T3/T4   |   1 |           1 |      0.7 |   0.7 |   0.7 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T4/T5   |   1 |           1 |     -0.2 |  -0.2 |  -0.2 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T5/T6   |   1 |           1 |     -2.4 |  -2.4 |  -2.4 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T6/T7   |   1 |           1 |      0.5 |   0.5 |   0.5 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T7/T8   |   1 |           1 |     -1.2 |  -1.2 |  -1.2 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T8/T9   |   1 |           1 |     -2.7 |  -2.7 |  -2.7 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T9/T10  |   3 |           3 |     -3.3 |  -3.7 |  -2.0 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T10/T11 |   4 |           4 |      2.4 |   1.0 |   2.8 |   nan   |   nan   |         0.0 |
| ap      | konservativ   | T11/T12 |   5 |           5 |      1.2 |   0.0 |   2.7 |    -2.0 |     7.8 |        20.0 |
| ap      | konservativ   | T12/L1  |   7 |           7 |      0.8 |  -2.1 |   2.9 |    -3.1 |     4.5 |         0.0 |
| ap      | konservativ   | L1/L2   |   8 |           7 |     -1.2 |  -2.5 |   0.9 |    -2.4 |     4.7 |         0.0 |
| ap      | konservativ   | L2/L3   |   8 |           7 |     -2.2 |  -3.1 |  -0.6 |    -3.1 |     0.2 |        12.5 |
| ap      | konservativ   | L3/L4   |   8 |           7 |     -0.8 |  -3.9 |   1.6 |    -6.1 |     1.8 |        12.5 |
| ap      | konservativ   | L4/L5   |   8 |           7 |     -1.1 |  -2.7 |   0.7 |    -2.9 |     0.9 |        12.5 |
| ap      | konservativ   | L5/S1   |   5 |           5 |     -1.8 |  -3.4 |  -0.7 |    -3.6 |     0.8 |         0.0 |
| lat     | Dekompression | T12/L1  |   1 |           1 |      0.7 |   0.7 |   0.7 |   nan   |   nan   |         0.0 |
| lat     | Dekompression | L1/L2   |   2 |           2 |      0.4 |  -0.5 |   1.3 |   nan   |   nan   |         0.0 |
| lat     | Dekompression | L2/L3   |   2 |           2 |      0.0 |  -1.9 |   1.9 |   nan   |   nan   |         0.0 |
| lat     | Dekompression | L3/L4   |   2 |           2 |     -0.5 |  -1.9 |   0.9 |   nan   |   nan   |         0.0 |
| lat     | Dekompression | L4/L5   |   2 |           2 |      1.2 |   0.6 |   1.8 |   nan   |   nan   |         0.0 |
| lat     | Dekompression | L5/S1   |   2 |           2 |      4.0 |   3.3 |   4.6 |   nan   |   nan   |         0.0 |
| lat     | Fusion        | C6/C7   |   4 |           3 |     -2.1 |  -3.1 |  -0.9 |   nan   |   nan   |         0.0 |
| lat     | Fusion        | C7/T1   |   4 |           3 |     -1.6 |  -2.2 |  -1.2 |   nan   |   nan   |         0.0 |
| lat     | Fusion        | T1/T2   |   7 |           5 |      2.2 |   0.9 |   4.4 |    -1.6 |     2.7 |         0.0 |
| lat     | Fusion        | T2/T3   |   8 |           6 |     -0.8 |  -1.4 |   2.8 |    -3.3 |     2.7 |         0.0 |
| lat     | Fusion        | T3/T4   |   8 |           6 |     -1.3 |  -3.2 |  -0.3 |    -5.1 |    -0.1 |         0.0 |
| lat     | Fusion        | T4/T5   |   8 |           6 |      0.7 |  -2.3 |   3.4 |    -3.3 |     4.2 |        25.0 |
| lat     | Fusion        | T5/T6   |   8 |           6 |      0.6 |  -1.5 |   1.9 |    -2.5 |     2.5 |         0.0 |
| lat     | Fusion        | T6/T7   |   8 |           6 |      1.7 |  -0.1 |   3.0 |    -2.0 |     3.1 |         0.0 |
| lat     | Fusion        | T7/T8   |   8 |           6 |     -1.1 |  -2.3 |   1.2 |    -2.9 |     2.1 |        12.5 |
| lat     | Fusion        | T8/T9   |   8 |           6 |     -2.6 |  -5.6 |  -1.2 |    -8.5 |    -1.1 |        25.0 |
| lat     | Fusion        | T9/T10  |   8 |           6 |     -2.5 |  -3.7 |   0.6 |    -4.6 |     2.7 |        25.0 |
| lat     | Fusion        | T10/T11 |   9 |           6 |     -2.3 |  -2.9 |   3.1 |    -2.9 |     5.8 |        11.1 |
| lat     | Fusion        | T11/T12 |  21 |          15 |     -1.5 |  -3.6 |   0.4 |    -3.2 |    -0.3 |        19.0 |
| lat     | Fusion        | T12/L1  |  33 |          24 |     -0.8 |  -3.6 |   1.7 |    -2.4 |     1.0 |        15.2 |
| lat     | Fusion        | L1/L2   |  34 |          25 |     -0.6 |  -2.3 |   2.1 |    -1.6 |     1.0 |        23.5 |
| lat     | Fusion        | L2/L3   |  34 |          25 |      1.1 |  -1.5 |   4.2 |    -0.3 |     2.3 |        23.5 |
| lat     | Fusion        | L3/L4   |  34 |          25 |      0.9 |  -2.6 |   3.1 |    -0.7 |     2.1 |        11.8 |
| lat     | Fusion        | L4/L5   |  34 |          25 |      1.1 |  -1.4 |   2.6 |    -0.1 |     2.1 |         8.8 |
| lat     | Fusion        | L5/S1   |  34 |          25 |      0.0 |  -4.0 |   1.4 |    -2.6 |     0.9 |         5.9 |
| lat     | konservativ   | T1/T2   |   2 |           1 |     -1.3 |  -1.7 |  -0.9 |   nan   |   nan   |         0.0 |
| lat     | konservativ   | T2/T3   |   2 |           1 |      0.2 |  -2.6 |   3.0 |   nan   |   nan   |         0.0 |
| lat     | konservativ   | T3/T4   |   2 |           1 |      4.5 |   2.9 |   6.2 |   nan   |   nan   |         0.0 |
| lat     | konservativ   | T4/T5   |   3 |           2 |      2.7 |  -2.6 |   5.0 |   nan   |   nan   |        33.3 |
| lat     | konservativ   | T5/T6   |   3 |           2 |      1.5 |   1.3 |   1.5 |   nan   |   nan   |         0.0 |
| lat     | konservativ   | T6/T7   |   3 |           2 |     -2.1 |  -2.5 |  -1.9 |   nan   |   nan   |         0.0 |
| lat     | konservativ   | T7/T8   |   3 |           2 |     -1.6 |  -2.2 |   0.2 |   nan   |   nan   |         0.0 |
| lat     | konservativ   | T8/T9   |   3 |           2 |     -2.0 |  -2.3 |   1.8 |   nan   |   nan   |         0.0 |
| lat     | konservativ   | T9/T10  |   3 |           2 |     -0.9 |  -3.5 |   3.4 |   nan   |   nan   |        33.3 |
| lat     | konservativ   | T10/T11 |   5 |           4 |      1.9 |   0.1 |   1.9 |   nan   |   nan   |         0.0 |
| lat     | konservativ   | T11/T12 |   5 |           4 |     -2.6 |  -4.7 |   1.3 |   nan   |   nan   |        20.0 |
| lat     | konservativ   | T12/L1  |   7 |           6 |     -2.6 |  -4.1 |   0.8 |    -4.3 |     1.7 |        14.3 |
| lat     | konservativ   | L1/L2   |   8 |           7 |     -1.3 |  -5.0 |   1.1 |    -5.8 |     1.4 |        12.5 |
| lat     | konservativ   | L2/L3   |   8 |           7 |     -2.0 |  -3.6 |  -0.7 |    -5.4 |     0.2 |        12.5 |
| lat     | konservativ   | L3/L4   |   8 |           7 |      2.8 |  -0.8 |   3.3 |    -1.7 |     3.3 |         0.0 |
| lat     | konservativ   | L4/L5   |   8 |           7 |     -1.1 |  -1.5 |   0.1 |    -1.7 |     2.4 |        12.5 |
| lat     | konservativ   | L5/S1   |   8 |           7 |      0.9 |  -1.6 |   1.8 |    -2.0 |     2.0 |        12.5 |
| slip    | Dekompression | T12/L1  |   2 |           2 |     -6.3 |  -8.9 |  -3.7 |   nan   |   nan   |         0.0 |
| slip    | Dekompression | L1/L2   |   2 |           2 |     -0.3 |  -1.2 |   0.6 |   nan   |   nan   |         0.0 |
| slip    | Dekompression | L2/L3   |   2 |           2 |      1.4 |   0.7 |   2.1 |   nan   |   nan   |         0.0 |
| slip    | Dekompression | L3/L4   |   2 |           2 |      3.1 |  -0.8 |   7.0 |   nan   |   nan   |         0.0 |
| slip    | Dekompression | L4/L5   |   2 |           2 |      6.1 |   2.7 |   9.4 |   nan   |   nan   |         0.0 |
| slip    | Dekompression | L5/S1   |   2 |           2 |      0.6 |  -0.2 |   1.4 |   nan   |   nan   |         0.0 |
| slip    | Fusion        | T12/L1  |  19 |          17 |     -0.4 |  -1.3 |   3.7 |    -1.3 |     2.3 |         0.0 |
| slip    | Fusion        | L1/L2   |  17 |          16 |     -1.0 |  -2.2 |   4.0 |    -1.7 |     4.0 |         0.0 |
| slip    | Fusion        | L2/L3   |  11 |          11 |     -0.7 |  -2.3 |  -0.3 |    -3.3 |    -0.2 |         0.0 |
| slip    | Fusion        | L3/L4   |  11 |          11 |      4.1 |  -0.1 |   5.4 |    -1.5 |     5.4 |         0.0 |
| slip    | Fusion        | L4/L5   |   9 |           9 |     -1.8 |  -4.0 |   0.9 |    -4.2 |     6.1 |        11.1 |
| slip    | Fusion        | L5/S1   |  11 |          11 |      2.8 |  -4.4 |   6.7 |    -4.8 |     8.5 |         9.1 |
| slip    | konservativ   | T12/L1  |   6 |           6 |      0.8 |  -1.7 |   2.2 |    -3.7 |     3.3 |         0.0 |
| slip    | konservativ   | L1/L2   |   6 |           6 |      3.4 |  -2.8 |   5.4 |    -4.8 |     8.0 |         0.0 |
| slip    | konservativ   | L2/L3   |   6 |           6 |      0.1 |  -1.8 |   1.3 |    -5.9 |     1.8 |         0.0 |
| slip    | konservativ   | L3/L4   |   6 |           6 |      2.4 |  -1.7 |   4.2 |    -2.9 |     5.7 |         0.0 |
| slip    | konservativ   | L4/L5   |   6 |           6 |      0.8 |  -1.5 |   2.1 |    -2.0 |     2.9 |         0.0 |
| slip    | konservativ   | L5/S1   |   6 |           6 |     -0.2 |  -1.2 |   1.8 |    -3.1 |     2.1 |         0.0 |

### Δ und K je Segmentklasse (K = |prä| − |post|)
| plane   | group         | cls           |   n |   Patienten |   Median |   Q25 |   Q75 |   CI_lo |   CI_hi |   > MDC95 % | Größe   |
|:--------|:--------------|:--------------|----:|------------:|---------:|------:|------:|--------:|--------:|------------:|:--------|
| ap      | Fusion        | fern          |  65 |          18 |      0.5 |  -1.1 |   2.0 |    -0.2 |     1.5 |         4.6 | Δ       |
| ap      | Fusion        | UIV+2         |  22 |          18 |     -1.3 |  -2.2 |   1.0 |    -2.2 |     0.8 |         0.0 | Δ       |
| ap      | Fusion        | UIV+1         |  26 |          21 |      0.1 |  -2.7 |   1.2 |    -0.9 |     1.1 |         7.7 | Δ       |
| ap      | Fusion        | fusioniert    | 177 |          25 |      0.2 |  -2.3 |   3.7 |    -0.4 |     0.6 |        23.7 | Δ       |
| lat     | Fusion        | fern          |  88 |          18 |     -0.5 |  -2.4 |   1.6 |    -1.4 |    -0.1 |         4.5 | Δ       |
| lat     | Fusion        | UIV+2         |  22 |          18 |     -2.3 |  -3.6 |  -1.2 |    -3.4 |    -1.2 |        18.2 | Δ       |
| lat     | Fusion        | UIV+1         |  25 |          19 |     -1.7 |  -4.3 |   2.4 |    -3.2 |     0.1 |        16.0 | Δ       |
| lat     | Fusion        | fusioniert    | 168 |          25 |      0.8 |  -2.7 |   3.1 |    -0.2 |     1.2 |        17.3 | Δ       |
| lat     | Fusion        | LIV−1         |   9 |           9 |      0.2 |  -0.9 |   0.9 |    -0.9 |     3.5 |        11.1 | Δ       |
| slip    | Fusion        | fern          |  20 |          11 |     -1.1 |  -3.1 |   0.4 |    -2.9 |     0.2 |         0.0 | Δ       |
| slip    | Fusion        | UIV+2         |  14 |          13 |     -0.3 |  -0.8 |   1.2 |    -0.8 |     1.2 |         0.0 | Δ       |
| slip    | Fusion        | UIV+1         |  16 |          14 |      3.1 |  -0.9 |   5.6 |    -0.7 |     5.6 |         0.0 | Δ       |
| slip    | Fusion        | LIV−1         |   8 |           8 |      0.2 |  -3.9 |   5.0 |    -4.0 |     5.3 |         0.0 | Δ       |
| ap      | Fusion        | LIV−1         |  10 |           9 |      2.1 |   0.3 |   3.6 |     0.2 |     3.9 |         0.0 | Δ       |
| slip    | Fusion        | fusioniert    |  20 |           5 |     -0.8 |  -6.5 |   7.1 |    -6.7 |     2.8 |        10.0 | Δ       |
| ap      | konservativ   | alle          |  65 |           7 |     -1.1 |  -2.8 |   1.3 |    -1.6 |    -0.7 |         6.2 | Δ       |
| lat     | konservativ   | alle          |  81 |           7 |     -0.7 |  -2.6 |   1.9 |    -1.6 |     0.7 |         9.9 | Δ       |
| slip    | konservativ   | alle          |  36 |           6 |      0.8 |  -2.3 |   2.9 |    -1.2 |     2.0 |         0.0 | Δ       |
| ap      | Dekompression | fern          |   5 |           2 |      1.0 |   0.2 |   2.5 |   nan   |   nan   |         0.0 | Δ       |
| ap      | Dekompression | benachbart    |   3 |           2 |      1.4 |  -0.6 |   1.6 |   nan   |   nan   |         0.0 | Δ       |
| ap      | Dekompression | dekomprimiert |   3 |           2 |     -0.3 |  -1.0 |   1.1 |   nan   |   nan   |         0.0 | Δ       |
| lat     | Dekompression | fern          |   4 |           2 |     -0.3 |  -2.0 |   1.1 |   nan   |   nan   |         0.0 | Δ       |
| lat     | Dekompression | benachbart    |   4 |           2 |      3.2 |   1.2 |   4.2 |   nan   |   nan   |         0.0 | Δ       |
| lat     | Dekompression | dekomprimiert |   3 |           2 |      2.4 |   1.2 |   2.4 |   nan   |   nan   |         0.0 | Δ       |
| slip    | Dekompression | fern          |   5 |           2 |     -1.0 |  -2.1 |  -0.1 |   nan   |   nan   |         0.0 | Δ       |
| slip    | Dekompression | benachbart    |   4 |           2 |      0.6 |  -1.9 |   2.4 |   nan   |   nan   |         0.0 | Δ       |
| slip    | Dekompression | dekomprimiert |   3 |           2 |     10.9 |   5.1 |  11.8 |   nan   |   nan   |         0.0 | Δ       |
| ap      | Fusion        | fern          |  65 |          18 |      0.4 |  -0.7 |   1.4 |    -0.0 |     0.9 |       nan   | K       |
| ap      | Fusion        | UIV+2         |  22 |          18 |      0.5 |  -1.0 |   1.6 |    -0.4 |     1.5 |       nan   | K       |
| ap      | Fusion        | UIV+1         |  26 |          21 |      0.5 |  -0.1 |   2.6 |     0.1 |     2.0 |       nan   | K       |
| ap      | Fusion        | fusioniert    | 177 |          25 |      1.7 |  -0.2 |   5.6 |     1.0 |     2.6 |       nan   | K       |
| lat     | Fusion        | fern          |  88 |          18 |     -0.3 |  -1.6 |   1.6 |    -0.8 |     0.1 |       nan   | K       |
| lat     | Fusion        | UIV+2         |  22 |          18 |      0.1 |  -1.9 |   2.3 |    -1.4 |     1.4 |       nan   | K       |
| lat     | Fusion        | UIV+1         |  25 |          19 |      0.1 |  -2.4 |   2.7 |    -2.0 |     1.7 |       nan   | K       |
| lat     | Fusion        | fusioniert    | 168 |          25 |     -0.2 |  -2.7 |   2.8 |    -0.8 |     0.2 |       nan   | K       |
| lat     | Fusion        | LIV−1         |   9 |           9 |     -0.2 |  -0.9 |   0.9 |    -3.5 |     0.9 |       nan   | K       |
| slip    | Fusion        | fern          |  20 |          11 |     -1.0 |  -2.5 |   0.9 |    -1.8 |     1.4 |       nan   | K       |
| slip    | Fusion        | UIV+2         |  14 |          13 |     -0.2 |  -0.7 |   1.0 |    -0.7 |     1.2 |       nan   | K       |
| slip    | Fusion        | UIV+1         |  16 |          14 |     -0.0 |  -2.6 |   4.1 |    -2.3 |     4.0 |       nan   | K       |
| slip    | Fusion        | LIV−1         |   8 |           8 |      4.1 |  -3.8 |   5.0 |    -3.9 |     5.3 |       nan   | K       |
| ap      | Fusion        | LIV−1         |  10 |           9 |      0.5 |   0.3 |   1.0 |     0.2 |     1.1 |       nan   | K       |
| slip    | Fusion        | fusioniert    |  20 |           5 |     -0.0 |  -4.5 |   1.8 |    -0.5 |     1.0 |       nan   | K       |
| ap      | konservativ   | alle          |  65 |           7 |     -1.2 |  -3.4 |   0.7 |    -2.4 |     0.1 |       nan   | K       |
| lat     | konservativ   | alle          |  81 |           7 |     -0.2 |  -2.6 |   1.8 |    -1.6 |     1.0 |       nan   | K       |
| slip    | konservativ   | alle          |  36 |           6 |      1.2 |  -1.5 |   2.4 |    -0.8 |     2.0 |       nan   | K       |
| ap      | Dekompression | fern          |   5 |           2 |      1.7 |   1.0 |   2.6 |   nan   |   nan   |       nan   | K       |
| ap      | Dekompression | benachbart    |   3 |           2 |     -1.7 |  -2.1 |  -1.6 |   nan   |   nan   |       nan   | K       |
| ap      | Dekompression | dekomprimiert |   3 |           2 |      0.3 |   0.1 |   0.7 |   nan   |   nan   |       nan   | K       |
| lat     | Dekompression | fern          |   4 |           2 |      1.1 |   0.7 |   1.6 |   nan   |   nan   |       nan   | K       |
| lat     | Dekompression | benachbart    |   4 |           2 |     -3.2 |  -4.2 |  -1.2 |   nan   |   nan   |       nan   | K       |
| lat     | Dekompression | dekomprimiert |   3 |           2 |     -2.4 |  -2.4 |  -1.2 |   nan   |   nan   |       nan   | K       |
| slip    | Dekompression | fern          |   5 |           2 |     -0.1 |  -2.1 |   1.0 |   nan   |   nan   |       nan   | K       |
| slip    | Dekompression | benachbart    |   4 |           2 |      0.6 |  -1.9 |   2.4 |   nan   |   nan   |       nan   | K       |
| slip    | Dekompression | dekomprimiert |   3 |           2 |    -10.9 | -11.5 |  -5.7 |   nan   |   nan   |       nan   | K       |

### Listhese (|Slip| ≥ 10 % WKD) prä vs. post, Anteil in %
| group         | seg    |   n |   prae |   post |
|:--------------|:-------|----:|-------:|-------:|
| Dekompression | L1/L2  |   2 |    0.0 |    0.0 |
| Dekompression | L2/L3  |   2 |   50.0 |   50.0 |
| Dekompression | L3/L4  |   2 |    0.0 |  100.0 |
| Dekompression | L4/L5  |   2 |    0.0 |   50.0 |
| Dekompression | L5/S1  |   2 |   50.0 |    0.0 |
| Dekompression | T12/L1 |   2 |    0.0 |   50.0 |
| Fusion        | L1/L2  |  17 |   23.5 |   17.6 |
| Fusion        | L2/L3  |  11 |    9.1 |   27.3 |
| Fusion        | L3/L4  |  11 |    0.0 |    0.0 |
| Fusion        | L4/L5  |   9 |   33.3 |   22.2 |
| Fusion        | L5/S1  |  11 |   36.4 |   18.2 |
| Fusion        | T12/L1 |  19 |    5.3 |    0.0 |
| konservativ   | L1/L2  |   6 |    0.0 |    0.0 |
| konservativ   | L2/L3  |   6 |   33.3 |    0.0 |
| konservativ   | L3/L4  |   6 |   33.3 |   33.3 |
| konservativ   | L4/L5  |   6 |   16.7 |   16.7 |
| konservativ   | L5/S1  |   6 |   16.7 |   33.3 |
| konservativ   | T12/L1 |   6 |    0.0 |    0.0 |

## F2 – Ging eine Skoliosekorrektur zu Lasten der Lordose?

Je Episode: `K_ap_sum` = Σ K a.p. über die behandelten Segmente (konservativ: alle lumbalen), `k_cCobb_max` = Korrektur
der Hauptkurve, `d_LL` = Δ Lordose L1–S1 (Deckplatten, degen), `dLL_seg` = Σ Δ seitlich L1/L2–L5/S1.
Negatives ρ zwischen Korrektur und ΔLL hieße: mehr Skoliosekorrektur, mehr Lordoseverlust.

| group         | x           | y       |   n |   rho |   Patienten |   CI_lo |   CI_hi |
|:--------------|:------------|:--------|----:|------:|------------:|--------:|--------:|
| Dekompression | K_ap_sum    | d_LL    |   2 | nan   |           2 |   nan   |   nan   |
| Dekompression | K_ap_sum    | dLL_seg |   2 | nan   |           2 |   nan   |   nan   |
| Dekompression | k_cCobb_max | d_LL    |   2 | nan   |           2 |   nan   |   nan   |
| Fusion        | K_ap_sum    | d_LL    |  26 |   0.5 |          19 |     0.0 |     0.8 |
| Fusion        | K_ap_sum    | dLL_seg |  33 |   0.6 |          24 |     0.2 |     0.8 |
| Fusion        | k_cCobb_max | d_LL    |  27 |   0.4 |          20 |     0.0 |     0.7 |
| konservativ   | K_ap_sum    | d_LL    |   7 |   0.6 |           6 |    -0.4 |     1.0 |
| konservativ   | K_ap_sum    | dLL_seg |   8 |   0.7 |           7 |    -0.0 |     1.0 |
| konservativ   | k_cCobb_max | d_LL    |   7 |   0.6 |           6 |    -0.1 |     1.0 |

Je Segment (gleiches Segment): K a.p. bzw. Reposition des Slips gegen Δ seitlich.

| group         | x      | y     |   n |   rho |   Patienten |   CI_lo |   CI_hi |
|:--------------|:-------|:------|----:|------:|------------:|--------:|--------:|
| Dekompression | k_ap   | d_lat |  10 |  -0.3 |           2 |   nan   |   nan   |
| Dekompression | k_slip | d_lat |  11 |   0.3 |           2 |   nan   |   nan   |
| Fusion        | k_ap   | d_lat | 190 |   0.2 |          25 |     0.0 |     0.3 |
| Fusion        | k_slip | d_lat |  78 |   0.0 |          18 |    -0.2 |     0.2 |
| konservativ   | k_ap   | d_lat |  43 |   0.1 |           7 |    -0.2 |     0.3 |
| konservativ   | k_slip | d_lat |  35 |   0.1 |           6 |    -0.5 |     0.6 |

## F3 – Hängt die Cage-Einbringseite mit der Öffnungsrichtung zusammen?

Cages mit a.p.-Seite: 66. Kreuztabelle Einbringseite × Öffnung des Segments präop.:

| side       |   offen links |   offen rechts |   – |   All |
|:-----------|--------------:|---------------:|----:|------:|
| von links  |            10 |             13 |   1 |    24 |
| von rechts |            31 |             10 |   1 |    42 |
| All        |            41 |             23 |   2 |    66 |

Korrektur am Cage-Segment je Zugang (konvex = von der offenen Seite):

| approach            |   n |   Patienten |   Median |   Q25 |   Q75 |   CI_lo |   CI_hi | Größe   |
|:--------------------|----:|------------:|---------:|------:|------:|--------:|--------:|:--------|
| neutral             |  50 |          19 |      0.6 |  -0.7 |   2.3 |    -0.4 |     1.7 | k_ap    |
| konkav              |   9 |           7 |      8.4 |   7.7 |  11.2 |     5.7 |    21.6 | k_ap    |
| konvex              |   1 |           1 |     11.2 |  11.2 |  11.2 |   nan   |   nan   | k_ap    |
| neutral             |  52 |          20 |      1.1 |  -1.8 |   3.0 |    -0.3 |     1.9 | d_lat   |
| keine a.p.-Baseline |   1 |           1 |      1.7 |   1.7 |   1.7 |   nan   |   nan   | d_lat   |
| konkav              |   8 |           6 |      1.9 |  -2.3 |   4.2 |    -2.7 |     5.4 | d_lat   |
| konvex              |   1 |           1 |      7.6 |   7.6 |   7.6 |   nan   |   nan   | d_lat   |
| konkav              |   3 |           2 |      1.4 |  -0.2 |   1.4 |   nan   |   nan   | k_slip  |
| neutral             |   5 |           3 |     -5.9 |  -6.4 |  -5.8 |   nan   |   nan   | k_slip  |
| konvex              |   1 |           1 |      0.1 |   0.1 |   0.1 |   nan   |   nan   | k_slip  |

## F4 – Größere segmentale Korrektur bei langstreckiger Fusion?

Kurz = 1–2 neu fusionierte Segmente, lang = ≥ 3 oder Becken. `K_mean` = mittlere K a.p. je fusioniertem Segment,
`rel_mean` = K/|prä|, `pre_abs` = mittlere |prä| (Confounding). Spearman Anzahl Segmente ~ K_mean:
ρ = 0.09 (n = 35, CI -0.23; 0.41).

| Länge   |   n |   Patienten |   Median |   Q25 |   Q75 |   CI_lo |   CI_hi | Größe    |
|:--------|----:|------------:|---------:|------:|------:|--------:|--------:|:---------|
| lang    |  27 |          19 |      1.2 |   0.1 |   5.8 |     0.6 |     5.2 | K_mean   |
| kurz    |   8 |           8 |      3.2 |   1.6 |   4.6 |     0.8 |     5.3 | K_mean   |
| lang    |  26 |          19 |      0.4 |   0.1 |   0.6 |     0.2 |     0.6 | rel_mean |
| kurz    |   8 |           8 |      0.7 |   0.4 |   0.8 |     0.2 |     0.8 | rel_mean |
| lang    |  27 |          19 |      3.6 |   2.1 |   8.6 |     2.4 |     7.4 | pre_abs  |
| kurz    |   9 |           9 |      4.5 |   4.1 |   5.4 |     2.2 |     8.1 | pre_abs  |

## F5a – Segmental → regional → global

Δ der regionalen/globalen Parameter (degen_pipeline, Status ok):

| group         | seg       |   n |   Patienten |   Median |   Q25 |   Q75 |   CI_lo |   CI_hi |
|:--------------|:----------|----:|------------:|---------:|------:|------:|--------:|--------:|
| Dekompression | L1PA      |   1 |           1 |      0.6 |   0.6 |   0.6 |   nan   |   nan   |
| Dekompression | L4_tilt   |   2 |           2 |      1.5 |   0.8 |   2.3 |   nan   |   nan   |
| Dekompression | L5_tilt   |   2 |           2 |      1.8 |   1.7 |   1.8 |   nan   |   nan   |
| Dekompression | LL        |   2 |           2 |      5.4 |   0.8 |  10.0 |   nan   |   nan   |
| Dekompression | PT        |   1 |           1 |      0.6 |   0.6 |   0.6 |   nan   |   nan   |
| Dekompression | SS        |   2 |           2 |      1.3 |   1.1 |   1.4 |   nan   |   nan   |
| Dekompression | cCobb_max |   2 |           2 |     -6.9 |  -7.5 |  -6.3 |   nan   |   nan   |
| Fusion        | L4_tilt   |  33 |          24 |     -1.8 |  -2.9 |   0.1 |    -2.6 |    -0.4 |
| Fusion        | SS        |  29 |          21 |     -0.7 |  -4.0 |   4.1 |    -3.6 |     2.1 |
| Fusion        | cCobb_max |  36 |          26 |     -8.5 | -17.8 |  -2.1 |   -12.0 |    -4.4 |
| Fusion        | L1PA      |  13 |           9 |     -0.6 |  -3.8 |   3.7 |    -3.8 |     3.7 |
| Fusion        | L5_tilt   |  28 |          19 |     -0.9 |  -2.8 |   0.3 |    -2.2 |    -0.1 |
| Fusion        | LL        |  27 |          20 |      1.8 | -10.1 |   9.0 |    -1.3 |     5.9 |
| Fusion        | PT        |  13 |           9 |      0.3 |  -4.4 |   5.3 |    -4.4 |     3.1 |
| Fusion        | SVA_wkd   |   6 |           4 |     -0.7 |  -0.9 |  -0.1 |   nan   |   nan   |
| Fusion        | T1PA      |   3 |           3 |     -3.9 |  -4.2 |   0.7 |   nan   |   nan   |
| konservativ   | L4_tilt   |   7 |           6 |      1.7 |  -0.0 |   3.8 |    -0.8 |     4.3 |
| konservativ   | L5_tilt   |   8 |           7 |     -0.2 |  -2.9 |   1.5 |    -3.5 |     1.7 |
| konservativ   | LL        |   7 |           6 |     -1.5 |  -5.8 |  -0.7 |    -9.6 |     5.2 |
| konservativ   | SS        |   7 |           6 |      0.1 |  -3.6 |   1.3 |    -4.6 |     2.5 |
| konservativ   | cCobb_max |   8 |           7 |      9.7 |   1.8 |  16.7 |     1.7 |    18.5 |
| konservativ   | L1PA      |   3 |           2 |     -0.5 |  -1.4 |  -0.3 |   nan   |   nan   |
| konservativ   | PT        |   3 |           2 |     -4.4 |  -5.0 |  -1.0 |   nan   |   nan   |
| konservativ   | SVA_wkd   |   2 |           1 |     -0.0 |  -0.2 |   0.2 |   nan   |   nan   |
| konservativ   | T1PA      |   1 |           1 |     -0.2 |  -0.2 |  -0.2 |   nan   |   nan   |

Beitrag der Segmente zur lumbalen Gesamtänderung (Δ_Segment · sign(ΣΔ) / Σ|Δ|, > 0 = in Richtung der
Gesamtänderung; Median je Gruppe):

| plane   | group         | seg    |   n |   Patienten |   Median |   Q25 |   Q75 |   CI_lo |   CI_hi |
|:--------|:--------------|:-------|----:|------------:|---------:|------:|------:|--------:|--------:|
| ap      | Dekompression | T12/L1 |   1 |           1 |      0.3 |   0.3 |   0.3 |   nan   |   nan   |
| ap      | Dekompression | L1/L2  |   2 |           2 |      0.4 |   0.3 |   0.4 |   nan   |   nan   |
| ap      | Dekompression | L2/L3  |   2 |           2 |     -0.0 |  -0.1 |  -0.0 |   nan   |   nan   |
| ap      | Dekompression | L3/L4  |   2 |           2 |      0.1 |   0.1 |   0.2 |   nan   |   nan   |
| ap      | Dekompression | L4/L5  |   2 |           2 |     -0.2 |  -0.2 |  -0.2 |   nan   |   nan   |
| ap      | Dekompression | L5/S1  |   1 |           1 |      0.2 |   0.2 |   0.2 |   nan   |   nan   |
| ap      | Fusion        | T12/L1 |  33 |          25 |     -0.1 |  -0.1 |   0.1 |    -0.1 |     0.1 |
| ap      | Fusion        | L1/L2  |  36 |          26 |     -0.0 |  -0.1 |   0.2 |    -0.1 |     0.1 |
| ap      | Fusion        | L2/L3  |  36 |          26 |      0.2 |   0.0 |   0.3 |     0.0 |     0.3 |
| ap      | Fusion        | L3/L4  |  36 |          26 |      0.2 |  -0.1 |   0.3 |     0.0 |     0.3 |
| ap      | Fusion        | L4/L5  |  36 |          26 |      0.0 |  -0.1 |   0.1 |    -0.0 |     0.0 |
| ap      | Fusion        | L5/S1  |  24 |          16 |     -0.0 |  -0.0 |   0.3 |    -0.0 |     0.3 |
| ap      | konservativ   | T12/L1 |   7 |           7 |      0.1 |  -0.1 |   0.2 |    -0.2 |     0.2 |
| ap      | konservativ   | L1/L2  |   8 |           7 |      0.1 |  -0.2 |   0.2 |    -0.2 |     0.2 |
| ap      | konservativ   | L2/L3  |   8 |           7 |      0.1 |  -0.2 |   0.2 |    -0.2 |     0.2 |
| ap      | konservativ   | L3/L4  |   8 |           7 |      0.2 |  -0.1 |   0.2 |    -0.0 |     0.3 |
| ap      | konservativ   | L4/L5  |   8 |           7 |      0.1 |   0.0 |   0.1 |    -0.0 |     0.1 |
| ap      | konservativ   | L5/S1  |   5 |           5 |      0.1 |  -0.1 |   0.1 |    -0.1 |     0.3 |
| lat     | Dekompression | T12/L1 |   1 |           1 |      0.0 |   0.0 |   0.0 |   nan   |   nan   |
| lat     | Dekompression | L1/L2  |   2 |           2 |      0.0 |  -0.0 |   0.1 |   nan   |   nan   |
| lat     | Dekompression | L2/L3  |   2 |           2 |      0.0 |  -0.1 |   0.2 |   nan   |   nan   |
| lat     | Dekompression | L3/L4  |   2 |           2 |     -0.0 |  -0.1 |   0.1 |   nan   |   nan   |
| lat     | Dekompression | L4/L5  |   2 |           2 |      0.1 |   0.1 |   0.1 |   nan   |   nan   |
| lat     | Dekompression | L5/S1  |   2 |           2 |      0.3 |   0.2 |   0.3 |   nan   |   nan   |
| lat     | Fusion        | T12/L1 |  33 |          24 |      0.1 |  -0.1 |   0.2 |    -0.0 |     0.2 |
| lat     | Fusion        | L1/L2  |  34 |          25 |      0.1 |  -0.0 |   0.2 |     0.0 |     0.2 |
| lat     | Fusion        | L2/L3  |  34 |          25 |      0.1 |   0.0 |   0.2 |     0.0 |     0.2 |
| lat     | Fusion        | L3/L4  |  34 |          25 |      0.1 |   0.1 |   0.2 |     0.1 |     0.2 |
| lat     | Fusion        | L4/L5  |  34 |          25 |     -0.1 |  -0.1 |   0.1 |    -0.1 |     0.0 |
| lat     | Fusion        | L5/S1  |  34 |          25 |      0.1 |  -0.1 |   0.3 |    -0.0 |     0.2 |
| lat     | konservativ   | T12/L1 |   7 |           6 |      0.2 |   0.1 |   0.3 |     0.1 |     0.3 |
| lat     | konservativ   | L1/L2  |   8 |           7 |      0.1 |   0.0 |   0.2 |    -0.0 |     0.3 |
| lat     | konservativ   | L2/L3  |   8 |           7 |      0.1 |   0.0 |   0.2 |    -0.0 |     0.2 |
| lat     | konservativ   | L3/L4  |   8 |           7 |     -0.1 |  -0.2 |   0.1 |    -0.2 |     0.2 |
| lat     | konservativ   | L4/L5  |   8 |           7 |     -0.0 |  -0.2 |   0.1 |    -0.2 |     0.1 |
| lat     | konservativ   | L5/S1  |   8 |           7 |     -0.0 |  -0.1 |   0.1 |    -0.2 |     0.2 |

## F5b – Nachbarsegmente kurzstreckiger Fusionen

Änderungsrate pro Jahr (ab 3 Monaten Verlauf), Referenz = alle lumbalen Segmente konservativer Episoden:

| plane   | cls                    |   n |   Patienten |   Median |   Q25 |   Q75 |   CI_lo |   CI_hi |
|:--------|:-----------------------|----:|------------:|---------:|------:|------:|--------:|--------:|
| ap      | fern                   |  14 |           4 |      1.1 |   0.2 |   2.4 |     nan |     nan |
| ap      | UIV+2                  |   4 |           4 |      0.1 |  -1.1 |   1.0 |     nan |     nan |
| ap      | UIV+1                  |   4 |           4 |     -0.6 |  -2.2 |   0.6 |     nan |     nan |
| lat     | fern                   |   9 |           4 |     -0.0 |  -3.2 |   0.8 |     nan |     nan |
| lat     | UIV+2                  |   4 |           4 |     -0.8 |  -2.0 |  -0.3 |     nan |     nan |
| lat     | UIV+1                  |   4 |           4 |      0.3 |  -1.0 |   2.7 |     nan |     nan |
| lat     | LIV−1                  |   3 |           3 |      1.0 |  -1.5 |   1.3 |     nan |     nan |
| slip    | fern                   |   9 |           4 |     -1.9 |  -5.0 |   0.4 |     nan |     nan |
| slip    | UIV+2                  |   3 |           3 |      0.5 |  -0.2 |   1.9 |     nan |     nan |
| slip    | UIV+1                  |   4 |           4 |      4.4 |   1.9 |  10.6 |     nan |     nan |
| slip    | LIV−1                  |   3 |           3 |      2.6 |   2.4 |   4.1 |     nan |     nan |
| ap      | LIV−1                  |   2 |           2 |      1.1 |   0.6 |   1.6 |     nan |     nan |
| ap      | konservativ (Referenz) |  27 |           4 |     -0.8 |  -2.7 |   1.0 |     nan |     nan |
| lat     | konservativ (Referenz) |  30 |           4 |     -0.9 |  -1.9 |   1.4 |     nan |     nan |
| slip    | konservativ (Referenz) |  12 |           2 |      0.5 |  -0.8 |   1.5 |     nan |     nan |

Neu aufgetretene Listhesen (|Slip| prä < 10 %, post ≥ 10 %):

| cls                    |   Segmente |   neu |
|:-----------------------|-----------:|------:|
| LIV−1                  |          3 |     0 |
| UIV+1                  |          4 |     0 |
| UIV+2                  |          3 |     0 |
| fern                   |          9 |     2 |
| konservativ (Referenz) |         12 |     1 |

## F6 – Ging die Korrektur wieder verloren?

Korrekturverlust seit `early` (> 0 = zurück in Richtung Ausgangswert) je Zeitklasse (Monate nach Ereignis):

| plane   | Zeit   | cls        |   n |   Patienten |   Median |   Q25 |   Q75 |   CI_lo |   CI_hi |   Verlust > MDC95 % |
|:--------|:-------|:-----------|----:|------------:|---------:|------:|------:|--------:|--------:|--------------------:|
| ap      | 0–3    | UIV+1      |   4 |           4 |      1.1 |  -0.7 |   3.3 |   nan   |   nan   |                 0.0 |
| ap      | 0–3    | UIV+2      |   3 |           3 |      4.6 |   3.5 |   7.1 |   nan   |   nan   |                33.3 |
| ap      | 0–3    | fern       |  15 |           3 |      1.1 |   0.5 |   2.9 |   nan   |   nan   |                 0.0 |
| ap      | 0–3    | fusioniert |  29 |           5 |      0.0 |  -1.7 |   1.7 |    -1.0 |     0.8 |                 3.4 |
| ap      | 3–6    | UIV+1      |   1 |           1 |      5.0 |   5.0 |   5.0 |   nan   |   nan   |                 0.0 |
| ap      | 3–6    | UIV+2      |   1 |           1 |      5.6 |   5.6 |   5.6 |   nan   |   nan   |                 0.0 |
| ap      | 3–6    | fern       |   4 |           1 |      3.5 |   2.3 |   4.6 |   nan   |   nan   |                 0.0 |
| ap      | 3–6    | fusioniert |  11 |           2 |     -1.9 |  -3.0 |   0.9 |   nan   |   nan   |                 0.0 |
| ap      | 6–12   | UIV+1      |   3 |           3 |      2.6 |  -0.0 |   2.8 |   nan   |   nan   |                 0.0 |
| ap      | 6–12   | UIV+2      |   3 |           3 |      5.9 |   4.0 |   6.1 |   nan   |   nan   |                 0.0 |
| ap      | 6–12   | fern       |   3 |           2 |      1.9 |   0.4 |   3.4 |   nan   |   nan   |                 0.0 |
| ap      | 6–12   | fusioniert |  27 |           5 |      0.3 |  -1.4 |   1.6 |    -2.7 |     1.4 |                 3.7 |
| ap      | 12–24  | LIV−1      |   2 |           2 |     -0.6 |  -2.2 |   0.9 |   nan   |   nan   |                 0.0 |
| ap      | 12–24  | UIV+1      |   4 |           4 |      1.2 |   0.7 |   2.7 |   nan   |   nan   |                 0.0 |
| ap      | 12–24  | UIV+2      |   2 |           2 |      1.1 |   0.9 |   1.4 |   nan   |   nan   |                 0.0 |
| ap      | 12–24  | fern       |  12 |           3 |      1.0 |  -1.0 |   2.0 |   nan   |   nan   |                 0.0 |
| ap      | 12–24  | fusioniert |  40 |           6 |      1.2 |  -1.2 |   2.6 |    -0.4 |     1.8 |                 5.0 |
| ap      | 24–36  | LIV−1      |   1 |           1 |      0.5 |   0.5 |   0.5 |   nan   |   nan   |                 0.0 |
| ap      | 24–36  | UIV+1      |   2 |           2 |     -1.2 |  -1.6 |  -0.7 |   nan   |   nan   |                 0.0 |
| ap      | 24–36  | UIV+2      |   2 |           2 |     -1.9 |  -2.3 |  -1.5 |   nan   |   nan   |                 0.0 |
| ap      | 24–36  | fern       |   9 |           2 |      0.6 |  -1.1 |   1.4 |   nan   |   nan   |                22.2 |
| ap      | 24–36  | fusioniert |   9 |           2 |     -2.0 |  -2.6 |  -1.0 |   nan   |   nan   |                 0.0 |
| ap      | 36–60  | LIV−1      |   1 |           1 |     -0.0 |  -0.0 |  -0.0 |   nan   |   nan   |                 0.0 |
| ap      | 36–60  | UIV+1      |   3 |           3 |      2.8 |   1.1 |   4.5 |   nan   |   nan   |                 0.0 |
| ap      | 36–60  | UIV+2      |   3 |           3 |      2.1 |   0.9 |   2.8 |   nan   |   nan   |                 0.0 |
| ap      | 36–60  | fern       |   3 |           1 |     -2.1 |  -3.9 |   0.5 |   nan   |   nan   |                 0.0 |
| ap      | 36–60  | fusioniert |  15 |           4 |      1.0 |  -0.4 |   2.5 |   nan   |   nan   |                 0.0 |
| lat     | 0–3    | LIV−1      |   1 |           1 |      1.6 |   1.6 |   1.6 |   nan   |   nan   |                 0.0 |
| lat     | 0–3    | UIV+1      |   5 |           4 |     -1.2 |  -2.5 |   4.3 |   nan   |   nan   |                 0.0 |
| lat     | 0–3    | UIV+2      |   4 |           4 |      2.6 |   1.9 |   3.3 |   nan   |   nan   |                 0.0 |
| lat     | 0–3    | fern       |  22 |           4 |      1.9 |   0.2 |   3.1 |   nan   |   nan   |                 4.5 |
| lat     | 0–3    | fusioniert |  34 |           6 |      1.4 |   0.1 |   3.9 |     0.7 |     2.1 |                 5.9 |
| lat     | 3–6    | UIV+1      |   2 |           2 |     -3.4 |  -4.8 |  -2.0 |   nan   |   nan   |                 0.0 |
| lat     | 3–6    | UIV+2      |   2 |           2 |      1.4 |   0.9 |   1.9 |   nan   |   nan   |                 0.0 |
| lat     | 3–6    | fern       |  10 |           2 |      0.8 |   0.2 |   1.8 |   nan   |   nan   |                 0.0 |
| lat     | 3–6    | fusioniert |  14 |           3 |      1.1 |  -1.9 |   3.1 |   nan   |   nan   |                 0.0 |
| lat     | 6–12   | LIV−1      |   1 |           1 |     10.3 |  10.3 |  10.3 |   nan   |   nan   |                 0.0 |
| lat     | 6–12   | UIV+1      |   3 |           3 |     -0.1 |  -0.5 |   0.1 |   nan   |   nan   |                 0.0 |
| lat     | 6–12   | UIV+2      |   2 |           2 |      0.3 |  -0.5 |   1.1 |   nan   |   nan   |                 0.0 |
| lat     | 6–12   | fern       |  12 |           2 |      1.7 |  -0.6 |   3.0 |   nan   |   nan   |                 0.0 |
| lat     | 6–12   | fusioniert |  26 |           5 |      1.5 |  -0.7 |   2.9 |    -0.4 |     2.4 |                 3.8 |
| lat     | 12–24  | LIV−1      |   2 |           2 |      5.6 |   1.7 |   9.4 |   nan   |   nan   |                50.0 |
| lat     | 12–24  | UIV+1      |   3 |           3 |      0.4 |  -4.6 |   2.3 |   nan   |   nan   |                 0.0 |
| lat     | 12–24  | UIV+2      |   2 |           2 |      5.2 |   4.8 |   5.7 |   nan   |   nan   |                 0.0 |
| lat     | 12–24  | fern       |  11 |           3 |      3.1 |   0.7 |   4.3 |   nan   |   nan   |                18.2 |
| lat     | 12–24  | fusioniert |  36 |           5 |      0.3 |  -1.9 |   3.0 |    -1.9 |     2.3 |                11.1 |
| lat     | 24–36  | LIV−1      |   1 |           1 |      0.1 |   0.1 |   0.1 |   nan   |   nan   |                 0.0 |
| lat     | 24–36  | UIV+1      |   2 |           2 |      0.8 |  -0.7 |   2.2 |   nan   |   nan   |                 0.0 |
| lat     | 24–36  | UIV+2      |   2 |           2 |      0.5 |  -0.3 |   1.3 |   nan   |   nan   |                 0.0 |
| lat     | 24–36  | fern       |   9 |           2 |      3.1 |   2.8 |   5.5 |   nan   |   nan   |                 0.0 |
| lat     | 24–36  | fusioniert |   9 |           2 |      2.0 |  -0.4 |   4.8 |   nan   |   nan   |                11.1 |
| lat     | 36–60  | LIV−1      |   1 |           1 |      0.9 |   0.9 |   0.9 |   nan   |   nan   |                 0.0 |
| lat     | 36–60  | UIV+1      |   3 |           3 |      0.2 |  -0.7 |   0.7 |   nan   |   nan   |                 0.0 |
| lat     | 36–60  | UIV+2      |   3 |           3 |      1.1 |   0.2 |   1.2 |   nan   |   nan   |                 0.0 |
| lat     | 36–60  | fern       |   2 |           1 |     -0.8 |  -1.6 |  -0.1 |   nan   |   nan   |                 0.0 |
| lat     | 36–60  | fusioniert |  10 |           3 |     -0.9 |  -2.3 |   1.6 |   nan   |   nan   |                10.0 |
| slip    | 0–3    | LIV−1      |   1 |           1 |     -6.5 |  -6.5 |  -6.5 |   nan   |   nan   |                 0.0 |
| slip    | 0–3    | UIV+1      |   2 |           2 |      2.8 |   1.3 |   4.3 |   nan   |   nan   |                 0.0 |
| slip    | 0–3    | UIV+2      |   2 |           2 |      2.3 |   2.2 |   2.4 |   nan   |   nan   |                 0.0 |
| slip    | 0–3    | fern       |   5 |           2 |      8.0 |   1.9 |   8.2 |   nan   |   nan   |                 0.0 |
| slip    | 3–6    | UIV+1      |   1 |           1 |     -1.3 |  -1.3 |  -1.3 |   nan   |   nan   |                 0.0 |
| slip    | 3–6    | UIV+2      |   1 |           1 |     -1.0 |  -1.0 |  -1.0 |   nan   |   nan   |                 0.0 |
| slip    | 3–6    | fern       |   3 |           1 |      5.4 |  -0.0 |   5.6 |   nan   |   nan   |                 0.0 |
| slip    | 6–12   | LIV−1      |   1 |           1 |      8.1 |   8.1 |   8.1 |   nan   |   nan   |                 0.0 |
| slip    | 6–12   | UIV+1      |   2 |           2 |     -3.4 |  -4.2 |  -2.5 |   nan   |   nan   |                 0.0 |
| slip    | 6–12   | UIV+2      |   1 |           1 |      1.6 |   1.6 |   1.6 |   nan   |   nan   |                 0.0 |
| slip    | 6–12   | fern       |   2 |           1 |      3.8 |   2.2 |   5.3 |   nan   |   nan   |                 0.0 |
| slip    | 12–24  | LIV−1      |   1 |           1 |     -2.4 |  -2.4 |  -2.4 |   nan   |   nan   |                 0.0 |
| slip    | 12–24  | UIV+1      |   1 |           1 |     -4.9 |  -4.9 |  -4.9 |   nan   |   nan   |                 0.0 |
| slip    | 12–24  | fern       |   2 |           1 |      8.6 |   7.1 |  10.1 |   nan   |   nan   |                 0.0 |
| slip    | 12–24  | fusioniert |   5 |           1 |     10.4 |   6.5 |  11.5 |   nan   |   nan   |                 0.0 |
| slip    | 24–36  | LIV−1      |   1 |           1 |      4.1 |   4.1 |   4.1 |   nan   |   nan   |                 0.0 |
| slip    | 24–36  | UIV+1      |   1 |           1 |      1.4 |   1.4 |   1.4 |   nan   |   nan   |                 0.0 |
| slip    | 24–36  | UIV+2      |   1 |           1 |      1.5 |   1.5 |   1.5 |   nan   |   nan   |                 0.0 |
| slip    | 24–36  | fern       |   2 |           1 |      5.8 |   3.3 |   8.3 |   nan   |   nan   |                 0.0 |
| slip    | 36–60  | LIV−1      |   1 |           1 |     -7.6 |  -7.6 |  -7.6 |   nan   |   nan   |                 0.0 |
| slip    | 36–60  | UIV+1      |   3 |           3 |      1.5 |  -1.2 |   5.3 |   nan   |   nan   |                 0.0 |
| slip    | 36–60  | UIV+2      |   3 |           3 |      0.8 |  -0.9 |   1.7 |   nan   |   nan   |                 0.0 |
| slip    | 36–60  | fern       |   1 |           1 |      4.6 |   4.6 |   4.6 |   nan   |   nan   |                 0.0 |

Ereignisse (erste Visite mit Verlust > MDC95, je Episode × Größe × Segment): 16

| patient                              | patient_id   | ep_date             | group   | plane   | seg    | cls        |   months |   loss |   mdc95 | Langzeitkomplikation   |
|:-------------------------------------|:-------------|:--------------------|:--------|:--------|:-------|:-----------|---------:|-------:|--------:|:-----------------------|
| 3e48a807-cecc-452a-92c2-cd9addf74626 | 0TQN         | 2022-06-07 00:00:00 | Fusion  | ap      | L2/L3  | fusioniert |      1.0 |   11.8 |     7.5 |                        |
| 61744113-1e16-4075-96cc-1d8f74e4c84e | 2Y8F         | 2022-03-06 00:00:00 | Fusion  | ap      | T12/L1 | fusioniert |     17.6 |    9.6 |     7.0 |                        |
| 61744113-1e16-4075-96cc-1d8f74e4c84e | 2Y8F         | 2022-03-06 00:00:00 | Fusion  | ap      | T7/T8  | fusioniert |     17.6 |    9.1 |     6.7 |                        |
| 61744113-1e16-4075-96cc-1d8f74e4c84e | 2Y8F         | 2022-03-06 00:00:00 | Fusion  | lat     | L3/L4  | LIV−1      |     17.6 |   13.3 |     6.7 |                        |
| 61744113-1e16-4075-96cc-1d8f74e4c84e | 2Y8F         | 2022-03-06 00:00:00 | Fusion  | lat     | L4/L5  | fern       |     17.6 |    9.1 |     7.1 |                        |
| 61744113-1e16-4075-96cc-1d8f74e4c84e | 2Y8F         | 2022-03-06 00:00:00 | Fusion  | lat     | L5/S1  | fern       |     17.6 |   16.6 |    13.3 |                        |
| 61744113-1e16-4075-96cc-1d8f74e4c84e | 2Y8F         | 2022-03-06 00:00:00 | Fusion  | lat     | T6/T7  | fusioniert |     17.6 |   12.4 |     7.5 |                        |
| 61744113-1e16-4075-96cc-1d8f74e4c84e | 2Y8F         | 2022-03-06 00:00:00 | Fusion  | lat     | T7/T8  | fusioniert |     17.6 |   16.2 |     7.3 |                        |
| 8c393a9a-01d5-4d74-888d-3bbc5b9ecb9b | 9NNS         | 2020-01-27 00:00:00 | Fusion  | ap      | T1/T2  | fern       |     24.1 |    7.6 |     6.4 |                        |
| 8c393a9a-01d5-4d74-888d-3bbc5b9ecb9b | 9NNS         | 2020-01-27 00:00:00 | Fusion  | ap      | T2/T3  | fern       |     24.1 |   10.5 |     8.2 |                        |
| 8c393a9a-01d5-4d74-888d-3bbc5b9ecb9b | 9NNS         | 2020-01-27 00:00:00 | Fusion  | lat     | L1/L2  | fusioniert |     23.9 |    7.1 |     5.9 |                        |
| 8c393a9a-01d5-4d74-888d-3bbc5b9ecb9b | 9NNS         | 2020-01-27 00:00:00 | Fusion  | lat     | L2/L3  | fusioniert |      2.8 |    7.1 |     5.8 |                        |
| 97fcafbc-58a4-4934-bd3e-14ded03350a2 | 3R9Q         | 2020-03-30 00:00:00 | Fusion  | ap      | L2/L3  | UIV+2      |      0.2 |    9.6 |     7.5 |                        |
| ae5b1a5d-3fb2-4608-bd0c-bbc9c102ba1c | 27NG         | 2024-12-17 00:00:00 | Fusion  | lat     | T3/T4  | fern       |      1.1 |   11.1 |    10.1 |                        |
| aff2d06f-99c2-4c10-99e5-e74f02ddef37 | 4AEM         | 2024-04-16 00:00:00 | Fusion  | lat     | L4/L5  | fusioniert |      0.9 |    9.4 |     7.1 |                        |
| f48a523c-c2cd-47ca-8717-be3d95742779 | 6R96         | 2021-06-28 00:00:00 | Fusion  | lat     | L3/L4  | fusioniert |     51.2 |    6.9 |     6.7 |                        |

## Grenzen

Kleine Fallzahl (Machbarkeitsnachweis); Indikations-Confounding (konservativ = Spontanverlauf, keine Kontrollgruppe);
MDC aus Rater-Streuung unterschätzt den Messfehler zwischen Visiten; Slip-Modell schrumpft große Slips; kein Slip auf
WS-lat; Fusionsstrecke ±1 an UIV/LIV unsicher (siehe Konzept §8).
