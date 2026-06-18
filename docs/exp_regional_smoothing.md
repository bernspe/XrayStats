# Regional Spline Smoothing Effect
## Ziel
 - Outlier in den Nodes sowie eine Verschiebung der Nodes an den Enden der Spline führen zu einer gravierenden Änderung der COBB-Winkel
 - Untersuchen, ob die Anwendung eines regionalen Spline-Smoothing-Algorithmus die Stabilität der COBB-Winkel verbessert, insbesondere in Bezug auf die Endknoten der Spline

## Grundlagen
    - [exp_Annotation_correctness.md](exp_Annotation_correctness.md)
    - [annotation_correctness.ipynb](../annotation_correctness.ipynb) und importierte Funktionen

## Daten
 - es können fertige Datensätze aus data/df4d_spine_measurements.pkl geladen werden
 - das Smoothing soll nur auf cobbType==Spline und auf die inflection Datensätze angewendet werden
 - cobbType==Cobb_regional dient als Referenz

## Hypothese
 - Smoothing führt zu einer Reduktion der Standardabweichung der Cobb-Winkel über die Rater hinweg, insbesondere bei den Endknoten der Spline
 - Die Cobb-Winkel nach Smoothing korrelieren stärker mit den Cobb_regional-Winkeln als die ungesmoothten Spline-Winkel
 - Smoothing erhöht die ICC der Cobb-Winkel über die Rater hinweg

## Methodik: Dynamic Smoothing-Window

### Hintergrund

Die Inflection-basierte Cobb-Winkel-Berechnung (`getInflectionCobbAngles`) identifiziert die Wendepunkte (Extrema) der Tangentialwinkelkurve entlang der Spline-Nodes, um die Messabschnitte automatisch zu bestimmen. Das Verfahren nutzt `find_tangent_extrema_indices`, das lokale Maxima und Minima der Node-Tangentialwinkel als Apex-Vertebrae liefert.

**Problem:** Bei verrauschten Annotationen (z.B. leicht verschobene Nodes) liefert `find_tangent_extrema_indices` mit `smoothing_window=1` zu viele falsche lokale Extrema. Beispiel: Bild 376 (C-Kurve) produzierte 4 Segmente statt 1–2.

### Algorithmus

Das **Dynamic Smoothing-Window**-Verfahren (`find_optimal_smoothing_window`) bestimmt automatisch das kleinste Glättungsfenster, das die Anzahl der Extrema auf ein physiologisch plausibles Maß reduziert:

```python
def find_optimal_smoothing_window(tangent_angles, target_max_extrema=2):
    for w in range(1, len(tangent_angles)):
        if len(find_tangent_extrema_indices(tangent_angles, smoothing_window=w)) <= target_max_extrema:
            return w
    return max(1, len(tangent_angles) // 3)
```

1. **Sweep:** Fensterbreite `w` wird von 1 (keine Glättung) schrittweise erhöht.
2. **Stopp-Kriterium:** Sobald `find_tangent_extrema_indices` mit Fenster `w` höchstens `target_max_extrema` Extrema liefert, wird `w` zurückgegeben.
3. **Fallback:** Falls kein `w` das Kriterium erfüllt, wird `len(tangent_angles) // 3` verwendet.

### Parameter

| Parameter | Default | Bedeutung |
|---|---|---|
| `target_max_extrema` | `2` | Maximale Anzahl Extrema; 2 deckt C- und S-Kurven ab; 3 für Triple-Kurven |

### Adaptive Eigenschaft

Das Verfahren passt sich automatisch an den Geräuschpegel jeder Messung an:
- **Saubere S-Kurve** (z.B. Bild 975): `w=1` reicht aus → keine unnötige Glättung
- **Verrauschte S-Kurve** (z.B. Bild 974): `w=4` reduziert 3→2 Extrema
- **Verrauschte C-Kurve** (z.B. Bild 376): `w=8–9` reduziert 4→2 Extrema

### Integration in die Pipeline

Das Dynamic Smoothing-Window ist in `annotation_analysis.py` in `getInflectionCobbAngles` und `get_inflection_overlay_data` integriert und wird **vor** der eigentlichen Gauss-Glättung angewendet:

```
build_angles_df(smooth_sigma=σ)
  └─ getInflectionCobbAngles(xmd, smooth_sigma=σ)
       ├─ smooth_nodes(nodes, sigma=σ)           # Gauss-Glättung der Nodes
       ├─ find_optimal_smoothing_window(...)     # Dynamic Window auf Tangentialwinkeln
       └─ find_tangent_extrema_indices(w=opt)   # Extrema mit adaptivem Fenster
```

Der Sigma-Sweep in `regional_smoothing.ipynb` nutzt automatisch das Dynamic Smoothing-Window, da es in der Aufrufkette bereits verankert ist — das Notebook kann ohne Anpassungen neu ausgeführt werden.