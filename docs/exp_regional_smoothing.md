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