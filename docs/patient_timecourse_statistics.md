# Patient Clinical Timecourse statistics
Ziel ist die statistische Analyse und visuelle Darstellen des klinischen Timecourses von Patienten.

## Code
 - streamlit App
 - Zukünftig: AnalysisCard in Vue-Frontend (nicht Teil der aktuellen Implementierung)

## Datenbasis
 - Abrufen der Daten von der API (Token Eingabe erforderlich) und cachen
 - relevante Datenklasse: ClinicalEntryData in /Users/peterbernstein/Django/xray_api/api/models.py. Die dort enthaltenen Daten stellen klinische Ereignisse (z.B. OP) für einen Patienten mit dazugehörigem Datum (referring_date) dar
 - Das Feld data von ClinicalEntryData enthält die Detailparameter des klinischen Ereignisses. Die darin enthaltenen Properties sind vom Typ ClinicalEntryVarData (/Users/peterbernstein/Vue/xray-annotator/src/helpers/clinicalVars.ts), sie werden nachfolgend als ClinicalProps bezeichnet
 - das Feld index_date der Klasse Patient in /Users/peterbernstein/Django/xray_api/api/models.py bezeichnet das Datum des ersten klinischen Ereignisses

### ClinicalProps
Nachfolgend werden die Merkmale der klinischen Ereignisse aufgelistet

 - spinal_symptoms: Array<string>, fasst klinische Symptome zusammen wie z.B. L5-Syndrom, Fußheberparaese, Claudicatio spinalis,... 
 - spinal_instability_features: Array<string>, hierunter zählen Spondylolisthesen, Vakuumphänomene, Subluxationen
 - ambulatory_status_before: string, welche Gehstrecke der Patient vor dem klinischen Ereignis zurücklegen konnte (z.B. 100m, 500m, 1km, 5km, 10km, >10km), gehfähig, Rollstuhl,...
 - opioids_before: boolean, wurden vor der OP bereits Opiate eingenommen
 - mri_characteristics: Array<string>, beschreibt die wesentlichen MRT-Features der spinalen Stenose, z.B. Neuroforamenstenose, Recessusstenose (in Schizas Klassifikation), Facettengelenksarthrose, Ligamentum-flavum-Hypertrophie, Diskushernie,...
 - scoliosis_type: string, idiopathisch oder degenerativ
 - berjano_classification: string, beschreibt die Lagebeziehung des symptomatischen Segments zur Skoliosekrümmung
 - therapy: string, konservativ, decompression, fusion,...
 - therapy_specification: Array<string>, welche Wirbelsegmente sind von der Therapie betroffen, z.B. L4/5, L5/S1,...
 - fused_segments: number, Anzahl der fusionierten Segmente
 - fused_to_pelvis: boolean, ist die Fusion bis auf das Becken geführt worden
 - complication: Array<string>, welche Komplikationen sind im Rahmen des Aufenthaltes oder danach aufgetreten
 - ambulatory_status_after: string, wie ist der Patient entlassen worden
 - opioids_after: boolean, Opiate nach der OP erforderlich

## Analysen
 - Beziehung zwischen der Therapie des ersten und letzten klinischen Ereignisses als Sankey-Diagramm darstellen
 - Beziehung zwischen der Therapie und den Komplikationen als Sankey-Diagramm darstellen
 - Beziehung zwischen spinal_symptoms, mri-charateristics and therapy_specification darstellen - geeignete Visualisierung dazu herausfinden