# annotation_timing_app — Messdauer-Analyse je Annotator

Dieses Dokument beschreibt `annotation_timing_app.py` und bereitet die Migration in den
**Analysis Tab** des xray-annotators vor.

Start (Prototyp):
```
cd /Users/peterbernstein/Python/XrayStats
streamlit run annotation_timing_app.py
```

---

## 1. Zweck

Die App misst, wie lange Annotatoren für eine einzelne Messung benötigen, und ermöglicht
den Vergleich zwischen Annotatoren sowie die Analyse des zeitlichen Lernverlaufs.

**Kernmetrik:** `Δt = MeasureData.last_modified − MeasureData.created_at`

Messungen mit `Δt ≤ 0` werden verworfen (sofortige Speicherungen ohne inhaltliche Arbeit).

---

## 2. Datenbasis

| Quelle | Inhalt |
|---|---|
| `GET /api/imagedata/?detailed=True&complete=True` | Liefert pro Bild alle `measure_dataset`-Einträge inkl. `created_at`, `last_modified`, `user` (first/last name), `img_type` |
| `data/annotation_timing_cache.json` | Lokaler Disk-Cache des vollständigen API-Response (gleiches Muster wie `image_groups_cache.json`) |

Der Cache wird in der Sidebar über Token-Eingabe + Button befüllt und beim nächsten
Start automatisch geladen. Token ist nicht persistent gespeichert.

---

## 3. Filter

| Widget | Wirkt auf |
|---|---|
| Annotatoren (Multiselect) | alle Auswertungen |
| Bildtypen / img_type (Multiselect) | alle Auswertungen |
| Obere Perzentile (Slider, default 99%) | Histogramm-Anzeige **und** CI-Summenberechnung |

Der Perzentil-Cutoff entfernt Extremwerte aus der Visualisierung, **ohne** Median und
CI-Grenzen zu verschieben (diese werden immer auf dem ungefilterten Pool berechnet).

---

## 4. Statistiktabelle

Multi-Index `(Annotator, img_type)`. Spalten:

| Spalte | Berechnung |
|---|---|
| N | Anzahl Messungen im gefilterten Pool |
| Median | Median von `duration_s` |
| 95% CI low / high | Bootstrap-Perzentile (2000 Resamples, Seed 42) des Medians |
| **Summe im CI** | Summe aller `duration_s` mit `lo ≤ x ≤ hi` **und** `x ≤ cap` |
| Gesamt-Summe | Summe aller `duration_s` ohne Cutoff |

Zusätzlich je Annotator eine **„∑ Annotator"**-Zeile (über alle img_types) und am Ende
eine **TOTAL**-Zeile über alle gefilterten Annotatoren.

**Begründung für die CI-Summe:** Die Summe soll die realistische Gesamtarbeitszeit im
Normalbetrieb widerspiegeln — Extremausreißer (sehr lange Unterbrechungen, offene
Browser-Tabs) würden die Summe verzerren. Der CI-Band-Schnitt filtert diese heraus ohne
einen arbiträren festen Cutoff.

---

## 5. Zeitverlauf-Diagramm

Alle Annotatoren in einem Chart. X-Achse: **Tage seit erster Messung des jeweiligen
Annotators** (t₀ je Annotator, berechnet aus dem ungecappten Pool). Y-Achse: Messdauer
in Minuten.

- **Farbe** = Annotator
- **Marker-Form** = img_type
- **Trendlinie** = 7-Punkte-Rolling-Median je Annotator (über alle img_types), sichtbar
  ab ≥ 7 Messungen
- Legende zweispaltig: Annotator-Farben + img_type-Marker-Shapes als Dummy-Einträge

Die relative x-Achse (statt absoluter Datumsachse) macht Lernkurven direkt vergleichbar,
unabhängig vom Startzeitpunkt des Annotators.

---

## 6. Histogramm

Ein Panel je Annotator. Gestapelte Balken je img_type (gemeinsame Bin-Edges), Median-Linie
und 95%-CI-Schattierung in Rot. CI und Median werden auf dem **vollen** (ungecappten)
Datensatz des Annotators berechnet; nur die Histogramm-Daten selbst sind auf `cap`
begrenzt.

---

## 7. Migration in den Analysis Tab (xray-annotator)

Der Analysis Tab verwendet eine Registry-basierte Architektur
(`src/helpers/analysisRegistry.ts`) mit precomputed Snapshots (`/api/analysissnapshot/`).
Das Timing-Feature passt direkt in dieses Schema.

### 7.1 Backend: `api/helpers/analysis_data.py`

Neue Builder-Funktion hinzufügen:

```python
def build_annotation_timing_for_institution(institution) -> list[dict]:
    """
    Returns one row per (user, img_type) with timing statistics.
    Schema: [{
        user: str, img_type: str, n: int,
        median_s: float, ci_low_s: float, ci_high_s: float,
        sum_in_ci_s: float, total_sum_s: float,
        timeseries: [{ day: float, duration_s: float }]   # relative to user's t0
    }]
    """
    from .annotation_analysis import bootstrap_median_ci  # oder inline
    import numpy as np

    qs = MeasureData.objects.filter(
        user__institution=institution,
    ).select_related('user').prefetch_related('imagedata_set')

    rows = []
    for md in qs:
        imgs = md.imagedata_set.all()
        img_type = imgs[0].img_type if imgs else None
        delta = (md.last_modified - md.created_at).total_seconds()
        if delta <= 0 or not md.user:
            continue
        name = f"{md.user.first_name} {md.user.last_name}".strip() or md.user.username
        rows.append({
            'user': name, 'img_type': img_type or 'Unknown',
            'created_at': md.created_at, 'duration_s': delta,
        })

    if not rows:
        return []

    df = pd.DataFrame(rows)
    result = []
    for (user, img_type), grp in df.groupby(['user', 'img_type']):
        x = grp['duration_s'].values
        # Bootstrap CI (2000 resamples)
        rng = np.random.default_rng(42)
        medians = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(2000)]
        lo, hi = float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))
        ci_mask = (x >= lo) & (x <= hi)
        # timeseries relative to user's global t0
        t0 = df[df['user'] == user]['created_at'].min()
        ts = [
            {'day': round((r.created_at - t0).total_seconds() / 86400, 2),
             'duration_s': round(r.duration_s, 1)}
            for _, r in grp.sort_values('created_at').iterrows()
        ]
        result.append({
            'user': user, 'img_type': img_type,
            'n': len(x),
            'median_s':    round(float(np.median(x)), 1),
            'ci_low_s':    round(lo, 1),
            'ci_high_s':   round(hi, 1),
            'sum_in_ci_s': round(float(x[ci_mask].sum()), 1),
            'total_sum_s': round(float(x.sum()), 1),
            'timeseries':  ts,
        })
    return result
```

### 7.2 Backend: `api/models.py` — SNAPSHOT_TYPES

```python
SNAPSHOT_TYPES = [
    'angles', 'icc', 'icc_regional', 'icc_segmental',
    'intermodal_lu', 'cumulative_annotations', 'xray_workup',
    'annotation_timing',   # ← neu
]
```

### 7.3 Backend: `api/tasks.py`

```python
from api.helpers.analysis_data import (
    ...
    build_annotation_timing_for_institution,  # ← neu
)

builders = {
    ...
    'annotation_timing': build_annotation_timing_for_institution,   # ← neu
}
```

### 7.4 Frontend: TypeScript-Typen (`src/helpers/analysisUtils.ts`)

```typescript
export interface TimingRow {
  user: string
  img_type: string
  n: number
  median_s: number
  ci_low_s: number
  ci_high_s: number
  sum_in_ci_s: number
  total_sum_s: number
  timeseries: { day: number; duration_s: number }[]
}
```

### 7.5 Frontend: Store (`src/stores/analysisstore.ts`)

```typescript
// state
annotationTiming: null as TimingRow[] | null,
isTimingFetching: false,
timingFetchError: '',
timingMeta: null as SnapshotMeta | null,

// action
async ensureAnnotationTiming() {
  if (this.annotationTiming) return
  this.isTimingFetching = true
  try {
    const snap = await fetchSnapshot<TimingRow[]>('annotation_timing')
    this.annotationTiming = snap.data
    this.timingMeta = { computed_at: snap.computed_at, needs_recompute: snap.needs_recompute }
  } catch {
    this.timingFetchError = 'Failed to load timing data.'
  } finally {
    this.isTimingFetching = false
  }
}
```

### 7.6 Frontend: Registry (`src/helpers/analysisRegistry.ts`)

```typescript
const AnnotationTiming = defineAsyncComponent(
  () => import('../components/analysis/AnnotationTiming.vue')
)

// in analysisRegistry array:
{
  id:        'annotation-timing',
  label:     'Annotation Duration',
  category:  'Study Performance',
  component: AnnotationTiming,
},
```

### 7.7 Frontend: Komponente `AnnotationTiming.vue`

Strukturell analog zu `CumulativeAnnotations.vue`:

- `onMounted` → `store.ensureAnnotationTiming()`
- Snapshot-Alter-Badge (`formatSnapshotAge(store.timingMeta)`) wie in den anderen Komponenten
- **Tabelle** (HTML-Table oder MDB-Table): Zeilen = `(user, img_type)` + Subtotals +
  TOTAL, Spalten = N · Median · CI · Summe im CI · Gesamt. Formatierung: `s → "Xm YYs"`,
  `s > 3600 → "Xh YYm ZZs"`.
- **Zeitverlauf-Chart** (Chart.js Scatter): ein Dataset je `(user, img_type)`,
  Farbkodierung nach User (gleiche `colorPalette` wie in den anderen Komponenten),
  Marker-Shape nach img_type. X-Achse = Tage (number), Y-Achse = Minuten.
  Trendlinie: Rolling-Median clientseitig über das `timeseries`-Array berechnen
  (einfaches Sliding-Window, kein Bibliotheks-Import nötig).
- **Histogramm** pro User: Chart.js Bar (stacked), ein Dataset je img_type;
  Annotation-Plugin für Median-Linie und CI-Band (bereits in anderen Analyse-Charts
  verwendet — prüfen, ob `chartjs-plugin-annotation` schon installiert ist).
- Filter (Annotator, img_type) als `<select multiple>` oder Checkboxen — clientseitig
  über `computed` auf `store.annotationTiming`, kein erneuter API-Call.
- Perzentil-Cutoff als Range-Input; CI-Summen-Neuberechnung clientseitig.

---

## 8. Offene Punkte

- **img_type-Zuordnung im Backend**: `MeasureData` hat keine direkte `img_type`-Spalte —
  der Lookup über `imagedata_set` (M2M-Relation) kann teuer sein. Besser:
  `ImageData.objects.filter(measure_dataset=md).values_list('img_type', flat=True).first()`
  oder denormalisiert in einem `annotated()`-Queryset. Bei großen Datenmengen Prefetch
  oder einen dedizierten Index prüfen.
- **Negative Δt**: Im Prototyp verworfen. Im Backend sollte geprüft werden, ob
  `auto_now=True` auf `last_modified` in seltenen Fällen eine frühere Zeit als
  `created_at` liefern kann (Clock-Skew bei Replicas). Aktuell kein bekanntes Problem.
- **Snapshot-Aktualität**: Das Timing-Snapshot wird mit den anderen Snapshots gemeinsam
  durch `recompute_analysis_for_institution` aktualisiert (täglich via Cronjob + manuell
  per Staff-POST). Bei hoher Annotationsfrequenz kann das Bild um bis zu 24h veraltet
  sein — für eine Qualitäts-/Planungs-Metrik akzeptabel, bei Bedarf separates
  Recompute-Intervall.
- **Perzentil-Cutoff und CI-Summe clientseitig**: Der Streamlit-Prototyp berechnet diese
  dynamisch. Im Vue-Frontend braucht die Komponente die rohen `timeseries`-Daten (im
  Snapshot enthalten), um den Cutoff-Slider clientseitig ohne Server-Roundtrip anzubieten.
  Alternative: Cutoff serverseitig auf einen Standardwert (z. B. p99) fixieren und
  vorberechnen — einfacher, aber weniger flexibel.
- **Chart.js Scatter + Rolling-Median**: Chart.js unterstützt keinen nativen
  Rolling-Median. Entweder clientseitig per 7-Punkt-Sliding-Window vorberechnen (wenige
  Zeilen JS) oder `chartjs-plugin-trendline` einbinden — ersteres bevorzugen
  (keine neue Abhängigkeit).
