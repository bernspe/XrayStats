# icfx_app — ICFx × xray_api Analyse-App

Streamlit-App zur wissenschaftlichen Auswertung von SF-36-, WHODAS- und ICF-Patientendaten
aus der ICFx-Plattform, verknüpft mit klinischen Daten aus der xray_api.

**Datei:** `icfx_app.py`  
**Start:**
```bash
cd /Users/peterbernstein/Python/XrayStats
/Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run icfx_app.py
```

---

## Authentifizierung

Der Zugang erfolgt über ein **Bearer Token** aus dem xray-annotator
(Account-Info → Token kopieren). Das ICFx-Institutionstoken wird automatisch
über `GET https://xray-api.renecol.org/api/icfxtoken/` abgeholt — der Nutzer
muss sich nur einmal anmelden.

### Auth-Flow (2-stufig)

```
Bearer Token (xray_api)
  → GET /api/icfxtoken/
    → ICFx-Token (client_credentials, institution-gebunden)
      → GET {ICFX_API}/icfdata/
```

---

## Datenquellen und Join-Strategie

### Schritt 1 — xray_api: Patientenliste

`GET https://xray-api.renecol.org/api/patients/`

Liefert alle Patienten der Institution. Gefiltert werden nur Patienten mit
gesetztem `hasICFxEntry`-Feld (= ICFx-UUID, entspricht
`clinicalbasedata.data.icfx_uuid`).

Extrahierte Felder pro Patient:

| Feld | Quelle im API-Response |
|------|------------------------|
| `patient_uuid` | `uuid` |
| `patient_id` | `patient_id` |
| `icfx_uuid` | `hasICFxEntry` |
| `birthdate` | `clinicalbasedata.data.birthdate` |
| `sex` | `clinicalbasedata.data.sex` |
| `diagnoses` | `clinicalbasedata.data.diagnoses` |
| `height`, `weight` | `clinicalbasedata.data.*` |
| `therapy`, `fused_segments`, `scoliosis_type` | letzter `clinicalentrydata`-Eintrag |

### Schritt 2 — ICFx API: Datensätze

`GET https://static.94.87.140.128.clients.your-server.de/icfxapi/icfdata/`

Liefert alle ICFx-Datensätze der Institution. Filter:
- `owner == creator` → nur Patienteneingaben (keine Admin-Einträge)
- `owner in icfx_uuids` → nur Patienten, die auch in der xray_api bekannt sind

### Join Key

```
xray_api:  Patient.hasICFxEntry  (= clinicalbasedata.data.icfx_uuid)
ICFx API:  icfdata.owner
```

### Mehrfacheinträge pro Patient

Ein Patient kann mehrere ICFx-Datensätze haben (z.B. Verlaufsmessungen).
Deduplizierungsstrategie wählbar in der Sidebar:

- **Letzter Eintrag** (`groupby("owner").last()` nach `created`-Datum) —
  Standard für Cross-Sectional-Analysen
- **Vollständigster Eintrag** — Zeile mit den meisten ausgefüllten
  Instrument-Spalten (`sf36`, `whodas`, `icf`, `env`)

---

## Instrumente und Auswertungslogik

### SF-36

Normierungsquelle: `http://icfx-ressources.renecol.org/sf36_de.json`  
Kategorie-Definitionen: `sf36_categories_de.json`

**Normierung:** Rohwert → 0–100-Skala. Richtung je Item aus dem
Definitionsobjekt (`default`-Feld):
- `default == n_answers - 1` → aufsteigende Skala: `score = 100/n * (raw + 1)`
- sonst → absteigende Skala: `score = 100/n * (n - raw)`

**Kategorien** (Abbreviation → deutscher Titel aus Definitionsdatei):
PF, RP, BP, GH, VT, SF, RE, MH

**Normvergleich:** Normdaten aus  
Ellert U & Kurth BM (2013): *Gesundheitsbezogene Lebensqualität bei
Erwachsenen in Deutschland.* Bundesgesundheitsblatt 56(5–6):643–649. DEGS1.

Datei: `/Users/peterbernstein/Vue/icfx-app2/src/assets/sf36_normdata.json`

Normwerte sind als 95%-Konfidenzintervall-Grenzen (`{abbr}_min`, `{abbr}_max`)
gespeichert, aufgeteilt nach Altersgruppe (`18-29` … `70-79`) und Geschlecht
(`f`, `m`, `t`).

**Matching-Logik** (`norm_comparison()`):
1. Alter aus `birthdate` berechnen (ganzzählige Jahre)
2. Altersgruppe zuweisen (18–79; außerhalb: kein Match)
3. Geschlecht mappen: `weiblich/female/w/f → f`, `männlich/male/m → m`, sonst `t`
4. Normreferenz = Mittelwert der per-Patient gematchten Werte

Visualisierung im SF-36-Tab:
- **Vergleichschart**: Kohortenmittelwert ± SD vs. gematchtes Norm-95%-KI
- **Differenz-Chart**: Δ Patient − Norm je Subskala (blau = besser, rot = schlechter)
- **Boxplot** mit Norm-Mittelwert-Markern

### WHODAS-12

Normierungsquelle: `http://icfx-ressources.renecol.org/whodas12_de.json`

**Normierung:** Rohwert (1 = kein Problem … 5 = volles Problem) → invertiert:
`score = (5 - raw) * 20` → 0–100, höher = besser

Gesamt-Score: Mittelwert aller Domänen.

### ICF

Nur Beeinträchtigungs-Codes (`d`-Codes, keine Umweltfaktoren `e`).  
Nur selektierte Items (`selected == 1`).

Ausgabe: Prävalenz (Anteil Patienten mit Code-Eintrag) und mittlere Ausprägung
(1–4) der häufigsten Codes.

---

## Tabs

| Tab | Inhalt |
|-----|--------|
| **SF36** | Normvergleich (alters-/geschlechtsgematcht), Differenz-Chart, Boxplot |
| **WHODAS** | Domänen-Balkendiagramm, Korrelationsmatrix |
| **ICF** | Prävalenz und Ausprägung der d-Codes (Top 30 / Top 20) |
| **Kombiniert** | SF36 × WHODAS Korrelationsheatmap, Merge-Tabelle mit xray_api |
| **Rohdaten** | xray_api Patienten-Tabelle, ICFx-Rohzeilen nach Dedup |

---

## Externe Ressourcen

| Ressource | URL |
|-----------|-----|
| xray_api | `https://xray-api.renecol.org/` |
| ICFx API | `https://static.94.87.140.128.clients.your-server.de/icfxapi/` |
| ICFx Definitionen | `http://icfx-ressources.renecol.org/` |
| SF-36 Normdaten | `/Users/peterbernstein/Vue/icfx-app2/src/assets/sf36_normdata.json` |
| Venv | `/Users/peterbernstein/Python/ICFXStats/venv/bin/` |

---

## TODOs / Geplante Erweiterungen

> Alle Punkte setzen eine ausreichende Fallzahl voraus.

### Filterung nach klinischem Verlauf

Patienten haben in der xray_api einen `index_date` sowie mehrere
`clinicalentrydata`-Einträge mit `referring_date`. Geplant:
- Zeitraum-Filter (Präoperativ / Postoperativ relativ zum `index_date`)
- Verlaufsgruppen: z.B. Patienten mit ≥ 2 ICFx-Einreichungen über Zeit →
  Längsschnittvergleich der SF-36/WHODAS-Scores

### Filterung nach Therapie / Therapien

`clinicalentrydata.data.therapy` enthält den Therapietyp (konservativ /
operativ / etc.) sowie `therapy_specification`, `fused_segments`,
`fused_to_pelvis`. Geplant:
- Subgruppenvergleich SF-36/WHODAS nach Therapiekategorie
- Bereits ansatzweise im „Kombiniert"-Tab umgesetzt (SF-36 nach Therapie),
  ausbaubar mit Signifikanztests (Mann-Whitney U, Kruskal-Wallis)

### Multivariate Analyse

Sobald n ≳ 30–40 auswertbare Fälle vorliegen:
- Korrelation SF-36/WHODAS mit klinischen Variablen (Cobb-Winkel, Diagnose,
  Anzahl fusionierter Segmente, Alter, BMI)
- Regressionsmodell: Prädikatoren für SF-36-Subscores
- Cluster-Analyse der Patientenprofile (ICF + WHODAS + SF-36 gemeinsam)
- Prä-/Post-Vergleich bei operierten Patienten (paired Wilcoxon)

---

*Erstellt: 2026-07-30*
