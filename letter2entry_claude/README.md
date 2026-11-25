# letter2entry mit Claude – Konzept

Claude als unabhängiger zweiter Extraktor für die Arztbriefe der Patienten, die noch
**nicht validiert** sind. Wo Claude und der aktuelle Stand (SCADS-Vorschlag plus
Reviewer-Entscheidung) auseinanderliegen, entsteht eine Prüfliste für die Validierung.
Lokal, nur lesend gegen die xray-api. Bereits validierte Fälle werden nicht an Claude
geschickt und verursachen keine Kosten.

## Ausgangslage (Stand xray_api / xray-annotator)

| Was | Wo |
|---|---|
| Validierungsstatus | `Patient.progresstag[].status` ∈ `START`, `FINISH`, `VALID`, `INCORRECT` (Filter „Validated“ im Frontend = irgendein Tag `VALID`) |
| SCADS-Aufruf, Prompt-Bau, Parser | `Django/xray_api/api/helpers/letter2entry.py` (`system_prompt()`, `parse_annotations()` sind reine Funktionen) |
| Prompt-Kopf | `letter2entry_prompt.txt` |
| Felder + Feld-Prompts | `GET /api/clinicalfields/` (`key`, `kind`, `prompt`, `llm_extract`) |
| Brief + Annotationen | `GET /api/clinicalentrydata/` → `letter = {sections, annotations[]}` |
| Aktuelle Feldwerte | `ClinicalEntryData.data`, Diagnosen in `ClinicalBaseData.data.diagnoses` |
| Vokabular | `GET /api/clinicalentrydata/getentryoptions/` + Diagnosen aus `/api/clinicalbasedata/` |

Annotationen: `{section, quote, field, value, status, original?}` mit
`status` ∈ `predicted` (offen), `accepted` (ggf. mit `original` = SCADS-Wert vor
Korrektur), `declined`, `user`.

## Auswahl der Briefe

Alle `ClinicalEntryData` mit `letter.sections`, deren Patient **kein** `VALID`-Tag hat.
Dabei werden drei Stände unterschieden, weil sie verschieden viel Vergleichsmaterial haben:

| Brief-Stand | Vergleich Claude gegen |
|---|---|
| `imported` (nie extrahiert) | nur aktuelle `data`-Werte (meist leer) – Claude ist der erste Vorschlag |
| `unreviewed` (SCADS-Vorschläge offen) | SCADS-Vorschläge |
| `reviewed`, nicht validiert | Reviewer-Ergebnis (`accepted` + `user`) **und** was der Reviewer verworfen hat (`declined`) |

Gerade die letzte Gruppe ist der eigentliche Hebel: Weil der Reviewer nicht genau
gearbeitet hat, zeigen die Abweichungen zwischen Claude und dem Review-Ergebnis, wo man
hinschauen muss.

## Ablauf

```
xray-api (GET, Bearer-Token XRAY_TOKEN)
  ├─ /patients/  → Patienten ohne VALID
  ├─ /clinicalentrydata/, /clinicalbasedata/, /clinicalfields/, getentryoptions
        │
        ▼  data/letters.json (Cache, lokal, gitignored)
run_claude.py
  ├─ nur Briefe, für die noch kein Claude-Ergebnis im Cache liegt (keine Doppelkosten)
  ├─ Prompt = letter2entry.system_prompt(fields, options)   ← identisch zu SCADS
  ├─ Claude, Structured Output (field ∈ Feld-Keys)
  ├─ letter2entry.parse_annotations()                        ← Zitat muss im Abschnitt stehen
  └─ data/claude_results.json  {entry_uuid: [annotations]}
        │
        ▼
review_app.py (Streamlit)
  ├─ Liste: Briefe nach Zahl der Abweichungen sortiert (größte zuerst)
  ├─ Brief-Ansicht: Abschnitte mit markierten Zitaten, je Feld drei Spalten
  │    aktueller Stand | SCADS (inkl. declined/original) | Claude
  │    Abweichungen farbig: nur Claude · nur aktuell · anderer Wert
  ├─ je Abweichung entscheiden: Claude richtig / aktueller Stand richtig / beide falsch
  └─ data/decisions.csv
```

`letter2entry.py` wird per `importlib` aus dem Django-Repo geladen (Pfad per
`LETTER2ENTRY_PY`), nicht kopiert. So sind Prompt und Zitatfilter dieselben wie in der
Produktion.

### Prompt und Ausgabeformat

Beides wird zur Laufzeit gebaut, nie fest im Code hinterlegt:

- **Felder und Feld-Prompts** kommen aus `GET /api/clinicalfields/`. Die API liefert die
  *effektive* Liste der Institution des Token-Inhabers: ihre eigenen Felder, sonst die
  Standardliste (`institution=None`, Seed in Migration `0039_seed_clinical_entry_fields`).
  Es werden nur Felder mit `llm_extract` und `is_active` genutzt, dazu der feste Block
  `base.diagnoses`.
- **Prompt** = `letter2entry.system_prompt(fields, options)`: Kopf aus
  `letter2entry_prompt.txt`, dann je Feld ein `## key`-Block mit `prompt`, dann das
  Vokabular.
- **Ausgabe** wie in der Produktion: `[{section, quote, field, value}]`, `value` als
  String. Bei Claude erzwingt ein JSON-Schema dieses Format, das aus der Feldliste
  abgeleitet wird: `field` als Enum der Keys, `section` als Enum der Überschriften
  des Briefs. Die Regel für Bool- und Zahlenwerte („true“/„false“, Zahl) folgt aus
  `kind`. Im Prompt-Kopf stehen diese Feldnamen dagegen fest (`opioids_before`, …) –
  neue Bool-Felder einer Institution wären dort nicht abgedeckt.

Claude extrahiert **unabhängig**, ohne die vorhandenen Annotationen zu sehen. Würde man
sie mitschicken („prüfe diese Vorschläge“), übernähme das Modell die Fehler des Reviewers
eher, als sie zu finden.

## Was die Entscheidungen nebenbei liefern

Jede Entscheidung in der App ist Validierungsarbeit, die sowieso anfällt, und zugleich ein
Messpunkt: Über `decisions.csv` ergibt sich je Feld, wie oft Claude bzw. SCADS/Reviewer
richtig lag. Die Qualität wird also auf den Arbeitsfällen gemessen, nicht mit einem
eigenen Lauf auf validierten Fällen.

Einschränkung: Gemessen wird nur, wo sich beide unterscheiden. Fehler, die beide gleich
machen, sieht man so nicht. Bei Feldern, wo das auffällt, hilft eine Stichprobe von Hand.

## Zurückschreiben

Zunächst nicht – die App schreibt nur `decisions.csv`. Wenn die Ergebnisse überzeugen, ist
der nächste Schritt ein PATCH auf `ClinicalEntryData.data`/`letter.annotations`
(`status = 'user'` für übernommene Claude-Funde) oder der Tausch von `ask_llm()` in
`xray_api` gegen den Anthropic-Client.

## Modell und Kosten

- Standard `claude-opus-5-5` ($4 / $20 pro 1M Tokens): ~10k Tokens Eingabe, ~2k Ausgabe
  ≈ 8 ct pro Brief. Der für alle Briefe gleiche System-Prompt wird gecacht.
- Jeder Brief wird genau einmal geschickt; das Ergebnis liegt danach im Cache.
- Vorher ein Testlauf mit ~10 Briefen und Kostenanzeige aus `response.usage`.
- Batch-API halbiert die Kosten, die Daten liegen dort aber bis zu 29 Tage und sind nicht
  ZDR-fähig (siehe unten). Ohne ZDR-Vereinbarung macht das keinen Unterschied.

## Datenschutz (Anthropic, Stand der Seiten 2026-11-20)

- **Kein Training:** Eingaben und Ausgaben der kommerziellen API werden standardmäßig
  nicht zum Training verwendet, in den Commercial Terms ist das vertraglich
  ausgeschlossen. Einen Schalter dafür gibt es nicht, weil Training gar nicht erst
  vorgesehen ist. Ausnahme ist aktiv eingereichtes Feedback (z. B. Daumen hoch/runter in
  Anthropic-Oberflächen). In dieser App gibt es so etwas nicht.
  [privacy.claude.com – Training](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training),
  [Commercial Terms](https://www.anthropic.com/legal/commercial-terms)
- **Speicherdauer:** Eingaben und Ausgaben werden innerhalb von 30 Tagen gelöscht. Von
  automatischen Systemen wegen Verstößen gegen die Nutzungsrichtlinien markierte
  Inhalte bis zu 2 Jahre.
  [privacy.claude.com – Speicherdauer](https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data)
- **Zero Data Retention (ZDR):** Auf Anfrage beim Anthropic-Vertrieb, je Organisation.
  Dann wird nach der Antwort nichts gespeichert. Gilt für die Messages-API mit Caching,
  Structured Outputs und Thinking, also alles, was diese App braucht. Nicht für
  Batch-API, Files-API und die Console/Playground.
  [API and data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention)
- **Vertrag:** Die Verarbeitung erfolgt nach dem Data Processing Addendum (DPA), das Teil
  der Commercial Terms ist. Vertragspartner für Kunden im EWR ist Anthropic Ireland, Limited.
  [DPA](https://www.anthropic.com/legal/data-processing-addendum)
- **Datenstandort:** Die API kennt `inference_geo` zur Wahl der Region. Welche Regionen
  angeboten werden, ist noch zu prüfen. Alternativ läuft Claude über AWS Bedrock oder Google
  Vertex in EU-Regionen.
- Briefe nie in claude.ai oder im Console-Playground einfügen – das läuft nicht unter
  denselben Bedingungen.

Für die Freigabe durch Datenschutz/Ethik: anonymisierte Briefe, kein Training, 30 Tage
Löschfrist bzw. ZDR, DPA mit Anthropic Ireland.

## Benutzung

```bash
cd /Users/peterbernstein/Python/XrayStats/letter2entry_claude
PY=/Users/peterbernstein/Python/ICFXStats/venv/bin/python
$PY run_claude.py check                 # Selbsttest, ohne API
XRAY_TOKEN=… $PY run_claude.py fetch    # Briefe nicht validierter Patienten → data/letters.json
$PY run_claude.py run --limit 10        # Testlauf, Kosten je Brief werden ausgegeben
$PY run_claude.py run --limit 500       # Rest; schon bearbeitete Briefe werden übersprungen
/Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run review_app.py
```

`run` braucht Anthropic-Zugangsdaten (`ANTHROPIC_API_KEY` oder `ant auth login`).
Optionen: `--model` (Standard `claude-opus-5-5`), `--effort` (Standard `medium`).
Bei einer Ablehnung durch die Sicherheitsfilter springt automatisch ein anderes Modell
ein (`fallbacks="default"`); welches Modell geantwortet hat, steht im Ergebnis.
Ein erneuter `fetch` holt den aktuellen Stand, ohne die Claude-Ergebnisse zu verlieren.

## Dateien

| Datei | Zweck |
|---|---|
| `run_claude.py` | `fetch` (API → Cache), `run` (Claude, nur Briefe ohne Ergebnis), `check` (Selbsttest); `diff_entry()` für die App |
| `review_app.py` | Streamlit: Prüfliste nach offenen Abweichungen, Brief mit markierten Claude-Zitaten, Urteil je Abweichung, Auswertung |
| `data/` | `letters.json`, `claude_results.json`, `decisions.csv` (gitignored) |

Abhängigkeit `anthropic` ist im ICFXStats-venv installiert.
