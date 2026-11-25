#!/usr/bin/env python3
"""Arztbriefe nicht validierter Patienten -> Claude-Annotationen (Konzept: README.md).

    python run_claude.py fetch               # API -> data/letters.json  (XRAY_TOKEN oder Prompt)
    python run_claude.py run --limit 10      # nur Briefe ohne Ergebnis -> data/claude_results.json
    python run_claude.py check               # Selbsttest ohne API

Prompt und Zitatfilter kommen aus dem produktiven letter2entry.py (Django-Repo), damit
SCADS und Claude denselben Prompt sehen. Pfad per LETTER2ENTRY_PY.
"""
import argparse, getpass, importlib.util, json, os, sys, time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LETTERS = os.path.join(DATA, "letters.json")
RESULTS = os.path.join(DATA, "claude_results.json")
API = "https://xray-api.renecol.org/api"
L2E_PATH = os.environ.get("LETTER2ENTRY_PY", os.path.expanduser(
    "~/Django/xray_api/api/helpers/letter2entry.py"))
DIAG = "base.diagnoses"  # = letter2entry.DIAGNOSES_FIELD
# $ pro 1M Tokens: input, cache write, cache read, output
PRICES = {"claude-opus-5-5": (4, 5, 0.20, 20), "claude-sonnet-5-5": (2, 2.5, 0.10, 10),
          "claude-haiku-5-5": (0.10, 0.125, 0.01, 0.50)}


def load_l2e():
    spec = importlib.util.spec_from_file_location("letter2entry", L2E_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- fetch

def fetch(token):
    import requests
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    def get(ep):
        r = s.get(f"{API}/{ep}/", timeout=180)
        r.raise_for_status()
        return r.json()

    fields = [f for f in get("clinicalfields") if f.get("llm_extract") and f.get("is_active", True)]
    keys = {f["key"] for f in fields}
    options = {k: v for k, v in get("clinicalentrydata/getentryoptions").items() if k in keys}
    patients = get("patients")
    validated = {p["uuid"] for p in patients
                 if any(t.get("status") == "VALID" for t in p.get("progresstag") or [])}
    base = {b["patient"]: (b.get("data") or {}).get("diagnoses") or []
            for b in get("clinicalbasedata") if b.get("patient")}
    base = {p: ([d] if isinstance(d, str) else d) for p, d in base.items()}
    options[DIAG] = sorted({str(d).strip() for ds in base.values() for d in ds if str(d).strip()})

    entries = []
    for e in get("clinicalentrydata"):
        letter = e.get("letter") or {}
        if not letter.get("sections") or e.get("patient") in validated:
            continue
        anns = letter.get("annotations") or []
        state = ("imported" if not anns else
                 "unreviewed" if any(a.get("status") == "predicted" for a in anns) else "reviewed")
        entries.append({"uuid": e["uuid"], "patient": e["patient"], "date": e.get("referring_date"),
                        "state": state, "sections": letter["sections"], "annotations": anns,
                        "data": e.get("data") or {}, "diagnoses": base.get(e["patient"], [])})

    os.makedirs(DATA, exist_ok=True)
    with open(LETTERS, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": time.strftime("%Y-%m-%d %H:%M"), "fields": fields,
                   "options": options, "entries": entries}, f, ensure_ascii=False, indent=1)
    by = {}
    for e in entries:
        by[e["state"]] = by.get(e["state"], 0) + 1
    print(f"{len(entries)} Briefe nicht validierter Patienten ({len(validated)} Patienten validiert, "
          f"ausgelassen): {by} -> {LETTERS}")


# --------------------------------------------------------------------------- prompt

def build_system(l2e, fields, options):
    """Produktiver Prompt + eine Zeile Wertformat aus `kind`.

    Der Prompt-Kopf nennt die Bool-/Zahlenfelder fest beim Namen; die Zeile hier deckt
    auch Felder ab, die eine Institution später ergänzt.
    """
    objs = [SimpleNamespace(key=f["key"], prompt=f.get("prompt") or "") for f in fields]
    kinds = lambda k: [f["key"] for f in fields if f["kind"] == k]
    return (l2e.system_prompt(objs, options)
            + "\n\n# Wertformat nach Feldtyp\n"
            + f'Ja/Nein-Felder ("true" oder "false"): {", ".join(kinds("bool")) or "-"}\n'
            + f'Zahlenfelder (nur die Zahl als Text): {", ".join(kinds("number")) or "-"}\n'
            + 'Gib die Einträge im Feld "annotations" des JSON-Objekts zurück.')


def schema(keys):
    item = {"type": "object", "additionalProperties": False,
            "required": ["section", "quote", "field", "value"],
            "properties": {"section": {"type": "string"}, "quote": {"type": "string"},
                           "field": {"type": "string", "enum": sorted(keys)},
                           "value": {"type": "string"}}}
    # ponytail: Schema für alle Briefe gleich (keine Section-Enums je Brief), damit der
    # Prompt-Cache hält; falsche Sections fängt parse_annotations() ab.
    return {"type": "object", "additionalProperties": False, "required": ["annotations"],
            "properties": {"annotations": {"type": "array", "items": item}}}


def cost(model, u):
    p = PRICES.get(model)
    if not p:
        return None
    return (u["input_tokens"] * p[0] + u["cache_creation_input_tokens"] * p[1]
            + u["cache_read_input_tokens"] * p[2] + u["output_tokens"] * p[3]) / 1e6


# --------------------------------------------------------------------------- run

def run(limit, model, effort):
    import anthropic
    l2e = load_l2e()
    with open(LETTERS, encoding="utf-8") as f:
        db = json.load(f)
    results = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else {}
    keys = {f["key"] for f in db["fields"]} | {DIAG}
    system = build_system(l2e, db["fields"], db["options"])
    fmt = {"type": "json_schema", "schema": schema(keys)}
    todo = [e for e in db["entries"] if e["uuid"] not in results][:limit]
    print(f"{len(todo)} Briefe an {model} (effort={effort}), {len(results)} schon im Cache")

    client = anthropic.Anthropic()
    total = 0.0
    # ponytail: sequentiell (~30 s/Brief). ThreadPool + Lock beim Speichern, wenn es eilt.
    for i, e in enumerate(todo, 1):
        t0 = time.monotonic()
        try:
            with client.beta.messages.stream(
                model=model, max_tokens=32000,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": json.dumps(e["sections"], ensure_ascii=False)}],
                output_config={"effort": effort, "format": fmt},
            ) as stream:
                msg = stream.get_final_message()
        except anthropic.APIStatusError as ex:
            print(f"  [{i}] {e['uuid'][:8]} API-Fehler {ex.status_code}: {ex.message}")
            if ex.status_code in (401, 403):
                break
            continue
        except anthropic.APIConnectionError as ex:
            print(f"  [{i}] {e['uuid'][:8]} Verbindung: {ex}")
            continue

        u = {k: getattr(msg.usage, k, 0) or 0 for k in
             ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
        c = cost(msg.model, u) or 0.0
        total += c
        if msg.stop_reason in ("refusal", "max_tokens"):
            print(f"  [{i}] {e['uuid'][:8]} abgebrochen: {msg.stop_reason}")
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        raw = json.loads(text).get("annotations", [])
        anns = l2e.parse_annotations(json.dumps(raw, ensure_ascii=False), e["sections"], keys)
        results[e["uuid"]] = {"model": msg.model, "effort": effort, "at": time.strftime("%Y-%m-%d %H:%M"),
                              "annotations": anns, "dropped": len(raw) - len(anns), "usage": u, "cost": c}
        with open(RESULTS, "w", encoding="utf-8") as f:  # nach jedem Brief: Abbruch kostet nichts doppelt
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"  [{i}/{len(todo)}] {e['uuid'][:8]} {e['state']:<10} {len(anns):>3} Ann. "
              f"(-{len(raw) - len(anns)} verworfen)  {time.monotonic() - t0:5.1f}s  ${c:.3f}")
    print(f"Kosten dieses Laufs: ${total:.2f}")


# --------------------------------------------------------------------------- diff (auch von review_app.py genutzt)

def norm(v):
    return " ".join(str(v).split()).lower()


def _vals(kind, v):
    """Ein Feldwert -> Menge angezeigter Werte. Bool/Zahl: nur 'gesetzt' zählt."""
    if kind == "bool":
        return ["true"] if v is True or norm(v) == "true" else []
    if kind == "number":
        try:
            n = float(v)
        except (TypeError, ValueError):
            return []
        return [str(int(n)) if n == int(n) else str(n)] if n else []
    items = v if isinstance(v, list) else [v]
    return [" ".join(str(x).split()) for x in items if x not in (None, "") and str(x).strip()]


def diff_entry(entry, claude_anns, kinds):
    """Zeilen {field, value, source, quote, section} mit source ∈ beide | nur Claude | nur aktuell.

    `aktuell` = ClinicalEntryData.data (bzw. die Diagnosen des Patienten). Diagnosen gelten
    je Patient, nicht je Brief: dort wird 'nur aktuell' nicht gemeldet.
    """
    cur, cla = {}, {}
    for k, kind in kinds.items():
        src = entry.get("diagnoses", []) if k == DIAG else entry["data"].get(k)
        for v in _vals(kind, src):
            cur.setdefault(k, {})[norm(v)] = v
    for a in claude_anns:
        kind = kinds.get(a["field"])
        for v in _vals(kind, a["value"]) if kind else []:
            cla.setdefault(a["field"], {}).setdefault(norm(v), (v, a["quote"], a.get("section", "")))
    rows = []
    for k in kinds:
        c, m = cur.get(k, {}), cla.get(k, {})
        for n in sorted(set(c) | set(m)):
            if n in c and n in m:
                rows.append({"field": k, "value": c[n], "source": "beide", "quote": m[n][1], "section": m[n][2]})
            elif n in m:
                rows.append({"field": k, "value": m[n][0], "source": "nur Claude", "quote": m[n][1], "section": m[n][2]})
            elif k != DIAG:
                rows.append({"field": k, "value": c[n], "source": "nur aktuell", "quote": "", "section": ""})
    return rows


def kinds_of(fields):
    return {**{f["key"]: f["kind"] for f in fields}, DIAG: "list"}


# --------------------------------------------------------------------------- check

def check():
    l2e = load_l2e()
    fields = [{"key": "spinal_symptoms", "kind": "list", "prompt": "Symptome"},
              {"key": "opioids_before", "kind": "bool", "prompt": "Opiate?"},
              {"key": "fused_segments", "kind": "number", "prompt": "Segmente"}]
    sysp = build_system(l2e, fields, {"spinal_symptoms": ["Lumbalgie"]})
    assert "## spinal_symptoms" in sysp and "opioids_before" in sysp and "Lumbalgie" in sysp
    s = schema({"spinal_symptoms", DIAG})
    assert s["properties"]["annotations"]["items"]["properties"]["field"]["enum"] == [DIAG, "spinal_symptoms"]

    sections = {"Anamnese": "Seit Jahren Lumbalgie,\nzudem Fußheberparese links."}
    raw = [{"section": "Anamnese", "quote": "Lumbalgie, zudem", "field": "spinal_symptoms", "value": "Lumbalgie"},
           {"section": "Anamnese", "quote": "Fußheberparese links", "field": "spinal_symptoms", "value": "Fußheberparese"},
           {"section": "Anamnese", "quote": "erfunden", "field": "spinal_symptoms", "value": "X"},
           {"section": "Anamnese", "quote": "Lumbalgie", "field": "unbekannt", "value": "Y"}]
    keys = {"spinal_symptoms", "opioids_before", "fused_segments", DIAG}
    anns = l2e.parse_annotations(json.dumps(raw), sections, keys)
    assert [a["value"] for a in anns] == ["Lumbalgie", "Fußheberparese"], anns  # Zitat-/Feldfilter

    entry = {"data": {"spinal_symptoms": ["lumbalgie ", "Ischialgie"], "opioids_before": True,
                      "fused_segments": 0}, "diagnoses": ["M41.1 Skoliose"]}
    anns += [{"field": "opioids_before", "value": "true", "quote": "q"},
             {"field": "fused_segments", "value": "2", "quote": "q"},
             {"field": DIAG, "value": "M48.0 Spinalstenose", "quote": "q"}]
    got = {(r["field"], norm(r["value"]), r["source"]) for r in diff_entry(entry, anns, kinds_of(fields))}
    assert got == {("spinal_symptoms", "lumbalgie", "beide"),
                   ("spinal_symptoms", "fußheberparese", "nur Claude"),
                   ("spinal_symptoms", "ischialgie", "nur aktuell"),
                   ("opioids_before", "true", "beide"),
                   ("fused_segments", "2", "nur Claude"),
                   (DIAG, "m48.0 spinalstenose", "nur Claude")}, got  # M41.1 nicht als 'nur aktuell'
    assert cost("claude-opus-5-5", {"input_tokens": 1e6, "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0, "output_tokens": 0}) == 4
    print("check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "check"])
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(os.environ.get("XRAY_TOKEN") or getpass.getpass("xray-api Token: "))
    elif a.cmd == "run":
        if not os.path.exists(LETTERS):
            sys.exit("Erst `python run_claude.py fetch`.")
        run(a.limit, a.model, a.effort)
    else:
        check()
