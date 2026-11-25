"""Review: Claude-Extraktion vs. aktueller Stand für nicht validierte Arztbriefe.

    cd letter2entry_claude
    /Users/peterbernstein/Python/ICFXStats/venv/bin/streamlit run review_app.py

Links der Brief, rechts die Items – beide scrollen unabhängig. Ein angewähltes Item wird
im Brief hervorgehoben und in den sichtbaren Bereich gescrollt. Liest data/letters.json +
data/claude_results.json (run_claude.py), schreibt nur data/decisions.csv.
"""
import hashlib, html, json, os, re, time

import pandas as pd
import streamlit as st

from run_claude import DATA, LETTERS, RESULTS, diff_entry, kinds_of, norm

DECISIONS = os.path.join(DATA, "decisions.csv")
KEY = ["entry", "field", "norm", "source"]
ORDER = {"nur Claude": 0, "nur aktuell": 0, "beide": 1}

st.set_page_config(page_title="letter2entry · Claude-Review", layout="wide")
st.html("""<style>
mark.ann {background:#fff3a3; padding:0 1px}
mark.sel {background:#ff9f43; outline:2px solid #e8590c; padding:0 1px}
.letter {white-space:pre-wrap; font-size:0.92em; line-height:1.5}
.letter h4 {margin:1em 0 .2em; font-size:1em}
</style>""")


@st.cache_data
def load(mtime_letters, mtime_results):
    with open(LETTERS, encoding="utf-8") as f:
        db = json.load(f)
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    return db, res


def load_decisions():
    if os.path.exists(DECISIONS):
        return pd.read_csv(DECISIONS, dtype=str).fillna("")
    return pd.DataFrame(columns=KEY + ["value", "verdict", "at"])


def find(text, phrase, ignore_case=False):
    """Fundstellen von `phrase` in `text`, Leerraum-tolerant (das LLM bricht anders um)."""
    words = phrase.split()
    if not words:
        return []
    pat = r"\s+".join(re.escape(w) for w in words)
    return [m.span() for m in re.finditer(pat, text, re.I if ignore_case else 0)]


def letter_html(sections, anns, sel, nonce=""):
    """Brief als HTML: alle Claude-Zitate hell, das gewählte Item kräftig mit id.

    `sel` = gewählte Zeile; Claude-Items über ihr Zitat, 'nur aktuell' über den Wert
    selbst (Groß/klein egal) – steht er nicht wörtlich im Brief, wird nichts markiert.
    Überlappungen: die Auswahl gewinnt, sonst das zuerst gefundene Zitat.
    """
    out, found = [], False
    for sec, text in sections.items():
        spans = []
        if sel is not None:
            if sel["quote"] and sel["section"] == sec:
                spans += [(a, b, "sel") for a, b in find(text, sel["quote"])[:1]]
            elif not sel["quote"]:
                spans += [(a, b, "sel") for a, b in find(text, sel["value"], ignore_case=True)]
        for q in {a["quote"] for a in anns if a["section"] == sec}:
            spans += [(a, b, "ann") for a, b in find(text, q)[:1]]
        kept = []
        for a, b, cls in sorted(spans, key=lambda s: (s[2] != "sel", s[0])):
            if all(b <= x or a >= y for x, y, _ in kept):
                kept.append((a, b, cls))
        h, pos = [], 0
        for a, b, cls in sorted(kept):
            ident = ""
            if cls == "sel" and not found:
                ident, found = f' id="sel" data-n="{nonce}"', True
            h += [html.escape(text[pos:a]), f'<mark class="{cls}"{ident}>{html.escape(text[a:b])}</mark>']
            pos = b
        h.append(html.escape(text[pos:]))
        out.append(f"<h4>{html.escape(sec)}</h4>{''.join(h)}")
    return f'<div class="letter">{"".join(out)}</div>', found


# Scrollt nur den Brief-Container (nächster scrollbarer Vorfahre), nicht die Seite.
# Streamlit führt das <script> nur beim ersten Rendern aus, nicht bei späteren Updates
# desselben Elements – deshalb ein einmal installierter Observer, der auf jede neue
# Auswahl (data-n = Nonce) reagiert.
SCROLL_JS = """<script>
(() => { if (window.__selObs) return;
  const go = () => { const el = document.getElementById('sel');
    if (!el || el.dataset.n === window.__selN) return;
    window.__selN = el.dataset.n;
    let p = el.parentElement;
    while (p && !(p.scrollHeight > p.clientHeight && /(auto|scroll)/.test(getComputedStyle(p).overflowY)))
      p = p.parentElement;
    if (!p) return;
    const r = el.getBoundingClientRect(), q = p.getBoundingClientRect();
    p.scrollTo({top: p.scrollTop + r.top - q.top - q.height / 3, behavior: 'smooth'});
  };
  window.__selObs = new MutationObserver(go);
  window.__selObs.observe(document.body, {childList: true, subtree: true});
  go();
})();</script>"""


if not (os.path.exists(LETTERS) and os.path.exists(RESULTS)):
    st.error("Erst `python run_claude.py fetch` und `python run_claude.py run --limit 10`.")
    st.stop()

db, res = load(os.path.getmtime(LETTERS), os.path.getmtime(RESULTS))
kinds = kinds_of(db["fields"])
labels = {f["key"]: f.get("label") or f["key"] for f in db["fields"]} | {"base.diagnoses": "Diagnosen"}
dec = load_decisions()
verdicts = {tuple(r[:4]): r[4] for r in dec[KEY + ["verdict"]].itertuples(index=False) if r[4]}

# Prüfliste: nur Briefe mit Claude-Ergebnis, nach offenen Abweichungen sortiert
rows = []
for e in db["entries"]:
    if e["uuid"] not in res:
        continue
    d = [r for r in diff_entry(e, res[e["uuid"]]["annotations"], kinds) if r["source"] != "beide"]
    open_ = sum((e["uuid"], r["field"], norm(r["value"]), r["source"]) not in verdicts for r in d)
    rows.append({"uuid": e["uuid"], "state": e["state"], "date": e.get("date") or "",
                 "patient": e["patient"][:8], "abweichungen": len(d), "offen": open_})
queue = pd.DataFrame(rows)

with st.sidebar:
    st.header("Prüfliste")
    st.caption(f"Stand API: {db['fetched_at']} · {len(db['entries'])} Briefe, {len(res)} mit Claude")
    if queue.empty:
        st.info("Noch keine Claude-Ergebnisse.")
        st.stop()
    states = st.multiselect("Brief-Stand", ["reviewed", "unreviewed", "imported"],
                            default=["reviewed", "unreviewed", "imported"])
    only_open = st.toggle("Nur mit offenen Abweichungen", value=True)
    q = queue[queue.state.isin(states)]
    if only_open:
        q = q[q.offen > 0]
    q = q.sort_values(["offen", "abweichungen"], ascending=False).set_index("uuid")
    if q.empty:
        st.success("Alles entschieden.")
        st.stop()
    uuid = st.selectbox("Brief", q.index, format_func=lambda u: (
        f"{q.loc[u, 'offen']} offen · {q.loc[u, 'state']} · {q.loc[u, 'date']} · {q.loc[u, 'patient']}"))
    pane_h = st.slider("Höhe der Fenster (px)", 400, 1400, 720, 20)

entry = next(e for e in db["entries"] if e["uuid"] == uuid)
result = res[uuid]

items = []
for r in diff_entry(entry, result["annotations"], kinds):
    v = verdicts.get((uuid, r["field"], norm(r["value"]), r["source"]), "")
    items.append(r | {"verdict": v, "status": "–" if r["source"] == "beide" else (v or "offen")})
# offene Abweichungen zuerst, dann entschiedene, dann Übereinstimmungen
items.sort(key=lambda r: (ORDER[r["source"]], r["status"] != "offen", r["field"]))

ss = st.session_state
gen_key, def_key = f"gen_{uuid}", f"def_{uuid}"
ss.setdefault(gen_key, 0)
ikey = lambda r: (r["field"], norm(r["value"]), r["source"])
is_open = lambda r: r["status"] == "offen"
# Vorauswahl als Item-Schlüssel, nicht als Zeilennummer: nach einem Urteil sortiert die
# Liste neu (entschiedene nach unten), die alte Nummer zeigte dann auf ein anderes Item.
default = next((i for i, r in enumerate(items) if ikey(r) == ss.get(def_key)),
               next((i for i, r in enumerate(items) if is_open(r)), 0 if items else None))

st.markdown(f"**{entry.get('date') or 'ohne Datum'} · Patient {entry['patient'][:8]} · {entry['state']}** "
            f"· {result['model']} (effort {result['effort']}) · {len(result['annotations'])} Annotationen, "
            f"{result['dropped']} verworfen · ${result.get('cost') or 0:.3f}")

left, right = st.columns([5, 6], gap="medium")

with right:
    table = pd.DataFrame([{"Status": r["status"], "Quelle": r["source"],
                           "Feld": labels.get(r["field"], r["field"]), "Wert": r["value"]}
                          for r in items])
    # Toolbar über der Tabelle, damit sie beim Scrollen der Liste sichtbar bleibt.
    bar = st.container()
    with st.container(height=pane_h - 70, border=True):
        ev = st.dataframe(
            table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            key=f"t_{uuid}_{ss[gen_key]}",
            selection_default={"selection": {"rows": [default]}} if default is not None else None,
            column_config={"Wert": st.column_config.TextColumn(width="large")})
    picked = ev.selection.rows
    sel = items[picked[0]] if picked else None

    with bar:
        if sel is None:
            st.caption("Item in der Liste anwählen.")
        elif sel["source"] == "beide":
            st.markdown(f"**{labels.get(sel['field'], sel['field'])}:** {sel['value']} "
                        f"– Claude und Daten stimmen überein")
        else:
            st.markdown(f"**{labels.get(sel['field'], sel['field'])}:** {sel['value']} · _{sel['source']}_",
                        help="Ist dieser Wert für diesen Brief korrekt? Nach dem Urteil springt die "
                             "Auswahl zum nächsten offenen Item.")
            b1, b2, b3, _ = st.columns([1.3, 1.2, 1.2, 2])
            ok = b1.button("✓ korrekt", type="primary", width="stretch")
            bad = b2.button("✗ falsch", width="stretch")
            reset = b3.button("↺ offen", width="stretch")
            if ok or bad or reset:
                choice = "korrekt" if ok else "falsch" if bad else ""
                k = (uuid, sel["field"], norm(sel["value"]), sel["source"])
                keep = dec[[tuple(r) != k for r in dec[KEY].itertuples(index=False)]]
                if choice:
                    keep = pd.concat([keep, pd.DataFrame([dict(zip(KEY, k), value=sel["value"],
                                      verdict=choice, at=time.strftime("%Y-%m-%d %H:%M"))])])
                os.makedirs(DATA, exist_ok=True)
                keep.to_csv(DECISIONS, index=False)
                # Nach einem Urteil zum nächsten offenen Item springen (erst unterhalb, dann von oben)
                i0 = picked[0]
                rest = [r for r in items[i0 + 1:] + items[:i0] if is_open(r)]
                ss[def_key] = ikey(rest[0]) if rest else ikey(sel)
                ss[gen_key] += 1
                st.rerun()

with left:
    nonce = hashlib.md5(json.dumps([uuid, sel], sort_keys=True).encode()).hexdigest()[:8]
    body, hit = letter_html(entry["sections"], result["annotations"], sel, nonce)
    if sel is not None and not hit:
        st.caption("Wert steht nicht wörtlich im Brief.")
    with st.container(height=pane_h, border=True):
        st.html(body + SCROLL_JS, unsafe_allow_javascript=True)

scads = [{"Feld": labels.get(a.get("field"), a.get("field")),
          "SCADS": (a.get("original") or a).get("value"), "Status": a.get("status"),
          "korrigiert zu": a.get("value") if a.get("original") else ""}
         for a in entry["annotations"] if a.get("status") != "user"]
with st.expander(f"SCADS-Vorschläge zu diesem Brief ({len(scads)})"):
    st.dataframe(pd.DataFrame(scads), hide_index=True, width="stretch")

d = dec[dec.verdict != ""]
with st.expander(f"Auswertung über alle Briefe ({len(d)} Entscheidungen)"):
    if d.empty:
        st.caption("Noch keine Entscheidungen.")
    else:
        t = (d.assign(korrekt=d.verdict == "korrekt", Feld=d.field.map(lambda k: labels.get(k, k)))
             .pivot_table(index="Feld", columns="source", values="korrekt", aggfunc=["sum", "count"])
             .fillna(0).astype(int))
        st.caption("nur Claude korrekt = Claude hat etwas gefunden, das in den Daten fehlt · "
                   "nur aktuell falsch = Fehler in den Daten, den Claude nicht übernommen hat")
        st.dataframe(t, width="stretch")
