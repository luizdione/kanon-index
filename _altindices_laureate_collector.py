#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flag 5: laureate alt-index collection by CANONICAL OpenAlex author id.

150/164 laureate ORCIDs are unlinked in OpenAlex, so the ORCID-keyed collector
(_altindices_collector.py) returned them empty. Here we resolve each laureate to
its canonical OpenAlex author id by name search + citation/works validation
against the collected profile (_per_researcher_full.csv), then fetch works by
filter=authorships.author.id:<id> and compute the SAME indices (h, g, h_I, h_m)
with identical formulas, so laureate and elite values are directly comparable.

Resumable: caches per-laureate to dados_reais/2026-05-31/laureate_altindices_cache.json.
Run repeatedly until 'todo 0'. Real-data-only; no fabricated values.
"""
import urllib.request, urllib.parse, urllib.error, json, csv, os, time, math
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "dados_reais/2026-05-31/laureate_altindices_cache.json")
CSV = os.path.join(HERE, "_per_researcher_full.csv")
MAILTO = "luiz.melo@ifrj.edu.br"
WORKS = "https://api.openalex.org/works"
AUTHORS = "https://api.openalex.org/authors"
BUDGET_S = float(os.environ.get("BUDGET_S", "38"))
MAXPAGES = 4
WORKERS = int(os.environ.get("WORKERS", "8"))

def load_cache(): return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save_cache(c):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"; json.dump(c, open(tmp, "w")); os.replace(tmp, CACHE)

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kanon-alt-laureate/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))

def lastname(name): return name.strip().split()[-1].lower()

def resolve_author(name, cit_csv):
    """Return (author_id, display_name, cited_by_count, works_count) best match or None."""
    q = urllib.parse.quote(name)
    url = f"{AUTHORS}?search={q}&per-page=15&mailto={MAILTO}"
    try: d = _get(url)
    except Exception: return None
    cands = d.get("results", [])
    ln = lastname(name)
    best = None; best_score = -1e9
    for c in cands:
        dn = c.get("display_name", "") or ""
        if ln not in dn.lower():  # require surname match
            continue
        cby = c.get("cited_by_count", 0) or 0
        wc = c.get("works_count", 0) or 0
        if wc < 5:  # avoid stub author records
            continue
        # citation-proximity score (lower |log ratio| better); prefer high-citation match
        if cit_csv and cit_csv > 0 and cby > 0:
            score = -abs(math.log((cby + 1.0) / (cit_csv + 1.0)))
        else:
            score = math.log(cby + 1.0)  # fallback: most-cited surname match
        if score > best_score:
            best_score = score; best = (c.get("id", "").split("/")[-1], dn, cby, wc)
    return best

def fetch_works_by_author(aid):
    out = []; cursor = "*"; pages = 0
    while cursor and pages < MAXPAGES:
        url = (WORKS + "?filter=authorships.author.id:" + aid +
               "&select=cited_by_count,authorships&sort=cited_by_count:desc"
               "&per-page=200&cursor=" + cursor + "&mailto=" + MAILTO)
        d = _get(url)
        for w in d["results"]:
            na = len(w.get("authorships", []))
            out.append((int(w.get("cited_by_count", 0)), na if na > 0 else 1))
        cursor = d["meta"].get("next_cursor"); pages += 1
        if not d["results"]: break
    return out

def h_index(c):
    s = sorted(c, reverse=True); h = 0
    for i, x in enumerate(s, 1):
        if x >= i: h = i
        else: break
    return h
def g_index(c):
    s = sorted(c, reverse=True); cum = 0; g = 0
    for i, x in enumerate(s, 1):
        cum += x
        if cum >= i * i: g = i
        else: break
    return g
def batista_hI(w):
    ws = sorted(w, key=lambda x: x[0], reverse=True); h = h_index([a[0] for a in ws])
    if h == 0: return 0.0
    Na = sum(a[1] for a in ws[:h]); return (h * h) / Na if Na > 0 else 0.0
def schreiber_hm(w):
    ws = sorted(w, key=lambda x: x[0], reverse=True); r = 0.0; hm = 0.0
    for c, na in ws:
        r += 1.0 / na
        if c >= r: hm = r
        else: break
    return hm

def work_one(r):
    name = r["name"]; cit_csv = float(r["total_citations"]) if r.get("total_citations") else 0.0
    res = resolve_author(name, cit_csv)
    if not res:
        return name, {"name": name, "field": r["field"], "group": "Nobel",
                      "resolved": False, "reason": "no surname/citation match"}
    aid, dn, cby, wc = res
    works = fetch_works_by_author(aid); cits = [a[0] for a in works]
    hrec = h_index(cits); hcsv = int(float(r["h_index"])) if r.get("h_index") else None
    return name, {"name": name, "field": r["field"], "group": "Nobel", "resolved": True,
                  "author_id": aid, "openalex_name": dn, "cited_by_count": cby,
                  "works_count": wc, "cit_csv": cit_csv,
                  "h_csv": hcsv, "n_works": len(works), "h_recomp": hrec,
                  "g": g_index(cits), "h_I": round(batista_hI(works), 4),
                  "h_m": round(schreiber_hm(works), 4),
                  "h_match": (hcsv is not None and abs(hrec - hcsv) <= 3)}

def main():
    cache = load_cache()
    rows = [r for r in csv.DictReader(open(CSV)) if r["group"].strip().lower().startswith("nobel")]
    todo = [r for r in rows if r["name"] not in cache]
    print("laureates %d cached %d todo %d" % (len(rows), len(cache), len(todo)), flush=True)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i in range(0, len(todo), WORKERS):
            if time.time() - t0 > BUDGET_S: print("budget reached", flush=True); break
            futs = {ex.submit(work_one, r): r for r in todo[i:i + WORKERS]}
            for f in as_completed(futs):
                r = futs[f]
                try: nm, rec = f.result(); cache[nm] = rec; done += 1
                except urllib.error.HTTPError as exc: cache[r["name"]] = {"error": "http %s" % exc.code, "name": r["name"]}
                except Exception as exc: print("ERR %s %s" % (repr(exc)[:70], r["name"]), flush=True)
            save_cache(cache); print("batch total done %d (%.0fs)" % (done, time.time() - t0), flush=True)
    save_cache(cache)
    res = sum(1 for v in cache.values() if v.get("resolved"))
    match = sum(1 for v in cache.values() if v.get("h_match"))
    print("saved %d/%d | resolved %d | h_match %d" % (len(cache), len(rows), res, match), flush=True)

if __name__ == "__main__":
    main()
