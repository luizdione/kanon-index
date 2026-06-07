#!/usr/bin/env python3
"""Resumable collector for alternative author-level indices (real OpenAlex data).

For each researcher in the KANON benchmark, computes from the per-work citation
vector and per-work author counts: recomputed h, Egghe g, Batista h_I
(= h^2 / N_a over the h-core), and Schreiber fractional h_m. Works are fetched
citation-sorted and capped at MAXPAGES*200 most-cited works, which preserves
h, g, h_I and h_m exactly while bounding the payload from hyperauthored papers.

Real-data-only; resumable (skips cached ORCIDs). NOTE: filtering works by
authorships.author.orcid misses authors whose ORCID is unlinked in OpenAlex
(observed for ~150/164 laureates). For a full-cohort run, resolve each profile
to its canonical OpenAlex author id and query filter=author.id:<id> instead.
h_alpha (Hirsch 2019) is not computed (requires every co-author's citations).
"""
import urllib.request, urllib.error, json, csv, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "dados_reais/2026-05-30/altindices_cache.json")
CSV = os.path.join(HERE, "_per_researcher_full.csv")
MAILTO = "luiz.melo@ifrj.edu.br"
BASE = "https://api.openalex.org/works"
BUDGET_S = float(os.environ.get("BUDGET_S", "36"))
MAXPAGES = 4
WORKERS = 16

def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save_cache(c):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"; json.dump(c, open(tmp, "w")); os.replace(tmp, CACHE)
def fetch_works(orcid):
    out = []; cursor = "*"; pages = 0
    while cursor and pages < MAXPAGES:
        url = (BASE + "?filter=authorships.author.orcid:https://orcid.org/" + orcid +
               "&select=cited_by_count,authorships&sort=cited_by_count:desc"
               "&per-page=200&cursor=" + cursor + "&mailto=" + MAILTO)
        req = urllib.request.Request(url, headers={"User-Agent": "kanon-alt/1.0"})
        d = json.load(urllib.request.urlopen(req, timeout=20))
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
    works = fetch_works(r["orcid"]); cits = [a[0] for a in works]
    return r["orcid"], {"name": r["name"], "field": r["field"], "group": r["group"],
        "data_quality": r["data_quality"],
        "h_csv": int(float(r["h_index"])) if r["h_index"] else None,
        "n_works": len(works), "h_recomp": h_index(cits), "g": g_index(cits),
        "h_I": round(batista_hI(works), 4), "h_m": round(schreiber_hm(works), 4)}
def main():
    cache = load_cache(); rows = list(csv.DictReader(open(CSV)))
    todo = [r for r in rows if r["orcid"] and r["orcid"] not in cache]
    print("total %d cached %d todo %d" % (len(rows), len(cache), len(todo)), flush=True)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i in range(0, len(todo), WORKERS):
            if time.time() - t0 > BUDGET_S: print("budget reached", flush=True); break
            futs = {ex.submit(work_one, r): r for r in todo[i:i + WORKERS]}
            for f in as_completed(futs):
                r = futs[f]
                try: orc, rec = f.result(); cache[orc] = rec; done += 1
                except urllib.error.HTTPError as exc: cache[r["orcid"]] = {"error": "http %s" % exc.code, "name": r["name"]}
                except Exception as exc: print("ERR %s %s" % (repr(exc)[:60], r["orcid"]), flush=True)
            save_cache(cache); print("batch ok total %d (%.0fs)" % (done, time.time() - t0), flush=True)
    save_cache(cache); print("saved cache %d/%d" % (len(cache), len(rows)), flush=True)
if __name__ == "__main__":
    main()
