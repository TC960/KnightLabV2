#!/usr/bin/env python3
"""TASK 2 -- LLM-extracted metadata vs hand-curated metadata.

metadata.jsonl holds country + sequencing type pulled from full text by an LLM for
250 papers. The correct datasheet has both HAND-CURATED at ~99% coverage. Same
papers, two independent annotators -> a direct measurement of how well an LLM
replaces a human curator on bibliographic-ish fields.

Two fields, deliberately different in kind:
  country     -- open vocabulary, needs surface normalisation before comparison
  sequencing  -- closed vocabulary, but the two label spaces are NOT identical
                 (the human sheet has a "Both" class the LLM prompt never had),
                 so raw agreement understates true agreement. Reported both ways.

Emits disagreement cases with the paper text windows needed to adjudicate them.
Writes meta_agreement.json + meta_disagreements.json. Reads only.
"""
import csv, json, os, re, random, collections, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
NEW_CSV = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
META = os.path.join(HERE, "metadata.jsonl")
PAPERS = os.path.join(P3, "EmilySong_GoldStandardPaper", "all_usable_papers.json")
NPERM = 10000

COUNTRY_ALIAS = {
    "usa": "united states", "us": "united states", "u.s.a.": "united states",
    "united states of america": "united states",
    "republic of korea": "south korea", "korea": "south korea",
    "korea, republic of": "south korea", "south korea": "south korea",
    "uk": "united kingdom", "u.k.": "united kingdom",
    "great britain": "united kingdom", "england": "united kingdom",
    "russian federation": "russia", "the netherlands": "netherlands",
    "holland": "netherlands", "czechia": "czech republic",
    "republic of china": "taiwan", "taiwan, china": "taiwan",
    "people's republic of china": "china", "prc": "china",
    "brasil": "brazil", "iran, islamic republic of": "iran",
}


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def norm_country(c):
    c = (c or "").strip().lower().rstrip(".")
    c = re.sub(r"\s+", " ", c)
    return COUNTRY_ALIAS.get(c, c)


def norm_seq(s, source):
    """Map both label spaces onto {16s, shotgun, both, other}."""
    s = (s or "").strip().lower()
    if not s:
        return ""
    if source == "human":
        return {"16s": "16s", "shotgun": "shotgun", "both": "both", "other": "other"}.get(s, "other")
    # LLM vocabulary
    if "16s" in s:
        return "16s"
    if "shotgun" in s or "metagenom" in s:
        return "shotgun"
    return "other"          # qPCR, other, ...


def perm_prop_vs(flags, nperm=NPERM, seed=0):
    """Not a group comparison -- just a bootstrap CI on an agreement rate.
    Each flag is one PAPER, so resampling papers is the correct unit."""
    rng = random.Random(seed)
    n = len(flags)
    boots = []
    for _ in range(2000):
        boots.append(st.mean([flags[rng.randrange(n)] for _ in range(n)]))
    boots.sort()
    return {"rate": round(st.mean(flags), 4), "n": n,
            "ci95": [round(boots[50], 4), round(boots[1949], 4)]}


def perm_vs_chance(a, b, nperm=NPERM, seed=0):
    """Is observed agreement better than chance given the two marginals?
    Shuffle one annotator's labels across PAPERS (paper-level shuffle) and
    recompute agreement. Also reports Cohen's kappa."""
    obs = st.mean([1.0 if x == y else 0.0 for x, y in zip(a, b)])
    b2 = list(b); rng = random.Random(seed); ge = 0; nulls = []
    for _ in range(nperm):
        rng.shuffle(b2)
        v = st.mean([1.0 if x == y else 0.0 for x, y in zip(a, b2)])
        nulls.append(v)
        if v >= obs - 1e-12:
            ge += 1
    # Cohen's kappa
    ca, cb = collections.Counter(a), collections.Counter(b)
    n = len(a)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    kappa = (obs - pe) / (1 - pe) if pe < 1 else 0.0
    return {"observed_agreement": round(obs, 4), "chance_agreement": round(pe, 4),
            "cohens_kappa": round(kappa, 4), "null_mean": round(st.mean(nulls), 4),
            "p_one_sided": round((ge + 1) / (nperm + 1), 4), "n": n, "n_perm": nperm}


def main():
    sheet = {norm_title(r["Title"]): r for r in csv.DictReader(open(NEW_CSV))}
    llm = [json.loads(l) for l in open(META)]
    papers = {norm_title(p["title"]): p for p in json.load(open(PAPERS))}

    joined = []
    for m in llm:
        k = norm_title(m["title"])
        g = sheet.get(k)
        if not g:
            continue
        joined.append({
            "title": m["title"], "key": k,
            "llm_country_raw": (m["meta"].get("country") or "").strip(),
            "hum_country_raw": g["Country"].strip(),
            "llm_seq_raw": (m["meta"].get("sequencing") or "").strip(),
            "hum_seq_raw": g["SequencingType"].strip(),
        })
    print(f"metadata.jsonl: {len(llm)}  joined to sheet: {len(joined)}")

    out = {"n_llm": len(llm), "n_joined": len(joined)}

    # ---------------- COUNTRY -------------------------------------------
    cc = [r for r in joined if r["llm_country_raw"] and r["hum_country_raw"]]
    a = [norm_country(r["llm_country_raw"]) for r in cc]
    b = [norm_country(r["hum_country_raw"]) for r in cc]
    flags = [1.0 if x == y else 0.0 for x, y in zip(a, b)]
    out["country"] = {
        "n_both_present": len(cc),
        "n_llm_blank": sum(1 for r in joined if not r["llm_country_raw"]),
        "n_human_blank": sum(1 for r in joined if not r["hum_country_raw"]),
        "agreement": perm_prop_vs(flags, seed=11),
        "vs_chance": perm_vs_chance(a, b, seed=12),
    }
    print("\n=== COUNTRY ===")
    print(f"  both present: {len(cc)}  (llm blank {out['country']['n_llm_blank']}, "
          f"human blank {out['country']['n_human_blank']})")
    print(f"  agreement {out['country']['agreement']['rate']:.4f} "
          f"CI95 {out['country']['agreement']['ci95']}")
    v = out["country"]["vs_chance"]
    print(f"  kappa={v['cohens_kappa']:.4f}  chance={v['chance_agreement']:.4f}  "
          f"p={v['p_one_sided']:.4f}")

    # exact-string agreement before normalisation, to size the normalisation gain
    raw_flags = [1.0 if r["llm_country_raw"].lower() == r["hum_country_raw"].lower() else 0.0
                 for r in cc]
    out["country"]["raw_string_agreement"] = round(st.mean(raw_flags), 4)
    print(f"  (raw string agreement before alias normalisation: {st.mean(raw_flags):.4f})")

    # ---------------- SEQUENCING ----------------------------------------
    ss = [r for r in joined if r["llm_seq_raw"] and r["hum_seq_raw"]]
    sa = [norm_seq(r["llm_seq_raw"], "llm") for r in ss]
    sb = [norm_seq(r["hum_seq_raw"], "human") for r in ss]
    sflags = [1.0 if x == y else 0.0 for x, y in zip(sa, sb)]
    out["sequencing"] = {
        "n_both_present": len(ss),
        "n_llm_blank": sum(1 for r in joined if not r["llm_seq_raw"]),
        "n_human_blank": sum(1 for r in joined if not r["hum_seq_raw"]),
        "agreement_all": perm_prop_vs(sflags, seed=13),
        "vs_chance": perm_vs_chance(sa, sb, seed=14),
    }
    # the human-only "Both" class the LLM schema cannot express
    nb = [(x, y) for x, y in zip(sa, sb) if y != "both"]
    out["sequencing"]["n_human_both"] = sum(1 for y in sb if y == "both")
    out["sequencing"]["agreement_excl_human_Both"] = perm_prop_vs(
        [1.0 if x == y else 0.0 for x, y in nb], seed=15)
    print("\n=== SEQUENCING ===")
    print(f"  both present: {len(ss)}  (llm blank {out['sequencing']['n_llm_blank']}, "
          f"human blank {out['sequencing']['n_human_blank']})")
    print(f"  agreement (all) {out['sequencing']['agreement_all']['rate']:.4f} "
          f"CI95 {out['sequencing']['agreement_all']['ci95']}")
    print(f"  human 'Both' rows (label the LLM schema cannot emit): "
          f"{out['sequencing']['n_human_both']}")
    print(f"  agreement excluding those: "
          f"{out['sequencing']['agreement_excl_human_Both']['rate']:.4f} "
          f"CI95 {out['sequencing']['agreement_excl_human_Both']['ci95']}")
    v = out["sequencing"]["vs_chance"]
    print(f"  kappa={v['cohens_kappa']:.4f}  chance={v['chance_agreement']:.4f}  p={v['p_one_sided']:.4f}")

    cm = collections.Counter((x, y) for x, y in zip(sa, sb))
    out["sequencing"]["confusion_llm_x_human"] = {f"llm={x}|human={y}": n
                                                  for (x, y), n in cm.most_common()}
    print("  confusion (llm | human):")
    for (x, y), n in cm.most_common():
        print(f"    llm={x:9s} human={y:9s} {n:4d}{'   <-- agree' if x == y else ''}")

    # ---------------- disagreement dossiers -----------------------------
    dis = []
    for r, x, y in zip(cc, a, b):
        if x != y:
            dis.append({"field": "country", "title": r["title"], "key": r["key"],
                        "llm": r["llm_country_raw"], "human": r["hum_country_raw"]})
    for r, x, y in zip(ss, sa, sb):
        if x != y:
            dis.append({"field": "sequencing", "title": r["title"], "key": r["key"],
                        "llm": r["llm_seq_raw"], "human": r["hum_seq_raw"],
                        "llm_norm": x, "human_norm": y})
    print(f"\ntotal disagreements: {len(dis)} "
          f"(country {sum(1 for d in dis if d['field']=='country')}, "
          f"sequencing {sum(1 for d in dis if d['field']=='sequencing')})")

    # attach adjudication evidence: text windows around the relevant cues
    CUE = {
        "country": r"(?i)\b(department|hospital|university|institute|college|centre|center|"
                   r"school of|affiliat|recruit|enrolled|participants were|cohort|"
                   r"ethic|informed consent|China|Chinese|Japan|Japanese|Korea|Italy|Italian|"
                   r"Germany|German|Spain|Spanish|Taiwan|USA|United States|Turkey|Russia|"
                   r"Finland|Israel|Egypt|India|Australia|Canada|Brazil|Netherlands|Austria)\b",
        "sequencing": r"(?i)(16S|V3-V4|V3–V4|V4|V1-V2|rRNA gene|amplicon|shotgun|"
                      r"metagenomic|whole[- ]genome sequencing|WGS|qPCR|quantitative PCR|"
                      r"Illumina|MiSeq|HiSeq|NovaSeq|Ion Torrent|454|PacBio|Nanopore)",
    }
    for d in dis:
        p = papers.get(d["key"])
        if not p:
            d["windows"] = None
            continue
        txt = re.sub(r"\s+", " ", p["text"])
        hits, seen = [], set()
        for m in re.finditer(CUE[d["field"]], txt):
            s = max(0, m.start() - 220); e = min(len(txt), m.end() + 220)
            w = txt[s:e]
            kk = w[:60]
            if kk in seen:
                continue
            seen.add(kk)
            hits.append(w)
            if len(hits) >= 14:
                break
        d["windows"] = hits
        d["char_len"] = len(txt)

    rng = random.Random(99)
    rng.shuffle(dis)
    json.dump(dis, open(os.path.join(HERE, "meta_disagreements.json"), "w"), indent=2)
    out["n_disagreements"] = len(dis)
    out["normalisation"] = {"country_aliases": COUNTRY_ALIAS,
                            "note": "sequencing mapped to {16s,shotgun,both,other}; "
                                    "LLM schema has no 'both' class"}
    json.dump(out, open(os.path.join(HERE, "meta_agreement.json"), "w"), indent=2)
    print("wrote meta_agreement.json + meta_disagreements.json")


if __name__ == "__main__":
    main()
