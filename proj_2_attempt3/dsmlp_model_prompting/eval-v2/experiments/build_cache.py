#!/usr/bin/env python3
"""Persist the Opus 4.8-generated LLM stage outputs (pilot) into cache/ for the run scripts.
RELATE tokens inherited noise from the regex retrieval (multi-word candidates); we split on
' and '/'&' and strip trailing non-taxon words — the kind of cleanup a real pipeline includes."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")

def clean(tok):
    tok = re.sub(r"\b(were|was|is|are|had|have|significantly|significant|phylum|genus|"
                 r"class|order|family|level|abundance)\b", "", tok, flags=re.I)
    return re.sub(r"\s+", " ", tok).strip(" ,.")

def split_taxa(lst):
    out = []
    for t in lst:
        for part in re.split(r"\s+and\s+|\s*&\s*|,", t):
            p = clean(part)
            if p and len(p) > 2:
                out.append(p)
    # dedup preserve order
    seen = set(); res = []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower()); res.append(t)
    return res

# ---------- RELATE (retrieve -> rerank with reject) ----------
relate = {
 0:  {"e":["Phyllobacterium","Lactobacillus salivarius","Coriobacteriia","Erysipelotrichia","Gemmatimonadetes","Verrucomicrobiae","Synergistia","Cloacibacillus","Synergistetes","Bacteroidetes phylum"],
      "d":["Rhodospirillales","Bacteroides","Bifidobacterium","Monoglobus","Bifidobacterium breve","Actinobacteria"]},
 1:  {"e":["Gordonibacter","Ruminiclostridium","Enorma","Lawsonella","Frisingicoccus"],
      "d":["Catabacter","Howardella","Marine_Methylotrophic_Group","Lachnospiraceae_AC2044_group"]},
 4:  {"e":["Proteobacteria and Actinobacteriota","Actinobacteria","Alphaproteobacteria","Gammaproteobacteria","Enterobacteriales and Bifidobacteriales","Faecalibacterium","Escherichia-Shigella","Subdoligranulum","Enterobacteriaceae-unclassified","Halomonas","Bifidobacterium"],
      "d":["Bacteroidia","Negativicutes","Bacilli","Coriobacteriia","Lachnospirales and Bacteroidales","Bacteroides","Megamonas","Prevotella","Lachnospiraceae-unclassified","Blautia and Lachnospira","Collinsella and Bacteroides","Parabacteroides"]},
 10: {"e":["Clostridium cluster IV","Akkermansia","Bifidobacterium"],
      "d":["Firmicutes were significantly"]},
 11: {"e":[], "d":[]},
}
json.dump({str(k): {"taxa_enriched": split_taxa(v["e"]), "taxa_depleted": split_taxa(v["d"])}
           for k, v in relate.items()}, open(os.path.join(C, "relate.json"), "w"), indent=2)

# ---------- GROUNDED (each taxon carries a verbatim sentence) ----------
grounded = {
 0:  {"e":["Phyllobacterium","Bacteroidia","Lactobacillus salivarius","Coriobacteriia","Erysipelotrichia","Gemmatimonadetes","Verrucomicrobiae","Cloacibacillus","Clostridia","Butyricicoccus","Ligilactobacillus","Clostridium_sensu_stricto_1","Turicibacter","Enterococcus","MND1","Dongia","RB41","Skermanella","Bryobacter","Rhodococcus","Devosia"],
      "d":["Rhodospirillales","Bacteroides","Bifidobacterium","Monoglobus","Bifidobacterium breve","Eubacterium hallii","Collinsella","Catenibacterium","Vicinamibacteraceae"]},
 1:  {"e":["Ruminiclostridium","Gordonibacter","Enorma","Lawsonella","Frisingicoccus","Anaerofilum"],
      "d":["Catabacter","Howardella","Marine_Methylotrophic_Group_3","Lachnospiraceae_AC2044_group"]},
 4:  {"e":["Proteobacteria","Actinobacteriota","Faecalibacterium","Escherichia-Shigella","Subdoligranulum","Enterobacteriaceae-unclassified","Halomonas","Bifidobacterium","Gammaproteobacteria","Alphaproteobacteria","Actinobacteria","Enterobacteriales","Bifidobacteriales","Clostridia-unclassified","Bifidobacteriaceae"],
      "d":["Bacteroidota","Bacteroides","Megamonas","Prevotella","Lachnospiraceae-unclassified","Blautia","Parabacteroides","Collinsella","Lachnospira","Bacteroidia","Negativicutes","Bacilli","Coriobacteriia","Lachnospirales","Bacteroidales","Rikenellaceae"]},
 10: {"e":["Clostridium cluster IV","Akkermansia","Bifidobacterium","lactic acid bacteria"],
      "d":["Firmicutes"]},
 11: {"e":[], "d":[]},
}
json.dump({str(k): {"taxa_enriched": v["e"], "taxa_depleted": v["d"]}
           for k, v in grounded.items()}, open(os.path.join(C, "grounded.json"), "w"), indent=2)

# ---------- ORIGINAL (single-shot samgated) — reuse the earlier Opus 4.8 extraction ----------
orig_full = {
 0:  {"e":["Phyllobacterium","Bacteroidia","Lactobacillus salivarius","Coriobacteriia","Erysipelotrichia","Gemmatimonadetes","Verrucomicrobiae","Synergistia","Cloacibacillus","Clostridia"],
      "d":["Rhodospirillales","Bacteroides","Bifidobacterium","Monoglobus","Bifidobacterium breve"]},
 1:  {"e":["Ruminiclostridium","Gordonibacter","Enorma","Lawsonella","Frisingicoccus","Anaerofilum"],
      "d":["Catabacter","Howardella","Marine_Methylotrophic_Group_3","Lachnospiraceae_AC2044_group"]},
 4:  {"e":["Proteobacteria","Actinobacteriota","Gammaproteobacteria","Alphaproteobacteria","Actinobacteria","Enterobacteriales","Bifidobacteriales","Clostridia-unclassified","Bifidobacteriaceae","Faecalibacterium","Escherichia-Shigella","Subdoligranulum","Enterobacteriaceae-unclassified","Halomonas","Bifidobacterium"],
      "d":["Bacteroidota","Bacteroidia","Negativicutes","Bacilli","Coriobacteriia","Lachnospirales","Bacteroidales","Rikenellaceae","Bacteroides","Megamonas","Prevotella","Lachnospiraceae-unclassified","Blautia","Lachnospira"]},
 10: {"e":["Clostridium cluster IV","Akkermansia","Bifidobacterium","lactic acid bacteria"], "d":["Firmicutes"]},
 11: {"e":[], "d":[]},
}
json.dump({str(k): {"taxa_enriched": v["e"], "taxa_depleted": v["d"]}
           for k, v in orig_full.items()}, open(os.path.join(C, "original.json"), "w"), indent=2)
print("wrote cache: original.json, relate.json, grounded.json")
